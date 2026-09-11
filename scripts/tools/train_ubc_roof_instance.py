"""Train and evaluate frozen-OlmoEarth instance segmentation on UBC v2.

The script supports the two roof-instance tasks present in the released dataset:

* ``single``: RGB imagery.
* ``multimodal``: aligned RGB and single-channel SAR imagery.

Images are trained and inferred as overlapping tiles. Predictions are restored to
the original 512 x 512 image coordinates, fused with mask NMS, and exported as a
standard COCO result JSON. The OlmoEarth encoder is always frozen; only the
feature pyramid and Mask R-CNN heads are optimized.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Literal

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from tqdm import tqdm

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import load_computed_config
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import load_model_from_path


TaskName = Literal["single", "multimodal"]
NUM_CLASSES = 13  # background + the 12 UBC roof categories
S2_DN_MAX = 10_000.0


def _coco_mask_module() -> Any:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pycocotools is required. Install the project's training/eval extras."
        ) from exc
    return mask_utils


@dataclass(frozen=True)
class SplitPaths:
    json_path: Path
    rgb_dir: Path
    sar_dir: Path | None


def resolve_split_paths(root: Path, task: TaskName, split: str) -> SplitPaths:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split: {split}")
    if task == "single":
        if split == "test":
            base = root / "test_set" / "single_modal_test"
            return SplitPaths(base / "roof_fine_12_test.json", base / "test", None)
        base = root / "fine-grained_building_roof_instance_segmentation"
        return SplitPaths(
            base / "annotations" / f"roof_fine_12_{split}.json",
            base / split,
            None,
        )
    if task == "multimodal":
        if split == "test":
            base = root / "test_set" / "multi_modal_test"
            return SplitPaths(
                base / "roof_fine_test.json", base / "test" / "rgb", base / "test" / "sar"
            )
        base = root / "multi-modal_fine-grained_building_roof_instance_segmentation"
        return SplitPaths(
            base / "annotations" / f"roof_fine_{split}.json",
            base / split / "rgb",
            base / split / "sar",
        )
    raise ValueError(f"unsupported task: {task}")


class CocoIndex:
    """Small COCO index that keeps the original image/category IDs unchanged."""

    def __init__(self, paths: SplitPaths) -> None:
        for path in (paths.json_path, paths.rgb_dir):
            if not path.exists():
                raise FileNotFoundError(path)
        if paths.sar_dir is not None and not paths.sar_dir.is_dir():
            raise FileNotFoundError(paths.sar_dir)
        self.paths = paths
        self.data = json.loads(paths.json_path.read_text(encoding="utf-8"))
        self.images = sorted(self.data["images"], key=lambda row: int(row["id"]))
        self.image_by_id = {int(row["id"]): row for row in self.images}
        self.annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in self.data["annotations"]:
            self.annotations_by_image[int(annotation["image_id"])].append(annotation)
        self.categories = {
            int(row["id"]): str(row["name"]) for row in self.data["categories"]
        }
        expected = set(range(1, NUM_CLASSES))
        if set(self.categories) != expected:
            raise ValueError(
                f"expected UBC category IDs 1..12, got {sorted(self.categories)}"
            )
        for row in self.images:
            name = Path(str(row["file_name"])).name
            if not (paths.rgb_dir / name).is_file():
                raise FileNotFoundError(paths.rgb_dir / name)
            if paths.sar_dir is not None and not (paths.sar_dir / name).is_file():
                raise FileNotFoundError(paths.sar_dir / name)

    def load_image(self, row: dict[str, Any]) -> np.ndarray:
        name = Path(str(row["file_name"])).name
        with Image.open(self.paths.rgb_dir / name) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        if rgb.shape[:2] != (int(row["height"]), int(row["width"])):
            raise ValueError(f"image size differs from JSON for {name}: {rgb.shape}")
        if self.paths.sar_dir is None:
            return rgb
        with Image.open(self.paths.sar_dir / name) as image:
            sar = np.asarray(image, dtype=np.float32).copy()
        if sar.ndim == 3:
            sar = sar[..., 0]
        if sar.shape != rgb.shape[:2]:
            raise ValueError(f"RGB/SAR shape mismatch for {name}: {rgb.shape} vs {sar.shape}")
        return np.concatenate([rgb, sar[..., None]], axis=-1)

    @staticmethod
    def decode_mask(annotation: dict[str, Any], height: int, width: int) -> np.ndarray:
        mask_utils = _coco_mask_module()
        segmentation = annotation["segmentation"]
        if isinstance(segmentation, list):
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        elif isinstance(segmentation, dict):
            rle = segmentation
            if isinstance(rle.get("counts"), list):
                rle = mask_utils.frPyObjects(rle, height, width)
        else:
            raise TypeError(f"unsupported COCO segmentation: {type(segmentation)}")
        mask = mask_utils.decode(rle)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2)
        return np.asarray(mask, dtype=np.uint8)


def _boxes_from_masks(masks: torch.Tensor) -> torch.Tensor:
    boxes: list[list[float]] = []
    for mask in masks:
        ys, xs = torch.where(mask > 0)
        if xs.numel() == 0:
            boxes.append([0.0, 0.0, 1.0, 1.0])
        else:
            # Torchvision boxes use an exclusive maximum corner.
            boxes.append(
                [
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max() + 1),
                    float(ys.max() + 1),
                ]
            )
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)


class UBCTileDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Object-aware random 128-pixel crops for Mask R-CNN training."""

    def __init__(
        self,
        index: CocoIndex,
        tile_size: int,
        samples_per_image: int,
        object_crop_probability: float,
        min_visible_fraction: float,
        min_mask_area: int,
        augment: bool,
        max_images: int | None = None,
    ) -> None:
        self.index = index
        self.images = (
            index.images[:max_images]
            if max_images is not None and max_images > 0
            else index.images
        )
        self.tile_size = tile_size
        self.samples_per_image = samples_per_image
        self.object_crop_probability = object_crop_probability
        self.min_visible_fraction = min_visible_fraction
        self.min_mask_area = min_mask_area
        self.augment = augment
        counts = Counter(
            int(ann["category_id"])
            for anns in index.annotations_by_image.values()
            for ann in anns
        )
        self.class_weights = {key: 1.0 / math.sqrt(value) for key, value in counts.items()}

    def __len__(self) -> int:
        return len(self.images) * self.samples_per_image

    def _crop_origin(self, row: dict[str, Any]) -> tuple[int, int]:
        width, height = int(row["width"]), int(row["height"])
        max_x, max_y = max(0, width - self.tile_size), max(0, height - self.tile_size)
        anns = self.index.annotations_by_image[int(row["id"])]
        if anns and random.random() < self.object_crop_probability:
            weights = [self.class_weights[int(ann["category_id"])] for ann in anns]
            ann = random.choices(anns, weights=weights, k=1)[0]
            x, y, w, h = (float(value) for value in ann["bbox"])
            jitter = self.tile_size * 0.2
            cx = x + w / 2 + random.uniform(-jitter, jitter)
            cy = y + h / 2 + random.uniform(-jitter, jitter)
            x0 = round(cx - self.tile_size / 2)
            y0 = round(cy - self.tile_size / 2)
            return min(max(x0, 0), max_x), min(max(y0, 0), max_y)
        return random.randint(0, max_x), random.randint(0, max_y)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row = self.images[item % len(self.images)]
        image = self.index.load_image(row)
        height, width = image.shape[:2]
        x0, y0 = self._crop_origin(row)
        x1, y1 = min(width, x0 + self.tile_size), min(height, y0 + self.tile_size)
        crop = image[y0:y1, x0:x1]
        if crop.shape[:2] != (self.tile_size, self.tile_size):
            pad_h, pad_w = self.tile_size - crop.shape[0], self.tile_size - crop.shape[1]
            crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        masks: list[np.ndarray] = []
        labels: list[int] = []
        original_areas: list[float] = []
        for ann in self.index.annotations_by_image[int(row["id"])]:
            ann_x, ann_y, ann_w, ann_h = (float(value) for value in ann["bbox"])
            if (
                ann_x + ann_w <= x0
                or ann_y + ann_h <= y0
                or ann_x >= x1
                or ann_y >= y1
            ):
                continue
            full_mask = self.index.decode_mask(ann, height, width)
            local = full_mask[y0:y1, x0:x1]
            if local.shape != (self.tile_size, self.tile_size):
                local = np.pad(
                    local,
                    ((0, self.tile_size - local.shape[0]), (0, self.tile_size - local.shape[1])),
                    mode="constant",
                )
            local_area = int(local.sum())
            full_area = max(float(ann.get("area", full_mask.sum())), 1.0)
            if local_area < self.min_mask_area:
                continue
            if local_area / full_area < self.min_visible_fraction:
                continue
            masks.append(local)
            labels.append(int(ann["category_id"]))
            original_areas.append(float(local_area))

        tensor = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))).float()
        if masks:
            mask_tensor = torch.from_numpy(np.stack(masks)).to(torch.uint8)
        else:
            mask_tensor = torch.zeros((0, self.tile_size, self.tile_size), dtype=torch.uint8)

        if self.augment:
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=(2,))
                mask_tensor = torch.flip(mask_tensor, dims=(2,))
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=(1,))
                mask_tensor = torch.flip(mask_tensor, dims=(1,))
            rotations = random.randrange(4)
            if rotations:
                tensor = torch.rot90(tensor, rotations, dims=(1, 2))
                mask_tensor = torch.rot90(mask_tensor, rotations, dims=(1, 2))
            # Photometric jitter is RGB-only; SAR alignment remains unchanged.
            if random.random() < 0.4:
                gain = random.uniform(0.85, 1.15)
                bias = random.uniform(-12.0, 12.0)
                tensor[:3] = (tensor[:3] * gain + bias).clamp(0, 255)

        target = {
            "boxes": _boxes_from_masks(mask_tensor),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": mask_tensor,
            "image_id": torch.tensor([int(row["id"])], dtype=torch.int64),
            "area": torch.tensor(original_areas, dtype=torch.float32),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        return tensor.contiguous(), target


def collate_detection(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    return [row[0] for row in batch], [row[1] for row in batch]


def _normalization_bounds(modality: Any) -> tuple[torch.Tensor, torch.Tensor]:
    config = load_computed_config()[modality.name]
    means = torch.tensor([config[band]["mean"] for band in modality.band_order])
    stds = torch.tensor([config[band]["std"] for band in modality.band_order])
    return means - 2 * stds, means + 2 * stds


class OlmoEarthPyramid(nn.Module):
    """Frozen OlmoEarth encoder plus a trainable four-level feature pyramid."""

    out_channels = 256

    def __init__(
        self,
        olmo: nn.Module,
        task: TaskName,
        patch_size: int,
        embedding_size: int,
    ) -> None:
        super().__init__()
        self.olmo = olmo
        self.task = task
        self.patch_size = patch_size
        for parameter in self.olmo.parameters():
            parameter.requires_grad = False
        self.olmo.eval()

        s2_low, s2_high = _normalization_bounds(Modality.SENTINEL2_L2A)
        s1_low, s1_high = _normalization_bounds(Modality.SENTINEL1)
        self.register_buffer("s2_low", s2_low.reshape(1, 1, 1, 1, -1))
        self.register_buffer("s2_range", (s2_high - s2_low).reshape(1, 1, 1, 1, -1))
        self.register_buffer("s1_low", s1_low.reshape(1, 1, 1, 1, -1))
        self.register_buffer("s1_range", (s1_high - s1_low).reshape(1, 1, 1, 1, -1))

        in_channels = embedding_size * (2 if task == "multimodal" else 1)
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1),
            nn.GroupNorm(32, self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
        )
        self.down3 = self._downsample()
        self.down4 = self._downsample()
        self.down5 = self._downsample()

    @classmethod
    def _downsample(cls) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cls.out_channels, cls.out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, cls.out_channels),
            nn.GELU(),
        )

    def train(self, mode: bool = True) -> "OlmoEarthPyramid":
        super().train(mode)
        self.olmo.eval()
        return self

    def _sample(self, images: torch.Tensor) -> MaskedOlmoEarthSample:
        batch, _, height, width = images.shape
        rgb = images[:, :3]
        s2 = torch.zeros(
            (batch, height, width, 1, 12), device=images.device, dtype=images.dtype
        )
        # PIL gives R,G,B. OlmoEarth expects B02,B03,B04 at indices 0,1,2.
        s2[..., 0, 0] = rgb[:, 2] * (S2_DN_MAX / 255.0)
        s2[..., 0, 1] = rgb[:, 1] * (S2_DN_MAX / 255.0)
        s2[..., 0, 2] = rgb[:, 0] * (S2_DN_MAX / 255.0)
        s2 = (s2 - self.s2_low) / self.s2_range.clamp_min(1e-6)
        s2[..., 3:] = 0  # restore padded channels after normalization
        online = float(MaskValue.ONLINE_ENCODER.value)
        kwargs: dict[str, Any] = {
            "sentinel2_l2a": s2,
            "sentinel2_l2a_mask": torch.full(
                (batch, height, width, 1, 1), online, device=images.device, dtype=images.dtype
            ),
            "timestamps": torch.tensor(
                [[[15, 6, 2020]]], device=images.device, dtype=torch.long
            ).expand(batch, -1, -1),
        }
        if self.task == "multimodal":
            sar = images[:, 3].clamp_min(1e-6)
            sar_db = 10.0 * torch.log10(sar)
            s1 = torch.zeros(
                (batch, height, width, 1, 2), device=images.device, dtype=images.dtype
            )
            s1[..., 0, 0] = sar_db
            s1 = (s1 - self.s1_low) / self.s1_range.clamp_min(1e-6)
            s1[..., 1] = 0  # unknown second polarization is treated as dropped
            kwargs.update(
                sentinel1=s1,
                sentinel1_mask=torch.full(
                    (batch, height, width, 1, 1),
                    online,
                    device=images.device,
                    dtype=images.dtype,
                ),
            )
        return MaskedOlmoEarthSample(**kwargs)

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        if images.shape[-1] % self.patch_size or images.shape[-2] % self.patch_size:
            raise ValueError("tile dimensions must be divisible by --patch-size")
        with torch.no_grad():
            encoded = self.olmo.encoder(
                self._sample(images), fast_pass=True, patch_size=self.patch_size
            )["tokens_and_masks"]
            s2 = encoded.sentinel2_l2a.mean(dim=(3, 4)).permute(0, 3, 1, 2)
            features = [s2]
            if self.task == "multimodal":
                s1 = encoded.sentinel1.mean(dim=(3, 4)).permute(0, 3, 1, 2)
                features.append(s1)
            frozen = torch.cat(features, dim=1).float()
        p2 = self.p2(frozen)
        p3 = self.down3(p2)
        p4 = self.down4(p3)
        p5 = self.down5(p4)
        return OrderedDict((str(i), value) for i, value in enumerate((p2, p3, p4, p5)))


def build_model(
    weights: Path,
    task: TaskName,
    tile_size: int,
    patch_size: int,
    device: torch.device,
) -> MaskRCNN:
    print(f"loading frozen OlmoEarth backbone from {weights}")
    olmo = load_model_from_path(weights).to(device)
    config = json.loads((weights / "config.json").read_text(encoding="utf-8"))
    embedding_size = int(config["model"]["encoder_config"]["embedding_size"])
    backbone = OlmoEarthPyramid(olmo, task, patch_size, embedding_size)
    feature_names = ["0", "1", "2", "3"]
    anchors = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,)),
        aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 4,
    )
    model = MaskRCNN(
        backbone,
        num_classes=NUM_CLASSES,
        min_size=tile_size,
        max_size=tile_size,
        image_mean=[0.0] * (4 if task == "multimodal" else 3),
        image_std=[1.0] * (4 if task == "multimodal" else 3),
        rpn_anchor_generator=anchors,
        box_roi_pool=MultiScaleRoIAlign(feature_names, output_size=7, sampling_ratio=2),
        mask_roi_pool=MultiScaleRoIAlign(feature_names, output_size=14, sampling_ratio=2),
        rpn_pre_nms_top_n_train=4000,
        rpn_pre_nms_top_n_test=2000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        box_detections_per_img=500,
        box_score_thresh=0.03,
    )
    return model.to(device)


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("backbone.olmo.")
    }


def load_trainable_state(model: nn.Module, checkpoint: dict[str, Any]) -> None:
    source = checkpoint.get("model", checkpoint)
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in current and current[key].shape == value.shape
    }
    skipped = sorted(set(source) - set(compatible))
    result = model.load_state_dict(compatible, strict=False)
    unexpected = [key for key in result.unexpected_keys if not key.startswith("backbone.olmo.")]
    missing = [key for key in result.missing_keys if not key.startswith("backbone.olmo.")]
    print(
        f"loaded {len(compatible)} trainable tensors; "
        f"shape/missing skipped={len(skipped)}, target-only={len(missing)}"
    )
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:10]}")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float] | None,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "task": args.task,
            "tile_size": args.tile_size,
            "patch_size": args.patch_size,
            "model": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


@dataclass
class Candidate:
    label: int
    score: float
    mask: np.ndarray
    bbox: tuple[float, float, float, float]


def sliding_origins(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _bbox_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-8)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = int(np.logical_and(a, b).sum())
    if intersection == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return intersection / max(union, 1)


def _mask_containment(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over the smaller mask, useful for clipped tile duplicates."""
    intersection = int(np.logical_and(a, b).sum())
    if intersection == 0:
        return 0.0
    return intersection / max(min(int(a.sum()), int(b.sum())), 1)


def mask_nms(
    candidates: list[Candidate],
    same_class_threshold: float,
    cross_class_threshold: float,
    containment_threshold: float,
    max_instances: int,
) -> list[Candidate]:
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda row: row.score, reverse=True):
        duplicate = False
        for previous in kept:
            threshold = (
                same_class_threshold
                if candidate.label == previous.label
                else cross_class_threshold
            )
            if _bbox_iou(candidate.bbox, previous.bbox) < 0.05:
                continue
            if (
                _mask_iou(candidate.mask, previous.mask) >= threshold
                or _mask_containment(candidate.mask, previous.mask)
                >= containment_threshold
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
            if len(kept) >= max_instances:
                break
    return kept


def _tile_tensor(image: np.ndarray, x0: int, y0: int, tile_size: int) -> torch.Tensor:
    tile = image[y0 : y0 + tile_size, x0 : x0 + tile_size]
    pad_h, pad_w = tile_size - tile.shape[0], tile_size - tile.shape[1]
    if pad_h or pad_w:
        tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1))).float()


@torch.no_grad()
def predict_image(
    model: MaskRCNN,
    image: np.ndarray,
    device: torch.device,
    tile_size: int,
    stride: int,
    tile_batch_size: int,
    score_threshold: float,
    mask_threshold: float,
    same_class_nms: float,
    cross_class_nms: float,
    containment_nms: float,
    max_instances: int,
    amp: bool,
) -> list[Candidate]:
    height, width = image.shape[:2]
    jobs = [
        (x0, y0)
        for y0 in sliding_origins(height, tile_size, stride)
        for x0 in sliding_origins(width, tile_size, stride)
    ]
    candidates: list[Candidate] = []
    for start in range(0, len(jobs), tile_batch_size):
        batch_jobs = jobs[start : start + tile_batch_size]
        tiles = [_tile_tensor(image, x, y, tile_size).to(device) for x, y in batch_jobs]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            outputs = model(tiles)
        for (x0, y0), output in zip(batch_jobs, outputs, strict=True):
            scores = output["scores"].detach().float().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            boxes = output["boxes"].detach().float().cpu().numpy()
            masks = output["masks"].detach().float().cpu().numpy()[:, 0]
            for score, label, box, local_probability in zip(
                scores, labels, boxes, masks, strict=True
            ):
                if float(score) < score_threshold:
                    continue
                local = local_probability >= mask_threshold
                valid_h, valid_w = min(tile_size, height - y0), min(tile_size, width - x0)
                local = local[:valid_h, :valid_w]
                if not local.any():
                    continue
                full = np.zeros((height, width), dtype=bool)
                full[y0 : y0 + valid_h, x0 : x0 + valid_w] = local
                bx1 = min(max(float(box[0]) + x0, 0.0), float(width))
                by1 = min(max(float(box[1]) + y0, 0.0), float(height))
                bx2 = min(max(float(box[2]) + x0, 0.0), float(width))
                by2 = min(max(float(box[3]) + y0, 0.0), float(height))
                candidates.append(
                    Candidate(int(label), float(score), full, (bx1, by1, bx2, by2))
                )
    return mask_nms(
        candidates,
        same_class_nms,
        cross_class_nms,
        containment_nms,
        max_instances,
    )


def candidate_to_coco(candidate: Candidate, image_id: int) -> dict[str, Any]:
    mask_utils = _coco_mask_module()
    rle = mask_utils.encode(np.asfortranarray(candidate.mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    bbox = mask_utils.toBbox(rle).astype(float).tolist()
    return {
        "image_id": image_id,
        "category_id": candidate.label,
        "segmentation": rle,
        "score": candidate.score,
        "bbox": bbox,
    }


def select_eval_images(
    images: list[dict[str, Any]], max_images: int | None, seed: int
) -> list[dict[str, Any]]:
    if max_images is None or max_images <= 0 or max_images >= len(images):
        return images
    rng = random.Random(seed)
    selected_ids = {int(row["id"]) for row in rng.sample(images, max_images)}
    return [row for row in images if int(row["id"]) in selected_ids]


@torch.no_grad()
def predict_split(
    model: MaskRCNN,
    index: CocoIndex,
    args: argparse.Namespace,
    device: torch.device,
    max_images: int | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    model.eval()
    rows = select_eval_images(index.images, max_images, args.seed)
    results: list[dict[str, Any]] = []
    image_ids: list[int] = []
    for row in tqdm(rows, desc="sliding-window inference"):
        image_id = int(row["id"])
        image_ids.append(image_id)
        image = index.load_image(row)
        candidates = predict_image(
            model,
            image,
            device,
            args.tile_size,
            args.stride,
            args.tile_batch_size,
            args.score_threshold,
            args.mask_threshold,
            args.same_class_nms,
            args.cross_class_nms,
            args.containment_nms,
            args.max_instances,
            args.amp,
        )
        results.extend(candidate_to_coco(candidate, image_id) for candidate in candidates)
    return results, image_ids


def evaluate_coco(
    ground_truth_path: Path,
    predictions: list[dict[str, Any]],
    image_ids: list[int] | None = None,
) -> dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pycocotools is required for COCO evaluation") from exc
    if not predictions:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    coco_gt = COCO(str(ground_truth_path))
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "segm")
    if image_ids is not None:
        evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = {
        "AP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "AP_small": float(evaluator.stats[3]),
        "AP_medium": float(evaluator.stats[4]),
        "AP_large": float(evaluator.stats[5]),
    }
    precision = evaluator.eval["precision"]
    iou_index = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.5)))
    for category_index, category_id in enumerate(evaluator.params.catIds):
        values = precision[iou_index, :, category_index, 0, -1]
        values = values[values > -1]
        name = coco_gt.cats[int(category_id)]["name"]
        metrics[f"AP50_{name}"] = float(values.mean()) if values.size else 0.0
    return metrics


def _checkpoint_args(checkpoint: dict[str, Any], args: argparse.Namespace) -> None:
    for name in ("task", "tile_size", "patch_size"):
        expected = checkpoint.get(name)
        actual = getattr(args, name)
        if expected is not None and expected != actual:
            raise ValueError(
                f"checkpoint {name}={expected!r}, but command uses {actual!r}"
            )


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device or default_device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_paths = resolve_split_paths(Path(args.data_root), args.task, "train")
    val_paths = resolve_split_paths(Path(args.data_root), args.task, "val")
    train_index, val_index = CocoIndex(train_paths), CocoIndex(val_paths)
    dataset = UBCTileDataset(
        train_index,
        args.tile_size,
        args.samples_per_image,
        args.object_crop_probability,
        args.min_visible_fraction,
        args.min_mask_area,
        augment=True,
        max_images=args.max_train_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=collate_detection,
    )
    model = build_model(
        Path(args.weights), args.task, args.tile_size, args.patch_size, device
    )
    if args.init_ckpt:
        checkpoint = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        load_trainable_state(model, checkpoint)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start_epoch = 1
    best_ap50 = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        _checkpoint_args(checkpoint, args)
        load_trainable_state(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if checkpoint.get("metrics"):
            best_ap50 = float(checkpoint["metrics"].get("AP50", -1.0))
    trainable = sum(parameter.numel() for parameter in parameters)
    frozen = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    print(
        f"task={args.task} device={device} train_images={len(dataset.images)} "
        f"val_images={len(val_index.images)} tiles/epoch={len(dataset)}"
    )
    print(f"parameters: trainable={trainable:,}, frozen={frozen:,}")
    history: list[dict[str, Any]] = []

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals: Counter[str] = Counter()
        batches = 0
        started = time.time()
        for step, (images, targets) in enumerate(
            tqdm(loader, desc=f"epoch {epoch}/{args.epochs}"), start=1
        ):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {key: value.to(device, non_blocking=True) for key, value in target.items()}
                for target in targets
            ]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=args.amp and device.type == "cuda",
            ):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values()) / args.accum_steps
            scaler.scale(loss).backward()
            if step % args.accum_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            for name, value in loss_dict.items():
                totals[name] += float(value.detach())
            totals["loss"] += float(sum(loss_dict.values()).detach())
            batches += 1
        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            "seconds": time.time() - started,
            "lr": scheduler.get_last_lr()[0],
            **{name: value / max(batches, 1) for name, value in totals.items()},
        }
        metrics: dict[str, float] | None = None
        should_eval = epoch % args.eval_every == 0 or epoch == args.epochs
        if should_eval:
            predictions, image_ids = predict_split(
                model, val_index, args, device, max_images=args.val_max_images
            )
            prediction_path = out_dir / f"val_epoch_{epoch:03d}.json"
            prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
            metrics = evaluate_coco(val_paths.json_path, predictions, image_ids)
            row.update({f"val_{key}": value for key, value in metrics.items()})
            print(
                f"epoch={epoch} loss={row['loss']:.4f} "
                f"val_AP={metrics['AP']:.4f} val_AP50={metrics['AP50']:.4f}"
            )
        else:
            print(f"epoch={epoch} loss={row['loss']:.4f}")
        history.append(row)
        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, epoch, args, metrics)
        if metrics is not None and metrics["AP50"] > best_ap50:
            best_ap50 = metrics["AP50"]
            save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, epoch, args, metrics)
            print(f"saved best.pt (AP50={best_ap50:.4f})")
        (out_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"training complete: {out_dir}")


def load_for_inference(
    args: argparse.Namespace, device: torch.device
) -> tuple[MaskRCNN, dict[str, Any]]:
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    _checkpoint_args(checkpoint, args)
    model = build_model(
        Path(args.weights), args.task, args.tile_size, args.patch_size, device
    )
    load_trainable_state(model, checkpoint)
    model.eval()
    return model, checkpoint


def evaluate_command(args: argparse.Namespace) -> None:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device or default_device)
    paths = resolve_split_paths(Path(args.data_root), args.task, args.split)
    index = CocoIndex(paths)
    model, _ = load_for_inference(args, device)
    predictions, image_ids = predict_split(
        model, index, args, device, max_images=args.max_images
    )
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(predictions), encoding="utf-8")
    metrics = evaluate_coco(paths.json_path, predictions, image_ids)
    metrics_path = out_json.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"predictions: {out_json}")
    print(f"metrics: {metrics_path}")


def predict_command(args: argparse.Namespace) -> None:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device or default_device)
    paths = resolve_split_paths(Path(args.data_root), args.task, args.split)
    index = CocoIndex(paths)
    model, _ = load_for_inference(args, device)
    predictions, _ = predict_split(model, index, args, device, max_images=args.max_images)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(predictions), encoding="utf-8")
    print(f"wrote {len(predictions)} instances to {out_json}")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", required=True, help="UBC_v2.0 directory")
    parser.add_argument("--weights", required=True, help="OlmoEarth checkpoint directory")
    parser.add_argument("--task", choices=("single", "multimodal"), required=True)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--device", default=None, help="Default: cuda when available")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True, help="CUDA bfloat16 autocast"
    )


def add_inference(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--same-class-nms", type=float, default=0.5)
    parser.add_argument("--cross-class-nms", type=float, default=0.7)
    parser.add_argument(
        "--containment-nms",
        type=float,
        default=0.85,
        help="suppress a clipped mask when it is mostly contained in another",
    )
    parser.add_argument("--max-instances", type=int, default=500)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train frozen-backbone Mask R-CNN")
    add_common(train_parser)
    add_inference(train_parser)
    train_parser.add_argument("--out-dir", required=True)
    train_parser.add_argument("--epochs", type=int, default=24)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--accum-steps", type=int, default=2)
    train_parser.add_argument("--workers", type=int, default=4)
    train_parser.add_argument("--samples-per-image", type=int, default=2)
    train_parser.add_argument(
        "--max-train-images",
        type=int,
        default=0,
        help="0 uses all images; set a small number only for smoke tests",
    )
    train_parser.add_argument("--object-crop-probability", type=float, default=0.85)
    train_parser.add_argument("--min-visible-fraction", type=float, default=0.25)
    train_parser.add_argument("--min-mask-area", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    train_parser.add_argument("--eval-every", type=int, default=4)
    train_parser.add_argument(
        "--val-max-images",
        type=int,
        default=0,
        help="0 evaluates the full val split (default); positive values are smoke tests only",
    )
    train_parser.add_argument("--resume", default=None)
    train_parser.add_argument(
        "--init-ckpt",
        default=None,
        help="warm-start compatible heads (e.g. multimodal from a single checkpoint)",
    )

    for name in ("eval", "predict"):
        sub = commands.add_parser(name)
        add_common(sub)
        add_inference(sub)
        sub.add_argument("--ckpt", required=True)
        sub.add_argument("--split", choices=("train", "val", "test"), default="test")
        sub.add_argument("--out-json", required=True)
        sub.add_argument("--max-images", type=int, default=0, help="0 processes all images")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.tile_size <= 0 or args.tile_size % args.patch_size:
        raise ValueError("--tile-size must be positive and divisible by --patch-size")
    if hasattr(args, "stride") and not 0 < args.stride <= args.tile_size:
        raise ValueError("expected 0 < --stride <= --tile-size")
    probability_names = (
        "score_threshold",
        "mask_threshold",
        "same_class_nms",
        "cross_class_nms",
        "containment_nms",
    )
    for name in probability_names:
        if hasattr(args, name) and not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.command == "train":
        if args.accum_steps < 1 or args.batch_size < 1 or args.samples_per_image < 1:
            raise ValueError("batch, accumulation, and sampling counts must be positive")
        if args.max_train_images < 0 or args.val_max_images < 0:
            raise ValueError("image limits must be non-negative")
        if args.eval_every < 1:
            raise ValueError("--eval-every must be positive")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate_command(args)
    else:
        predict_command(args)


if __name__ == "__main__":
    main()
