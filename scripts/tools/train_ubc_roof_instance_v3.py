"""UBC roof instance segmentation v3: Cascade + Seesaw + multi-GPU.

Extends the v2 neck with competition-proven detection heads:

* Cascade Mask R-CNN ROI heads (IoU 0.5 / 0.6 / 0.7)
* Seesaw classification loss for long-tailed roof types
* ``torchrun`` / DDP multi-GPU training
* Copy-Paste for rare classes
* Multi-scale sliding-window evaluation
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Literal

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, batched_nms
from tqdm import tqdm

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import load_computed_config
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import load_model_from_path

import sys

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
from ubc_cascade_seesaw import attach_cascade_seesaw, collect_class_counts  # noqa: E402


TaskName = Literal["single", "multimodal"]
S2RgbMode = Literal["rgb-only", "repeat"]
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
    """Per-image and globally class-balanced object-aware training crops."""

    def __init__(
        self,
        index: CocoIndex,
        tile_size: int,
        samples_per_image: int,
        object_crop_probability: float,
        min_visible_fraction: float,
        min_mask_area: int,
        balanced_extra_samples: int,
        augment: bool,
        max_images: int | None = None,
        copy_paste_probability: float = 0.0,
        require_instances: bool = False,
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
        self.balanced_extra_samples = balanced_extra_samples
        self.augment = augment
        self.copy_paste_probability = copy_paste_probability
        self.require_instances = require_instances
        self.annotated_image_ids = [
            int(row["id"])
            for row in self.images
            if index.annotations_by_image[int(row["id"])]
        ]
        counts = Counter(
            int(ann["category_id"])
            for anns in index.annotations_by_image.values()
            for ann in anns
        )
        self.class_weights = {key: 1.0 / math.sqrt(value) for key, value in counts.items()}
        allowed_image_ids = {int(row["id"]) for row in self.images}
        annotations_by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for image_id in allowed_image_ids:
            for annotation in index.annotations_by_image[image_id]:
                annotations_by_category[int(annotation["category_id"])].append(annotation)
        self.annotations_by_category = {
            category_id: annotations
            for category_id, annotations in annotations_by_category.items()
            if annotations
        }
        self.balanced_categories = sorted(self.annotations_by_category)
        self.balanced_category_weights = [
            math.sqrt(len(self.annotations_by_category[category_id]))
            for category_id in self.balanced_categories
        ]
        # Prefer pasting rarer classes (higher 1/sqrt(count)).
        self.copy_paste_categories = sorted(
            self.annotations_by_category,
            key=lambda category_id: self.class_weights[category_id],
            reverse=True,
        )
        self.copy_paste_weights = [
            self.class_weights[category_id] for category_id in self.copy_paste_categories
        ]

    def __len__(self) -> int:
        return len(self.images) * (
            self.samples_per_image + self.balanced_extra_samples
        )

    def _origin_for_annotation(
        self, row: dict[str, Any], annotation: dict[str, Any]
    ) -> tuple[int, int]:
        width, height = int(row["width"]), int(row["height"])
        max_x, max_y = max(0, width - self.tile_size), max(0, height - self.tile_size)
        x, y, w, h = (float(value) for value in annotation["bbox"])

        def choose_axis(start: float, extent: float, maximum: int) -> int:
            if extent <= self.tile_size:
                # Choose a crop that fully contains the selected instance whenever
                # possible. Random placement within this interval retains context
                # diversity without teaching an avoidably truncated target.
                low = max(0, math.ceil(start + extent - self.tile_size))
                high = min(maximum, math.floor(start))
                if low <= high:
                    return random.randint(low, high)
            center = start + extent / 2
            return min(max(round(center - self.tile_size / 2), 0), maximum)

        return choose_axis(x, w, max_x), choose_axis(y, h, max_y)

    def _crop_origin(
        self, row: dict[str, Any], target_annotation: dict[str, Any] | None = None
    ) -> tuple[int, int]:
        width, height = int(row["width"]), int(row["height"])
        max_x, max_y = max(0, width - self.tile_size), max(0, height - self.tile_size)
        if target_annotation is not None:
            return self._origin_for_annotation(row, target_annotation)
        return random.randint(0, max_x), random.randint(0, max_y)

    def _copy_paste(
        self,
        crop: np.ndarray,
        masks: list[np.ndarray],
        labels: list[int],
        original_areas: list[float],
    ) -> tuple[np.ndarray, list[np.ndarray], list[int], list[float]]:
        if (
            self.copy_paste_probability <= 0
            or not self.copy_paste_categories
            or random.random() >= self.copy_paste_probability
        ):
            return crop, masks, labels, original_areas
        category_id = random.choices(
            self.copy_paste_categories, weights=self.copy_paste_weights, k=1
        )[0]
        annotation = random.choice(self.annotations_by_category[category_id])
        source_row = self.index.image_by_id[int(annotation["image_id"])]
        source = self.index.load_image(source_row)
        source_h, source_w = source.shape[:2]
        full_mask = self.index.decode_mask(annotation, source_h, source_w)
        ys, xs = np.nonzero(full_mask)
        if ys.size == 0:
            return crop, masks, labels, original_areas
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        patch_h, patch_w = y1 - y0, x1 - x0
        if patch_h >= self.tile_size or patch_w >= self.tile_size:
            return crop, masks, labels, original_areas
        max_y = self.tile_size - patch_h
        max_x = self.tile_size - patch_w
        if max_y < 0 or max_x < 0:
            return crop, masks, labels, original_areas
        ty = random.randint(0, max_y)
        tx = random.randint(0, max_x)
        patch_rgb = source[y0:y1, x0:x1]
        patch_mask = full_mask[y0:y1, x0:x1].astype(bool)
        if not patch_mask.any():
            return crop, masks, labels, original_areas
        crop = crop.copy()
        crop[ty : ty + patch_h, tx : tx + patch_w][patch_mask] = patch_rgb[patch_mask]
        pasted = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
        pasted[ty : ty + patch_h, tx : tx + patch_w] = patch_mask.astype(np.uint8)
        # Occlude existing masks where the pasted instance lands.
        for index, mask in enumerate(masks):
            masks[index] = (mask.astype(bool) & ~pasted.astype(bool)).astype(np.uint8)
        keep = [i for i, mask in enumerate(masks) if int(mask.sum()) >= self.min_mask_area]
        masks = [masks[i] for i in keep]
        labels = [labels[i] for i in keep]
        original_areas = [float(masks[i].sum()) for i in range(len(masks))]
        masks.append(pasted)
        labels.append(category_id)
        original_areas.append(float(pasted.sum()))
        return crop, masks, labels, original_areas

    def _extract(
        self, row: dict[str, Any], target_annotation: dict[str, Any] | None
    ) -> tuple[np.ndarray, list[np.ndarray], list[int], list[float]]:
        image = self.index.load_image(row)
        height, width = image.shape[:2]
        x0, y0 = self._crop_origin(row, target_annotation)
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
            is_focus_target = (
                target_annotation is not None
                and int(ann["id"]) == int(target_annotation["id"])
            )
            if local_area / full_area < self.min_visible_fraction and not is_focus_target:
                continue
            masks.append(local)
            labels.append(int(ann["category_id"]))
            original_areas.append(float(local_area))
        return crop, masks, labels, original_areas

    def _pick_row_and_target(
        self, item: int
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        sample_slot = item // len(self.images)
        row = self.images[item % len(self.images)]
        target_annotation: dict[str, Any] | None = None
        if (
            self.balanced_extra_samples > 0
            and sample_slot >= self.samples_per_image
            and self.balanced_categories
        ):
            category_id = random.choices(
                self.balanced_categories,
                weights=self.balanced_category_weights,
                k=1,
            )[0]
            target_annotation = random.choice(self.annotations_by_category[category_id])
            row = self.index.image_by_id[int(target_annotation["image_id"])]
        elif random.random() < self.object_crop_probability:
            annotations = self.index.annotations_by_image[int(row["id"])]
            if annotations:
                weights = [
                    self.class_weights[int(annotation["category_id"])]
                    for annotation in annotations
                ]
                target_annotation = random.choices(annotations, weights=weights, k=1)[0]
        return row, target_annotation

    def __getitem__(self, item: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row, target_annotation = self._pick_row_and_target(item)
        crop, masks, labels, original_areas = self._extract(row, target_annotation)
        # An empty crop makes the mask branch parameter-free for this rank only,
        # which deadlocks DDP's unused-parameter handshake. Force a target.
        retries = 0
        while (
            self.require_instances
            and not masks
            and retries < 8
            and self.annotated_image_ids
        ):
            retries += 1
            row = self.index.image_by_id[random.choice(self.annotated_image_ids)]
            target_annotation = random.choice(
                self.index.annotations_by_image[int(row["id"])]
            )
            crop, masks, labels, original_areas = self._extract(row, target_annotation)

        if self.augment:
            crop, masks, labels, original_areas = self._copy_paste(
                crop, masks, labels, original_areas
            )

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


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(32 if out_ch % 32 == 0 else 8, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CBAM(nn.Module):
    """Lightweight channel+spatial attention for the fused fine feature map."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        channel = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x = x * channel
        spatial = torch.sigmoid(
            self.spatial(
                torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
            )
        )
        return x * spatial


class OlmoEarthPyramid(nn.Module):
    """OlmoEarth encoder + detail skip + five-level SimpleFPN."""

    out_channels = 256

    def __init__(
        self,
        olmo: nn.Module,
        task: TaskName,
        patch_size: int,
        embedding_size: int,
        s2_rgb_mode: S2RgbMode,
        unfreeze_backbone: bool = False,
        use_detail_skip: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = olmo.encoder
        self.task = task
        self.patch_size = patch_size
        self.s2_rgb_mode = s2_rgb_mode
        self.unfreeze_backbone = unfreeze_backbone
        self.use_detail_skip = use_detail_skip
        for parameter in self.encoder.parameters():
            parameter.requires_grad = unfreeze_backbone
        if not unfreeze_backbone:
            self.encoder.eval()

        s2_low, s2_high = _normalization_bounds(Modality.SENTINEL2_L2A)
        s1_low, s1_high = _normalization_bounds(Modality.SENTINEL1)
        self.register_buffer("s2_low", s2_low.reshape(1, 1, 1, 1, -1))
        self.register_buffer("s2_range", (s2_high - s2_low).reshape(1, 1, 1, 1, -1))
        self.register_buffer("s1_low", s1_low.reshape(1, 1, 1, 1, -1))
        self.register_buffer("s1_range", (s1_high - s1_low).reshape(1, 1, 1, 1, -1))

        token_ch = embedding_size * (2 if task == "multimodal" else 1)
        self.stem = nn.Sequential(
            nn.Conv2d(token_ch, self.out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, self.out_channels),
            nn.GELU(),
        )
        # Build a five-level pyramid: stride = patch, 2p, 4p, 8p, 16p.
        self.down3 = ConvGNAct(self.out_channels, self.out_channels, stride=2)
        self.down4 = ConvGNAct(self.out_channels, self.out_channels, stride=2)
        self.down5 = ConvGNAct(self.out_channels, self.out_channels, stride=2)
        self.down6 = ConvGNAct(self.out_channels, self.out_channels, stride=2)
        self.lateral = nn.ModuleList(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
            for _ in range(5)
        )
        self.smooth = nn.ModuleList(
            ConvGNAct(self.out_channels, self.out_channels) for _ in range(5)
        )

        if use_detail_skip:
            detail_in = 4 if task == "multimodal" else 3
            # Stride down to patch resolution so RGB edges survive tokenization.
            blocks: list[nn.Module] = [ConvGNAct(detail_in, 64)]
            remaining = patch_size
            ch = 64
            while remaining > 1:
                blocks.append(ConvGNAct(ch, 128, stride=2))
                ch = 128
                remaining //= 2
            blocks.append(nn.Conv2d(ch, self.out_channels, kernel_size=1, bias=False))
            blocks.append(nn.GroupNorm(32, self.out_channels))
            blocks.append(nn.GELU())
            self.detail = nn.Sequential(*blocks)
            self.fuse = nn.Sequential(
                ConvGNAct(self.out_channels * 2, self.out_channels),
                CBAM(self.out_channels),
            )
        else:
            self.detail = None
            self.fuse = None

    def train(self, mode: bool = True) -> "OlmoEarthPyramid":
        super().train(mode)
        if not self.unfreeze_backbone:
            self.encoder.eval()
        return self

    def _sample(self, images: torch.Tensor) -> MaskedOlmoEarthSample:
        batch, _, height, width = images.shape
        rgb = images[:, :3]
        s2 = torch.zeros(
            (batch, height, width, 1, 12), device=images.device, dtype=images.dtype
        )
        bgr = rgb[:, [2, 1, 0]].permute(0, 2, 3, 1) * (S2_DN_MAX / 255.0)
        if self.s2_rgb_mode == "repeat":
            s2[..., 0, :] = bgr.repeat(1, 1, 1, 4)
        else:
            s2[..., 0, :3] = bgr
        s2 = (s2 - self.s2_low) / self.s2_range.clamp_min(1e-6)
        if self.s2_rgb_mode == "rgb-only":
            s2[..., 3:] = 0
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
            s1[..., 1] = 0
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
        context = nullcontext() if self.unfreeze_backbone else torch.no_grad()
        with context:
            encoded = self.encoder(
                self._sample(images), fast_pass=True, patch_size=self.patch_size
            )["tokens_and_masks"]
            s2 = encoded.sentinel2_l2a.mean(dim=(3, 4)).permute(0, 3, 1, 2)
            features = [s2]
            if self.task == "multimodal":
                s1 = encoded.sentinel1.mean(dim=(3, 4)).permute(0, 3, 1, 2)
                features.append(s1)
            tokens = torch.cat(features, dim=1).float()

        c2 = self.stem(tokens)
        if self.detail is not None and self.fuse is not None:
            detail = self.detail(images)
            if detail.shape[-2:] != c2.shape[-2:]:
                detail = F.interpolate(
                    detail, size=c2.shape[-2:], mode="bilinear", align_corners=False
                )
            c2 = self.fuse(torch.cat([c2, detail], dim=1))
        c3 = self.down3(c2)
        c4 = self.down4(c3)
        c5 = self.down5(c4)
        c6 = self.down6(c5)
        feats = [c2, c3, c4, c5, c6]
        # Top-down refinement: inject coarse semantics into finer maps.
        laterals = [lateral(feat) for lateral, feat in zip(self.lateral, feats)]
        for index in range(len(laterals) - 2, -1, -1):
            laterals[index] = laterals[index] + F.interpolate(
                laterals[index + 1],
                size=laterals[index].shape[-2:],
                mode="nearest",
            )
        pyramid = [smooth(feat) for smooth, feat in zip(self.smooth, laterals)]
        return OrderedDict((str(i), value) for i, value in enumerate(pyramid))


def build_model(
    weights: Path,
    task: TaskName,
    tile_size: int,
    patch_size: int,
    s2_rgb_mode: S2RgbMode,
    device: torch.device,
    unfreeze_backbone: bool = False,
    use_detail_skip: bool = True,
    use_cascade: bool = True,
    use_seesaw: bool = True,
    class_counts: list[int] | None = None,
) -> MaskRCNN:
    state = "fine-tunable" if unfreeze_backbone else "frozen"
    print(
        f"loading {state} OlmoEarth backbone from {weights} "
        f"(v3 neck; cascade={use_cascade}, seesaw={use_seesaw})"
    )
    olmo = load_model_from_path(weights)
    config = json.loads((weights / "config.json").read_text(encoding="utf-8"))
    embedding_size = int(config["model"]["encoder_config"]["embedding_size"])
    backbone = OlmoEarthPyramid(
        olmo,
        task,
        patch_size,
        embedding_size,
        s2_rgb_mode,
        unfreeze_backbone,
        use_detail_skip=use_detail_skip,
    )
    feature_names = ["0", "1", "2", "3", "4"]
    anchors = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 5,
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
        rpn_pre_nms_top_n_train=6000,
        rpn_pre_nms_top_n_test=2000,
        rpn_post_nms_top_n_train=3000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        box_detections_per_img=300,
        box_score_thresh=0.02,
        box_nms_thresh=0.5,
    )
    if use_cascade:
        attach_cascade_seesaw(
            model,
            class_counts=class_counts,
            use_seesaw=use_seesaw,
        )
    return model.to(device)


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    # A fine-tuned encoder no longer matches the pretrained weights directory, so
    # it must be part of the checkpoint.
    keep_encoder = model.backbone.unfreeze_backbone
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if keep_encoder or not key.startswith("backbone.encoder.")
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
    unexpected = [
        key
        for key in result.unexpected_keys
        if not key.startswith("backbone.encoder.")
    ]
    missing = [
        key for key in result.missing_keys if not key.startswith("backbone.encoder.")
    ]
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
    best_ap50: float | None = None,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "task": args.task,
            "tile_size": args.tile_size,
            "patch_size": args.patch_size,
            "s2_rgb_mode": args.s2_rgb_mode,
            "unfreeze_backbone": getattr(args, "unfreeze_backbone", False),
            "detail_skip": getattr(args, "detail_skip", True),
            "neck_version": "v3",
            "cascade": getattr(args, "cascade", True),
            "seesaw": getattr(args, "seesaw", True),
            "class_counts": getattr(args, "_class_counts", None),
            "copy_paste_probability": getattr(args, "copy_paste_probability", 0.0),
            "eval_scales": getattr(args, "eval_scales", "1.0"),
            "model": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "best_ap50": best_ap50,
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
    source_tiles: frozenset[int]
    touches_tile_boundary: bool


@dataclass
class RawCandidate:
    """Tile-local candidate retained without allocating a full-image mask."""

    label: int
    score: float
    local_mask: np.ndarray
    bbox: tuple[float, float, float, float]
    x0: int
    y0: int
    valid_h: int
    valid_w: int
    tile_id: int
    touches_tile_boundary: bool


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


def fast_box_nms(
    candidates: list[RawCandidate], iou_threshold: float, max_candidates: int
) -> list[RawCandidate]:
    """Cheap class-aware box NMS before full-resolution mask materialization."""
    if not candidates:
        return []
    boxes = torch.tensor([row.bbox for row in candidates], dtype=torch.float32)
    scores = torch.tensor([row.score for row in candidates], dtype=torch.float32)
    labels = torch.tensor([row.label for row in candidates], dtype=torch.int64)
    keep = batched_nms(boxes, scores, labels, iou_threshold).tolist()
    if max_candidates > 0:
        keep = keep[:max_candidates]
    return [candidates[index] for index in keep]


def materialize_candidate(candidate: RawCandidate, height: int, width: int) -> Candidate:
    full = np.zeros((height, width), dtype=bool)
    full[
        candidate.y0 : candidate.y0 + candidate.valid_h,
        candidate.x0 : candidate.x0 + candidate.valid_w,
    ] = candidate.local_mask
    return Candidate(
        label=candidate.label,
        score=candidate.score,
        mask=full,
        bbox=candidate.bbox,
        source_tiles=frozenset((candidate.tile_id,)),
        touches_tile_boundary=candidate.touches_tile_boundary,
    )


def merge_candidates(a: Candidate, b: Candidate) -> Candidate:
    """Union two same-instance fragments produced by overlapping tiles."""
    return Candidate(
        label=a.label,
        score=max(a.score, b.score),
        mask=np.logical_or(a.mask, b.mask),
        bbox=(
            min(a.bbox[0], b.bbox[0]),
            min(a.bbox[1], b.bbox[1]),
            max(a.bbox[2], b.bbox[2]),
            max(a.bbox[3], b.bbox[3]),
        ),
        source_tiles=a.source_tiles | b.source_tiles,
        touches_tile_boundary=a.touches_tile_boundary or b.touches_tile_boundary,
    )


def mask_nms(
    candidates: list[Candidate],
    same_class_threshold: float,
    cross_class_threshold: float,
    containment_threshold: float,
    merge_iou_threshold: float,
    merge_containment_threshold: float,
    max_instances: int,
) -> list[Candidate]:
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda row: row.score, reverse=True):
        duplicate = False
        for index, previous in enumerate(kept):
            threshold = (
                same_class_threshold
                if candidate.label == previous.label
                else cross_class_threshold
            )
            if _bbox_iou(candidate.bbox, previous.bbox) < 0.05:
                continue
            mask_iou = _mask_iou(candidate.mask, previous.mask)
            containment = _mask_containment(candidate.mask, previous.mask)
            same_class = candidate.label == previous.label
            cross_window = candidate.source_tiles.isdisjoint(previous.source_tiles)
            boundary_fragment = (
                candidate.touches_tile_boundary or previous.touches_tile_boundary
            )
            if (
                same_class
                and cross_window
                and boundary_fragment
                and (
                    mask_iou >= merge_iou_threshold
                    or containment >= merge_containment_threshold
                )
            ):
                kept[index] = merge_candidates(previous, candidate)
                duplicate = True
                break
            if (
                mask_iou >= threshold or containment >= containment_threshold
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
    box_nms: float,
    pre_mask_nms_topk: int,
    same_class_nms: float,
    cross_class_nms: float,
    containment_nms: float,
    mask_merge_iou: float,
    mask_merge_containment: float,
    tile_border_margin: int,
    max_instances: int,
    amp: bool,
) -> list[Candidate]:
    height, width = image.shape[:2]
    jobs = [
        (x0, y0)
        for y0 in sliding_origins(height, tile_size, stride)
        for x0 in sliding_origins(width, tile_size, stride)
    ]
    raw_candidates: list[RawCandidate] = []
    for start in range(0, len(jobs), tile_batch_size):
        batch_jobs = jobs[start : start + tile_batch_size]
        tiles = [_tile_tensor(image, x, y, tile_size).to(device) for x, y in batch_jobs]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            outputs = model(tiles)
        for job_offset, ((x0, y0), output) in enumerate(
            zip(batch_jobs, outputs, strict=True)
        ):
            tile_id = start + job_offset
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
                bx1 = min(max(float(box[0]) + x0, 0.0), float(width))
                by1 = min(max(float(box[1]) + y0, 0.0), float(height))
                bx2 = min(max(float(box[2]) + x0, 0.0), float(width))
                by2 = min(max(float(box[3]) + y0, 0.0), float(height))
                margin = min(tile_border_margin, valid_h, valid_w)
                touches_boundary = margin > 0 and (
                    (
                        y0 > 0
                        and (local[:margin, :].any() or float(box[1]) <= margin)
                    )
                    or (
                        y0 + valid_h < height
                        and (
                            local[-margin:, :].any()
                            or float(box[3]) >= valid_h - margin
                        )
                    )
                    or (
                        x0 > 0
                        and (local[:, :margin].any() or float(box[0]) <= margin)
                    )
                    or (
                        x0 + valid_w < width
                        and (
                            local[:, -margin:].any()
                            or float(box[2]) >= valid_w - margin
                        )
                    )
                )
                raw_candidates.append(
                    RawCandidate(
                        int(label),
                        float(score),
                        local,
                        (bx1, by1, bx2, by2),
                        x0,
                        y0,
                        valid_h,
                        valid_w,
                        tile_id,
                        bool(touches_boundary),
                    )
                )
    raw_candidates = fast_box_nms(raw_candidates, box_nms, pre_mask_nms_topk)
    candidates = [
        materialize_candidate(candidate, height, width) for candidate in raw_candidates
    ]
    return mask_nms(
        candidates,
        same_class_nms,
        cross_class_nms,
        containment_nms,
        mask_merge_iou,
        mask_merge_containment,
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
    scales = parse_eval_scales(getattr(args, "eval_scales", "1.0"))
    results: list[dict[str, Any]] = []
    image_ids: list[int] = []
    for row in tqdm(rows, desc="sliding-window inference"):
        image_id = int(row["id"])
        image_ids.append(image_id)
        image = index.load_image(row)
        height, width = image.shape[:2]
        merged: list[Candidate] = []
        for scale in scales:
            if abs(scale - 1.0) < 1e-6:
                scaled = image
            else:
                new_h = max(1, int(round(height * scale)))
                new_w = max(1, int(round(width * scale)))
                # Keep channels last for RGB(/SAR).
                scaled = np.stack(
                    [
                        np.array(
                            Image.fromarray(image[..., channel]).resize(
                                (new_w, new_h), Image.BILINEAR
                            )
                        )
                        for channel in range(image.shape[2])
                    ],
                    axis=-1,
                )
            scale_h = scaled.shape[0] / height
            scale_w = scaled.shape[1] / width
            candidates = predict_image(
                model,
                scaled,
                device,
                args.tile_size,
                args.stride,
                args.tile_batch_size,
                args.score_threshold,
                args.mask_threshold,
                args.box_nms,
                args.pre_mask_nms_topk,
                args.same_class_nms,
                args.cross_class_nms,
                args.containment_nms,
                args.mask_merge_iou,
                args.mask_merge_containment,
                args.tile_border_margin,
                args.max_instances,
                args.amp,
            )
            for candidate in candidates:
                if abs(scale - 1.0) < 1e-6:
                    merged.append(candidate)
                    continue
                mask = np.array(
                    Image.fromarray(candidate.mask.astype(np.uint8) * 255).resize(
                        (width, height), Image.NEAREST
                    )
                ) > 127
                if not mask.any():
                    continue
                x1, y1, x2, y2 = candidate.bbox
                merged.append(
                    Candidate(
                        candidate.label,
                        candidate.score,
                        mask,
                        (
                            x1 / scale_w,
                            y1 / scale_h,
                            x2 / scale_w,
                            y2 / scale_h,
                        ),
                        candidate.source_tiles,
                        candidate.touches_tile_boundary,
                    )
                )
        if len(scales) > 1:
            merged = mask_nms(
                merged,
                args.same_class_nms,
                args.cross_class_nms,
                args.containment_nms,
                args.mask_merge_iou,
                args.mask_merge_containment,
                args.max_instances,
            )
        results.extend(candidate_to_coco(candidate, image_id) for candidate in merged)
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
    # Some valid COCO-style datasets (including UBC v2.0) omit the optional
    # top-level ``info`` field. pycocotools.loadRes() accesses it unconditionally.
    coco_gt.dataset.setdefault("info", {})
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
    for name in (
        "task",
        "tile_size",
        "patch_size",
        "s2_rgb_mode",
        "unfreeze_backbone",
        "detail_skip",
    ):
        if not hasattr(args, name):
            continue
        expected = checkpoint.get(name)
        actual = getattr(args, name)
        if expected is not None and expected != actual:
            raise ValueError(
                f"checkpoint {name}={expected!r}, but command uses {actual!r}"
            )


def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    # Pass device_id so NCCL knows the rank→GPU mapping (avoids hang warnings).
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    return True, local_rank, rank, world_size


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def is_main_process(rank: int) -> bool:
    return rank == 0


def parse_eval_scales(raw: str) -> list[float]:
    scales = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("--eval-scales must be a comma-separated list of positive floats")
    return scales


def train(args: argparse.Namespace) -> None:
    distributed, local_rank, rank, world_size = setup_distributed()
    try:
        _train_impl(args, distributed, local_rank, rank, world_size)
    finally:
        cleanup_distributed(distributed)


def _train_impl(
    args: argparse.Namespace,
    distributed: bool,
    local_rank: int,
    rank: int,
    world_size: int,
) -> None:
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if distributed:
        device = torch.device("cuda", local_rank)
    else:
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(args.device or default_device)
    out_dir = Path(args.out_dir)
    if is_main_process(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
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
        args.balanced_extra_samples,
        augment=True,
        max_images=args.max_train_images,
        copy_paste_probability=args.copy_paste_probability,
        require_instances=distributed,
    )
    sampler: DistributedSampler | None = None
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=collate_detection,
    )
    class_counts = collect_class_counts(train_index, NUM_CLASSES)
    args._class_counts = class_counts
    model = build_model(
        Path(args.weights),
        args.task,
        args.tile_size,
        args.patch_size,
        args.s2_rgb_mode,
        device,
        args.unfreeze_backbone,
        use_detail_skip=args.detail_skip,
        use_cascade=args.cascade,
        use_seesaw=args.seesaw,
        class_counts=class_counts,
    )
    if args.init_ckpt:
        checkpoint = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        load_trainable_state(model, checkpoint)
    if distributed:
        # Reduce DDP "grad strides do not match bucket view strides" noise/overhead.
        for parameter in model.parameters():
            parameter.data = parameter.data.contiguous()
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.ddp_find_unused,
            broadcast_buffers=False,
            static_graph=args.ddp_static_graph,
            gradient_as_bucket_view=True,
        )
    raw_model = unwrap_model(model)
    encoder_parameters = [
        parameter
        for parameter in raw_model.backbone.encoder.parameters()
        if parameter.requires_grad
    ]
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [
        parameter
        for parameter in raw_model.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_ids
    ]
    parameters = head_parameters + encoder_parameters
    groups: list[dict[str, Any]] = [{"params": head_parameters, "lr": args.lr}]
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start_epoch = 1
    best_ap50 = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        _checkpoint_args(checkpoint, args)
        load_trainable_state(raw_model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if checkpoint.get("best_ap50") is not None:
            best_ap50 = float(checkpoint["best_ap50"])
        elif checkpoint.get("metrics"):
            best_ap50 = float(checkpoint["metrics"].get("AP50", -1.0))
    trainable = sum(parameter.numel() for parameter in parameters)
    frozen = sum(
        parameter.numel()
        for parameter in raw_model.parameters()
        if not parameter.requires_grad
    )
    if is_main_process(rank):
        print(
            f"task={args.task} device={device} world_size={world_size} "
            f"train_images={len(dataset.images)} val_images={len(val_index.images)} "
            f"tiles/epoch={len(dataset)}"
        )
        print(
            f"crop sampling: per-image slots={args.samples_per_image}, "
            f"additional global class-balanced slots={dataset.balanced_extra_samples}, "
            f"copy_paste_p={args.copy_paste_probability}, "
            f"cascade={args.cascade}, seesaw={args.seesaw}"
        )
        print(f"parameters: trainable={trainable:,}, frozen={frozen:,}")
    history_path = out_dir / "history.json"
    history: list[dict[str, Any]] = []
    if args.resume and history_path.exists() and is_main_process(rank):
        loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
        if isinstance(loaded_history, list):
            history = [
                row
                for row in loaded_history
                if isinstance(row, dict) and int(row.get("epoch", 0)) < start_epoch
            ]

    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals: Counter[str] = Counter()
        batches = 0
        started = time.time()
        progress = tqdm(
            loader,
            desc=f"epoch {epoch}/{args.epochs}",
            disable=not is_main_process(rank),
        )
        for step, (images, targets) in enumerate(progress, start=1):
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
            if is_main_process(rank) and (
                step % args.log_every == 0 or step == len(loader)
            ):
                postfix = {
                    "loss": f"{totals['loss'] / batches:.4f}",
                    "cls": f"{totals['loss_classifier'] / batches:.4f}",
                    "box": f"{totals['loss_box_reg'] / batches:.4f}",
                    "mask": f"{totals['loss_mask'] / batches:.4f}",
                    "obj": f"{totals['loss_objectness'] / batches:.4f}",
                    "rpn_box": f"{totals['loss_rpn_box_reg'] / batches:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
                progress.set_postfix(postfix, refresh=True)
                # Newline so ``tail -f`` on a redirected log shows progress (tqdm
                # normally rewrites the same line with ``\\r`` and looks frozen).
                print(
                    f"epoch={epoch} step={step}/{len(loader)} "
                    f"loss={postfix['loss']} cls={postfix['cls']} "
                    f"box={postfix['box']} mask={postfix['mask']} "
                    f"obj={postfix['obj']} rpn_box={postfix['rpn_box']} "
                    f"lr={postfix['lr']}",
                    flush=True,
                )
        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            "seconds": time.time() - started,
            "lr": scheduler.get_last_lr()[0],
            **{name: value / max(batches, 1) for name, value in totals.items()},
        }
        metrics: dict[str, float] | None = None
        should_eval = epoch % args.eval_every == 0 or epoch == args.epochs
        if should_eval and is_main_process(rank):
            save_checkpoint(
                out_dir / "last.pt",
                raw_model,
                optimizer,
                scheduler,
                epoch,
                args,
                metrics=None,
                best_ap50=best_ap50,
            )
            predictions, image_ids = predict_split(
                raw_model, val_index, args, device, max_images=args.val_max_images
            )
            prediction_path = out_dir / f"val_epoch_{epoch:03d}.json"
            prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
            metrics = evaluate_coco(val_paths.json_path, predictions, image_ids)
            row.update({f"val_{key}": value for key, value in metrics.items()})
            print(
                f"epoch={epoch} loss={row['loss']:.4f} "
                f"cls={row['loss_classifier']:.4f} "
                f"box={row['loss_box_reg']:.4f} "
                f"mask={row['loss_mask']:.4f} "
                f"obj={row['loss_objectness']:.4f} "
                f"rpn_box={row['loss_rpn_box_reg']:.4f} "
                f"val_AP={metrics['AP']:.4f} val_AP50={metrics['AP50']:.4f}"
            )
        elif is_main_process(rank):
            print(
                f"epoch={epoch} loss={row['loss']:.4f} "
                f"cls={row['loss_classifier']:.4f} "
                f"box={row['loss_box_reg']:.4f} "
                f"mask={row['loss_mask']:.4f} "
                f"obj={row['loss_objectness']:.4f} "
                f"rpn_box={row['loss_rpn_box_reg']:.4f}"
            )
        if is_main_process(rank):
            history.append(row)
            if metrics is not None and metrics["AP50"] > best_ap50:
                best_ap50 = metrics["AP50"]
                save_checkpoint(
                    out_dir / "best.pt",
                    raw_model,
                    optimizer,
                    scheduler,
                    epoch,
                    args,
                    metrics,
                    best_ap50,
                )
                print(f"saved best.pt (AP50={best_ap50:.4f})")
            save_checkpoint(
                out_dir / "last.pt",
                raw_model,
                optimizer,
                scheduler,
                epoch,
                args,
                metrics,
                best_ap50,
            )
            history_path.write_text(
                json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if distributed:
            dist.barrier()
    if is_main_process(rank):
        print(f"training complete: {out_dir}")


def load_for_inference(
    args: argparse.Namespace, device: torch.device
) -> tuple[MaskRCNN, dict[str, Any]]:
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    _checkpoint_args(checkpoint, args)
    model = build_model(
        Path(args.weights),
        args.task,
        args.tile_size,
        args.patch_size,
        args.s2_rgb_mode,
        device,
        unfreeze_backbone=bool(checkpoint.get("unfreeze_backbone", False)),
        use_detail_skip=bool(checkpoint.get("detail_skip", args.detail_skip)),
        use_cascade=bool(checkpoint.get("cascade", getattr(args, "cascade", True))),
        use_seesaw=bool(checkpoint.get("seesaw", getattr(args, "seesaw", True))),
        class_counts=checkpoint.get("class_counts"),
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


def evaluate_json_command(args: argparse.Namespace) -> None:
    paths = resolve_split_paths(Path(args.data_root), args.task, args.split)
    index = CocoIndex(paths)
    rows = select_eval_images(index.images, args.max_images, args.seed)
    image_ids = [int(row["id"]) for row in rows]
    prediction_path = Path(args.predictions_json)
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    metrics = evaluate_coco(paths.json_path, predictions, image_ids)
    metrics_path = prediction_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
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
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=4, choices=range(1, 9))
    parser.add_argument(
        "--s2-rgb-mode",
        choices=("rgb-only", "repeat"),
        default="rgb-only",
        help=(
            "rgb-only maps B,G,R to S2 B02,B03,B04 and zeros other bands; "
            "repeat fills all 12 S2 channels with four B,G,R repetitions"
        ),
    )
    parser.add_argument("--device", default=None, help="Default: cuda when available")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True, help="CUDA bfloat16 autocast"
    )
    parser.add_argument(
        "--detail-skip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fuse RGB(/SAR) high-res skip into the finest FPN level",
    )
    parser.add_argument(
        "--eval-scales",
        default="1.0",
        help="comma-separated image scales for sliding-window inference, e.g. 0.75,1.0,1.25",
    )
    parser.add_argument(
        "--cascade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use Cascade Mask R-CNN ROI heads (IoU 0.5/0.6/0.7)",
    )
    parser.add_argument(
        "--seesaw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use Seesaw loss for long-tailed box classification",
    )


def add_inference(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--box-nms",
        type=float,
        default=0.8,
        help="class-aware box NMS before expensive full-image mask processing",
    )
    parser.add_argument(
        "--pre-mask-nms-topk",
        type=int,
        default=1000,
        help="maximum candidates entering full-image mask NMS and fusion",
    )
    parser.add_argument("--same-class-nms", type=float, default=0.5)
    parser.add_argument("--cross-class-nms", type=float, default=0.7)
    parser.add_argument(
        "--containment-nms",
        type=float,
        default=0.85,
        help="suppress a clipped mask when it is mostly contained in another",
    )
    parser.add_argument(
        "--mask-merge-iou",
        type=float,
        default=0.2,
        help="union same-class cross-tile boundary masks above this IoU",
    )
    parser.add_argument(
        "--mask-merge-containment",
        type=float,
        default=0.5,
        help="union boundary masks when intersection/smaller-mask exceeds this value",
    )
    parser.add_argument(
        "--tile-border-margin",
        type=int,
        default=3,
        help="pixels used to identify masks clipped by an internal tile boundary",
    )
    parser.add_argument("--max-instances", type=int, default=800)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser(
        "train", help="train v3 Cascade+Seesaw Mask R-CNN (DDP)"
    )
    add_common(train_parser)
    add_inference(train_parser)
    train_parser.add_argument("--out-dir", required=True)
    train_parser.add_argument("--epochs", type=int, default=24)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--accum-steps", type=int, default=2)
    train_parser.add_argument("--workers", type=int, default=4)
    train_parser.add_argument("--samples-per-image", type=int, default=4)
    train_parser.add_argument(
        "--max-train-images",
        type=int,
        default=0,
        help="0 uses all images; set a small number only for smoke tests",
    )
    train_parser.add_argument("--object-crop-probability", type=float, default=0.85)
    train_parser.add_argument(
        "--balanced-extra-samples",
        type=int,
        default=2,
        help=(
            "additional sqrt-balanced global crops per training-image equivalent; "
            "v3 default 2 to stress rare roof types"
        ),
    )
    train_parser.add_argument(
        "--copy-paste-probability",
        type=float,
        default=0.5,
        help="probability of pasting a rare-class instance onto the crop",
    )
    train_parser.add_argument(
        "--ddp-find-unused",
        action="store_true",
        help="enable DDP unused-parameter search (slow, deadlocks if ranks disagree)",
    )
    train_parser.add_argument(
        "--ddp-static-graph",
        action="store_true",
        help="declare the DDP graph static (faster, requires identical graph every step)",
    )
    train_parser.add_argument("--min-visible-fraction", type=float, default=0.5)
    train_parser.add_argument("--min-mask-area", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="fine-tune the OlmoEarth encoder instead of keeping it frozen",
    )
    train_parser.add_argument(
        "--backbone-lr",
        type=float,
        default=2e-5,
        help="encoder learning rate; only used with --unfreeze-backbone",
    )
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    train_parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="update training loss display every N batches (default: 500)",
    )
    train_parser.add_argument("--eval-every", type=int, default=4)
    train_parser.add_argument(
        "--val-max-images",
        type=int,
        default=0,
        help=(
            "0 evaluates the full val split; a positive value evaluates a "
            "fixed seed-random subset"
        ),
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

    eval_json_parser = commands.add_parser(
        "eval-json", help="evaluate an existing COCO prediction JSON without inference"
    )
    eval_json_parser.add_argument("--data-root", required=True, help="UBC_v2.0 directory")
    eval_json_parser.add_argument("--task", choices=("single", "multimodal"), required=True)
    eval_json_parser.add_argument(
        "--split", choices=("train", "val", "test"), default="val"
    )
    eval_json_parser.add_argument("--predictions-json", required=True)
    eval_json_parser.add_argument("--max-images", type=int, default=0)
    eval_json_parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if hasattr(args, "tile_size") and (
        args.tile_size <= 0 or args.tile_size % args.patch_size
    ):
        raise ValueError("--tile-size must be positive and divisible by --patch-size")
    if hasattr(args, "stride") and not 0 < args.stride <= args.tile_size:
        raise ValueError("expected 0 < --stride <= --tile-size")
    probability_names = (
        "score_threshold",
        "mask_threshold",
        "box_nms",
        "same_class_nms",
        "cross_class_nms",
        "containment_nms",
        "mask_merge_iou",
        "mask_merge_containment",
        "object_crop_probability",
    )
    for name in probability_names:
        if hasattr(args, name) and not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.command == "train":
        if args.accum_steps < 1 or args.batch_size < 1 or args.samples_per_image < 1:
            raise ValueError("batch, accumulation, and sampling counts must be positive")
        if args.max_train_images < 0 or args.val_max_images < 0:
            raise ValueError("image limits must be non-negative")
        if args.balanced_extra_samples < 0:
            raise ValueError("--balanced-extra-samples must be non-negative")
        if args.eval_every < 1:
            raise ValueError("--eval-every must be positive")
        if args.backbone_lr <= 0:
            raise ValueError("--backbone-lr must be positive")
        if not 0 <= args.copy_paste_probability <= 1:
            raise ValueError("--copy-paste-probability must be in [0, 1]")
        parse_eval_scales(args.eval_scales)
        if args.log_every < 1:
            raise ValueError("--log-every must be positive")
    if hasattr(args, "pre_mask_nms_topk") and args.pre_mask_nms_topk < 1:
        raise ValueError("--pre-mask-nms-topk must be positive")
    if hasattr(args, "tile_border_margin") and args.tile_border_margin < 1:
        raise ValueError("--tile-border-margin must be positive")
    if hasattr(args, "eval_scales"):
        parse_eval_scales(args.eval_scales)
    if args.command == "eval-json" and args.max_images < 0:
        raise ValueError("--max-images must be non-negative")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate_command(args)
    elif args.command == "eval-json":
        evaluate_json_command(args)
    else:
        predict_command(args)


if __name__ == "__main__":
    main()
