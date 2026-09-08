"""Optical floating-roof tank detection on frozen OlmoEarth.

For storage calculation on ADD_with_metadata, use optical_tank_volume.py.

Two stages (OilWatch-style, but backbone = your OlmoEarth weights)::

  1) Detection on uncropped scenes (tile / sliding window)
       RGB -> pad to sentinel2_l2a (12ch) -> frozen OlmoEarth
       -> DetectHead (center heatmap + box size)
  2) Optical storage volume on each crop
       circle / Hough -> R
       external / internal shadow lengths Lex, Lin
       -> H, h via solar+satellite geometry (paper Eq.1-2)
       -> V ≈ pi * R^2 * oil_depth

This does NOT replace SAR ``train_hh_tank_geom.py``. Optical volume uses
shadows; SAR volume uses O1/O2/O3. Same frozen weights, different heads.

Data (detection, COCO — matches Object_Detection_train_split)::

    data_root/
      train|test/
        image/*.{tif,tiff,jpg,png}
        coco_label.json          # COCO: images + annotations bbox=[x,y,w,h]

Also accepts YOLO layout (images/ + labels/*.txt) if no coco_label.json.

Examples::

    python scripts/tools/optical_tank_pipeline.py train-det \\
      --data-root G:/SAR_Oil_DataSet/Object_Detection_train_split \\
      --weights .../OlmoEarth-v1_2-Base \\
      --out-dir runs/tank_det --size 1024
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import load_model_from_path

try:
    from scripts.tools.optical_tank_geometry import (
        detect_tank_circle,
        index_tank_pairs,
        measure_shadow,
        parse_optical_metadata,
        read_extend_px,
        read_rgb_u8,
        to_dict,
    )
except ModuleNotFoundError:  # direct execution from scripts/tools
    from optical_tank_geometry import (
        detect_tank_circle,
        index_tank_pairs,
        measure_shadow,
        parse_optical_metadata,
        read_extend_px,
        read_rgb_u8,
        to_dict,
    )


# ---------------------------------------------------------------------------
# RGB -> OlmoEarth sentinel2_l2a (12 bands)
# ---------------------------------------------------------------------------

# S2 L2A order in OlmoEarth: B02,B03,B04,B08, B05,B06,B07,B8A,B11,B12, B01,B09
_S2_B02, _S2_B03, _S2_B04, _S2_B08 = 0, 1, 2, 3


def rgb_to_s2_l2a(rgb: np.ndarray, scale: float = 10000.0) -> np.ndarray:
    """rgb HxWx3 in [0,1] or [0,255] -> HxWx12 float (DN-like)."""
    x = rgb.astype(np.float32)
    if x.max() <= 1.5:
        x = x * scale
    h, w, _ = x.shape
    out = np.zeros((h, w, 12), dtype=np.float32)
    # map R,G,B -> B04,B03,B02 (approx optical)
    out[:, :, _S2_B04] = x[:, :, 0]
    out[:, :, _S2_B03] = x[:, :, 1]
    out[:, :, _S2_B02] = x[:, :, 2]
    # weak NIR proxy from red (better than zero for B08 slot)
    out[:, :, _S2_B08] = x[:, :, 0]
    return out


def load_rgb(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGB"))
    return arr.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Detect head on frozen OlmoEarth tokens
# ---------------------------------------------------------------------------


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DetectHead(nn.Module):
    """CenterNet-style: center heatmap + (log w, log h) at centers.

    Lighter than Cascade-RCNN but fits tiled OlmoEarth tokens. OilWatch uses
    Cascade+Swin for detection; here Swin is replaced by your OlmoEarth encoder.
    """

    def __init__(self, in_dim: int, patch_size: int, mid_dim: int = 256) -> None:
        if patch_size < 1 or (patch_size & (patch_size - 1)) != 0:
            raise ValueError("patch_size must be power of two")
        super().__init__()
        n = int(math.log2(patch_size))
        layers: list[nn.Module] = [ConvBNReLU(in_dim, mid_dim)]
        ch = mid_dim
        for _ in range(n):
            nxt = max(ch // 2, 64)
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvBNReLU(ch, nxt),
            ]
            ch = nxt
        self.neck = nn.Sequential(*layers)
        self.center = nn.Conv2d(ch, 1, 1)
        self.size = nn.Conv2d(ch, 2, 1)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        # tokens B,H',W',D
        feat = self.neck(tokens.permute(0, 3, 1, 2).contiguous())
        return {"center": self.center(feat), "size": self.size(feat)}


class OlmoOpticalDetector(nn.Module):
    """Frozen OlmoEarth + DetectHead. RGB padded into sentinel2_l2a."""

    def __init__(
        self, backbone: nn.Module, emb_dim: int, patch_size: int = 4, mid_dim: int = 256
    ) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.patch_size = patch_size
        # learnable mix of RGB-mapped S2 slots (keeps C=12 interface)
        self.band_gain = nn.Parameter(torch.ones(12))
        self.head = DetectHead(emb_dim, patch_size, mid_dim=mid_dim)

    def train(self, mode: bool = True):  # noqa: A003
        super().train(mode)
        self.backbone.eval()
        return self

    def encode_s2(self, s2_bhwc: torch.Tensor) -> torch.Tensor:
        """s2: B,H,W,12 in DN-ish scale -> tokens B,H',W',D."""
        b, h, w, c = s2_bhwc.shape
        assert c == 12
        x = s2_bhwc * self.band_gain.view(1, 1, 1, 12)
        x = x.unsqueeze(3)  # B,H,W,T=1,C
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=x,
            sentinel2_l2a_mask=torch.full(
                (b, h, w, 1, 3),
                MaskValue.ONLINE_ENCODER.value,
                device=x.device,
                dtype=x.dtype,
            ),
            timestamps=torch.tensor(
                [[[15, 5, 2020]]] * b, device=x.device, dtype=torch.long
            ),
        )
        with torch.no_grad():
            out = self.backbone.encoder(
                sample, fast_pass=True, patch_size=self.patch_size
            )
        return out["tokens_and_masks"].sentinel2_l2a.mean(dim=(3, 4))

    def forward(self, s2_bhwc: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.encode_s2(s2_bhwc))


# ---------------------------------------------------------------------------
# Detection dataset: COCO (preferred) or YOLO
# ---------------------------------------------------------------------------


def _fit_square_rgb(
    rgb: np.ndarray, size: int
) -> tuple[np.ndarray, tuple[float, float]]:
    """Pad/crop to size; return image and (ox, oy) so x' = x + ox, y' = y + oy."""
    h, w = rgb.shape[:2]
    pad_h, pad_w = max(0, size - h), max(0, size - w)
    pad_top, pad_left = pad_h // 2, pad_w // 2
    if pad_h or pad_w:
        rgb = np.pad(
            rgb,
            ((pad_top, pad_h - pad_top), (pad_left, pad_w - pad_left), (0, 0)),
            mode="reflect",
        )
        h, w = rgb.shape[:2]
    y0, x0 = max(0, (h - size) // 2), max(0, (w - size) // 2)
    ox, oy = float(pad_left - x0), float(pad_top - y0)
    return rgb[y0 : y0 + size, x0 : x0 + size], (ox, oy)


def _load_coco_index(coco_path: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    """file_name -> list of absolute (x, y, w, h) COCO boxes."""
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    id_to_name = {im["id"]: im["file_name"] for im in data["images"]}
    out: dict[str, list[tuple[float, float, float, float]]] = {
        name: [] for name in id_to_name.values()
    }
    for ann in data["annotations"]:
        name = id_to_name.get(ann["image_id"])
        if name is None:
            continue
        x, y, bw, bh = ann["bbox"]
        out[name].append((float(x), float(y), float(bw), float(bh)))
        # also key by stem for flexible matching
        out.setdefault(Path(name).name, out[name])
    return out


def _read_yolo_boxes(path: Path) -> list[tuple[float, float, float, float]]:
    """Return list of (xc, yc, w, h) normalized."""
    if not path.is_file():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.strip().split()
        if len(p) < 5:
            continue
        boxes.append((float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return boxes


def _gaussian2d(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)).astype(np.float32)


def _boxes_to_targets(
    boxes_xywh_px: list[tuple[float, float, float, float]],
    size: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Absolute pixel xywh (already in fitted canvas) -> heat, size_map, mask."""
    heat = np.zeros((size, size), dtype=np.float32)
    size_map = np.zeros((2, size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.float32)
    for x, y, bw, bh in boxes_xywh_px:
        cx, cy = x + bw / 2.0, y + bh / 2.0
        if not (0 <= cx < size and 0 <= cy < size):
            continue
        heat = np.maximum(heat, _gaussian2d(size, size, cx, cy, sigma))
        ix, iy = int(round(cx)), int(round(cy))
        size_map[0, iy, ix] = math.log(max(bw, 1.0))
        size_map[1, iy, ix] = math.log(max(bh, 1.0))
        mask[iy, ix] = 1.0
    return heat, size_map, mask


class OpticalDetDataset(Dataset):
    """Loads ``split/image`` + ``split/coco_label.json`` (or YOLO images/labels)."""

    def __init__(self, root: Path, split: str, size: int = 512, sigma: float = 3.0) -> None:
        self.root = root
        self.split = split
        self.size = size
        self.sigma = sigma
        coco_path = root / split / "coco_label.json"
        img_coco = root / split / "image"
        img_yolo = root / split / "images"
        self.coco_boxes: dict[str, list[tuple[float, float, float, float]]] | None = None
        self.yolo_dir: Path | None = None

        if coco_path.is_file() and img_coco.is_dir():
            self.img_dir = img_coco
            self.coco_boxes = _load_coco_index(coco_path)
            self.format = "coco"
        elif img_yolo.is_dir():
            self.img_dir = img_yolo
            self.yolo_dir = root / split / "labels"
            self.format = "yolo"
        else:
            raise FileNotFoundError(
                f"Need {split}/coco_label.json+image/ or {split}/images/+labels/"
            )

        self.paths = sorted(
            [
                *self.img_dir.glob("*.tif"),
                *self.img_dir.glob("*.tiff"),
                *self.img_dir.glob("*.jpg"),
                *self.img_dir.glob("*.png"),
            ]
        )
        if not self.paths:
            raise FileNotFoundError(self.img_dir)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[idx]
        rgb0 = load_rgb(path)
        h0, w0 = rgb0.shape[:2]
        rgb, (ox, oy) = _fit_square_rgb(rgb0, self.size)
        s2 = rgb_to_s2_l2a(rgb)

        boxes_fit: list[tuple[float, float, float, float]] = []
        if self.format == "coco" and self.coco_boxes is not None:
            raw = self.coco_boxes.get(path.name) or self.coco_boxes.get(path.stem, [])
            for x, y, bw, bh in raw:
                boxes_fit.append((x + ox, y + oy, bw, bh))
        else:
            assert self.yolo_dir is not None
            for xc, yc, bw, bh in _read_yolo_boxes(self.yolo_dir / f"{path.stem}.txt"):
                # YOLO normalized on original; approximate on fitted canvas
                boxes_fit.append(
                    (
                        xc * w0 + ox - bw * w0 / 2,
                        yc * h0 + oy - bh * h0 / 2,
                        bw * w0,
                        bh * h0,
                    )
                )

        heat, size_map, mask = _boxes_to_targets(boxes_fit, self.size, self.sigma)
        return {
            "s2": torch.from_numpy(s2),
            "center": torch.from_numpy(heat),
            "size": torch.from_numpy(size_map),
            "size_mask": torch.from_numpy(mask),
            "name": path.name,
        }


def det_loss(
    out: dict[str, torch.Tensor],
    center: torch.Tensor,
    size: torch.Tensor,
    size_mask: torch.Tensor,
) -> torch.Tensor:
    # focal-ish MSE on heatmap
    pred_c = out["center"][:, 0]
    pos = center > 0.5
    neg = ~pos
    loss_c = ((pred_c.sigmoid() - center).pow(2) * (pos.float() * 4 + neg.float())).mean()
    # size only at centers
    pred_s = out["size"]
    m = size_mask.unsqueeze(1)
    if m.sum() > 0:
        loss_s = ((pred_s - size).pow(2) * m).sum() / m.sum()
    else:
        loss_s = pred_s.sum() * 0.0
    return loss_c + 0.1 * loss_s


# ---------------------------------------------------------------------------
# Decode boxes from heatmaps
# ---------------------------------------------------------------------------


def decode_boxes(
    center_logits: torch.Tensor,
    size_logits: torch.Tensor,
    conf_thresh: float = 0.3,
    topk: int = 100,
    nms_dist: float = 16.0,
) -> list[list[tuple[float, float, float, float, float]]]:
    """Per-image list of (x1,y1,x2,y2,score) in pixel coords of the feature map."""
    probs = center_logits[:, 0].sigmoid()
    b, h, w = probs.shape
    results: list[list[tuple[float, float, float, float, float]]] = []
    for i in range(b):
        flat = probs[i].reshape(-1)
        k = min(topk, flat.numel())
        scores, idxs = torch.topk(flat, k)
        boxes: list[tuple[float, float, float, float, float]] = []
        for sc, idx in zip(scores.tolist(), idxs.tolist(), strict=True):
            if sc < conf_thresh:
                continue
            y, x = divmod(idx, w)
            lw = float(size_logits[i, 0, y, x])
            lh = float(size_logits[i, 1, y, x])
            bw, bh = math.exp(lw), math.exp(lh)
            boxes.append((x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2, sc))
        # simple distance NMS on centers
        kept: list[tuple[float, float, float, float, float]] = []
        for box in sorted(boxes, key=lambda t: -t[4]):
            cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
            if any(
                math.hypot(cx - 0.5 * (k[0] + k[2]), cy - 0.5 * (k[1] + k[3])) < nms_dist
                for k in kept
            ):
                continue
            kept.append(box)
        results.append(kept)
    return results


# ---------------------------------------------------------------------------
# Optical volume (paper Eq.1-2)
# ---------------------------------------------------------------------------


def optical_height_factor(
    solar_elev_deg: float,
    sat_elev_deg: float,
    solar_az_deg: float,
    sat_az_deg: float,
) -> float:
    """Denominator of paper Eq.1-2."""
    a = math.radians(solar_elev_deg)
    b = math.radians(sat_elev_deg)
    g = math.radians(solar_az_deg)
    t = math.radians(sat_az_deg)
    ca, cb = 1.0 / max(math.tan(a), 1e-6), 1.0 / max(math.tan(b), 1e-6)
    inside = ca * ca + cb * cb - 2 * ca * cb * math.cos(g - t)
    return math.sqrt(max(inside, 1e-8))


def volume_from_optical(
    r_m: float,
    lex_m: float,
    lin_m: float,
    solar_elev_deg: float,
    sat_elev_deg: float,
    solar_az_deg: float,
    sat_az_deg: float,
) -> dict[str, float | bool | str]:
    denom = optical_height_factor(
        solar_elev_deg, sat_elev_deg, solar_az_deg, sat_az_deg
    )
    H = lex_m / denom
    h = lin_m / denom
    oil = H - h
    valid = r_m > 0 and H > 0 and 0 <= h <= H and oil >= 0
    v_m3 = math.pi * r_m * r_m * oil if valid else float("nan")
    return {
        "R_m": r_m,
        "H_m": H,
        "h_m": h,
        "oil_height_m": oil,
        "V_m3": v_m3,
        "V_bbl": v_m3 / 0.158987 if valid else float("nan"),
        "geometry_valid": valid,
        "geometry_status": "ok" if valid else "invalid_geometry",
        "formula": "optical_shadow_eq1_2",
    }


def fit_circle_geometry_px(
    gray: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float | str]:
    """Estimate tank centre/radius, preferring a central well-supported Hough circle.

    ``E:/code`` measures the circle on a paired no-shadow image.  Callers may pass
    such an image here; a central bright component is used when Hough finds no circle.
    """
    g = gray
    if g.ndim == 3:
        g = (
            0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]
        ).astype(np.float32)
    if float(g.max()) <= 1.5:
        g8 = (g * 255).clip(0, 255).astype(np.uint8)
    else:
        g8 = g.clip(0, 255).astype(np.uint8)
    if mask is None:
        # Bright-region fallback, restricted to the component nearest image centre.
        thr = float(np.percentile(g, 60))
        mask = (g >= thr).astype(np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8)
    enhanced = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(g8)
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
    edges = cv2.Canny(blurred, 80, 180)
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(8, min(g.shape) // 4),
        param1=100,
        param2=22,
        minRadius=max(4, min(g.shape) // 20),
        maxRadius=min(g.shape) // 2,
    )
    if circles is not None:
        h, w = g.shape
        centre = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        def circle_score(circle: np.ndarray) -> float:
            x, y, r = (float(v) for v in circle)
            dist = float(np.linalg.norm(np.array([x, y]) - centre))
            yy, xx = np.ogrid[:h, :w]
            ring = np.abs(np.hypot(xx - x, yy - y) - r) <= 1.5
            support = float((edges[ring] > 0).mean()) if ring.any() else 0.0
            return support - 0.35 * dist / max(min(h, w), 1)

        x, y, radius = max(circles[0], key=circle_score)
        return {
            "cx_px": float(x),
            "cy_px": float(y),
            "radius_px": float(radius),
            "radius_method": "hough",
        }

    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n_labels == 1:
        return {
            "cx_px": g.shape[1] / 2.0,
            "cy_px": g.shape[0] / 2.0,
            "radius_px": 0.0,
            "radius_method": "failed",
        }
    h, w = g.shape
    label = min(
        range(1, n_labels),
        key=lambda i: math.hypot(
            float(centroids[i, 0]) - w / 2.0,
            float(centroids[i, 1]) - h / 2.0,
        )
        / max(math.sqrt(float(stats[i, cv2.CC_STAT_AREA])), 1.0),
    )
    area = float(stats[label, cv2.CC_STAT_AREA])
    return {
        "cx_px": float(centroids[label, 0]),
        "cy_px": float(centroids[label, 1]),
        "radius_px": math.sqrt(area / math.pi),
        "radius_method": "bright_component",
    }


def _dark_run_from_boundary(
    dark: np.ndarray,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
    start_u: float,
    max_length: int,
    offset_v: float,
    gap_tolerance: int = 2,
) -> float:
    """Measure a dark run along ``u`` while tolerating tiny edge/noise gaps."""
    h, w = dark.shape
    px, py = -dy, dx
    run = 0
    gap = 0
    seen_dark = False
    for step in range(max_length + 1):
        u = start_u + step
        x = int(round(cx + dx * u + px * offset_v))
        y = int(round(cy + dy * u + py * offset_v))
        if not (0 <= x < w and 0 <= y < h):
            break
        if dark[y, x]:
            seen_dark = True
            run = step + 1
            gap = 0
        elif seen_dark:
            gap += 1
            if gap > gap_tolerance:
                break
        elif step > gap_tolerance:
            break
    # ``run`` is updated only on dark pixels, so the trailing gap is not part of
    # the measured length and must not be subtracted again.
    return float(run)


def measure_shadow_geometry(
    rgb: np.ndarray,
    cx: float,
    cy: float,
    shadow_direction_deg: float,
    r_px: float,
    scanlines: int = 9,
) -> dict[str, float | int | str]:
    """Estimate external/internal shadows from multiple parallel boundary scans.

    This combines the existing pipeline with the useful parts of ``E:/code``:
    separate circle geometry, inner/outer regions, adaptive dark masks, boundary
    scans, noise-gap tolerance and explicit quality reporting.
    """
    if rgb.ndim == 3:
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        gray = rgb.astype(np.float32)
    h, w = gray.shape
    if r_px <= 0:
        return {
            "Lex_px": 0.0,
            "Lin_px": 0.0,
            "shadow_threshold": float("nan"),
            "external_scanlines": 0,
            "internal_scanlines": 0,
            "shadow_confidence": 0.0,
            "shadow_method": "invalid_circle",
        }

    gray8 = (
        (gray * 255).clip(0, 255).astype(np.uint8)
        if float(gray.max()) <= 1.5
        else gray.clip(0, 255).astype(np.uint8)
    )
    smooth = cv2.bilateralFilter(gray8, 5, 25, 25)
    threshold, dark_u8 = cv2.threshold(
        smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # Closing bridges tiny compression gaps without erasing thin inner shadows.
    dark_u8 = cv2.morphologyEx(
        dark_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    dark = dark_u8 > 0

    # North-up convention: azimuth is clockwise from image north.
    rad = math.radians(shadow_direction_deg)
    dx, dy = math.sin(rad), -math.cos(rad)
    scanlines = max(int(scanlines), 3)
    if scanlines % 2 == 0:
        scanlines += 1
    offsets = np.linspace(-0.45 * r_px, 0.45 * r_px, scanlines)
    external: list[float] = []
    internal: list[float] = []
    for offset in offsets:
        half_chord = math.sqrt(max(r_px * r_px - float(offset) ** 2, 0.0))
        lex = _dark_run_from_boundary(
            dark,
            cx,
            cy,
            dx,
            dy,
            half_chord + 1.0,
            max_length=int(max(h, w) * 0.6),
            offset_v=float(offset),
        )
        lin = _dark_run_from_boundary(
            dark,
            cx,
            cy,
            dx,
            dy,
            -half_chord + 1.0,
            max_length=max(int(2 * half_chord), 1),
            offset_v=float(offset),
        )
        if lex > 0:
            external.append(lex)
        if lin > 0:
            internal.append(lin)

    min_good = max(2, math.ceil(scanlines * 0.3))
    lex_px = float(np.median(external)) if len(external) >= min_good else 0.0
    lin_px = float(np.median(internal)) if len(internal) >= min_good else 0.0
    confidence = 0.7 * len(external) / scanlines + 0.3 * len(internal) / scanlines
    return {
        "Lex_px": lex_px,
        "Lin_px": lin_px,
        "shadow_threshold": float(threshold),
        "external_scanlines": len(external),
        "internal_scanlines": len(internal),
        "shadow_confidence": float(confidence),
        "shadow_method": "multi_scan" if lex_px > 0 else "insufficient_shadow",
    }


def load_optical_meta(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    need = {
        "filename",
        "solar_elev_deg",
        "sat_elev_deg",
        "solar_az_deg",
        "sat_az_deg",
    }
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"optical meta missing {sorted(missing)}")
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        def optional_float(name: str, default: float) -> float:
            value = row.get(name, default)
            return default if pd.isna(value) else float(value)

        fn = str(row["filename"])
        solar_az = float(row["solar_az_deg"])
        sat_az = float(row["sat_az_deg"])
        if "shadow_direction_deg" in df.columns and not pd.isna(
            row["shadow_direction_deg"]
        ):
            shadow_direction = float(row["shadow_direction_deg"])
        else:
            shadow_direction = (solar_az + 180.0) % 360.0
        entry = {
            "solar_elev_deg": float(row["solar_elev_deg"]),
            "sat_elev_deg": float(row["sat_elev_deg"]),
            "solar_az_deg": solar_az,
            "sat_az_deg": sat_az,
            "shadow_direction_deg": shadow_direction,
            "pixel_resolution": optional_float("pixel_resolution", 0.75),
            "circle_offset_x_px": optional_float("circle_offset_x_px", 0.0),
            "circle_offset_y_px": optional_float("circle_offset_y_px", 0.0),
        }
        out[fn] = entry
        out[Path(fn).stem] = entry
    return out


# ---------------------------------------------------------------------------
# Train / detect / volume
# ---------------------------------------------------------------------------


@torch.no_grad()
def _discover_emb_dim(backbone: nn.Module, patch_size: int, device: torch.device) -> int:
    m = OlmoOpticalDetector(backbone, 768, patch_size).to(device)
    dummy = torch.zeros(1, 64, 64, 12, device=device)
    return int(m.encode_s2(dummy).shape[-1])


def train_det(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone = load_model_from_path(args.weights).to(device)
    emb = _discover_emb_dim(backbone, args.patch_size, device)
    model = OlmoOpticalDetector(backbone, emb, args.patch_size, args.mid_dim).to(device)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    train_ds = OpticalDetDataset(Path(args.data_root), "train", args.size)
    root = Path(args.data_root)
    if (root / "val" / "coco_label.json").is_file() or (root / "val" / "images").is_dir():
        val_name = "val"
    elif (root / "test" / "coco_label.json").is_file() or (root / "test" / "image").is_dir():
        val_name = "test"
    else:
        val_name = "train"
    val_ds = OpticalDetDataset(root, val_name, args.size)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    print(f"optical-det emb={emb} train={len(train_ds)} val={len(val_ds)} (backbone frozen)")

    best = 1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            s2 = batch["s2"].to(device)
            out = model(s2)
            loss = det_loss(
                out,
                batch["center"].to(device),
                batch["size"].to(device),
                batch["size_mask"].to(device),
            )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += float(loss.item()) * s2.size(0)
            n += s2.size(0)
        # val loss
        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                s2 = batch["s2"].to(device)
                out = model(s2)
                loss = det_loss(
                    out,
                    batch["center"].to(device),
                    batch["size"].to(device),
                    batch["size_mask"].to(device),
                )
                vloss += float(loss.item()) * s2.size(0)
                vn += s2.size(0)
        tr, va = running / max(n, 1), vloss / max(vn, 1)
        print(f"epoch {epoch}: train_loss={tr:.4f} val_loss={va:.4f}")
        ckpt = {
            "epoch": epoch,
            "emb_dim": emb,
            "patch_size": args.patch_size,
            "mid_dim": args.mid_dim,
            "size": args.size,
            "band_gain": model.band_gain.detach().cpu(),
            "head": model.head.state_dict(),
            "val_loss": va,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if va < best:
            best = va
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  saved best.pt (val_loss={best:.4f})")


def _load_detector(args: argparse.Namespace, device: torch.device) -> OlmoOpticalDetector:
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    backbone = load_model_from_path(args.weights).to(device)
    model = OlmoOpticalDetector(
        backbone,
        int(ckpt["emb_dim"]),
        int(ckpt.get("patch_size", args.patch_size)),
        int(ckpt.get("mid_dim", 256)),
    ).to(device)
    model.band_gain.data.copy_(ckpt["band_gain"].to(device))
    model.head.load_state_dict(ckpt["head"])
    model.eval()
    return model


@torch.no_grad()
def detect_scenes(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_detector(args, device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tile = int(ckpt.get("size", args.size))
    stride = args.stride or max(tile // 2, 64)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.image_dir)
    paths = sorted(
        [
            *img_dir.glob("*.tif"),
            *img_dir.glob("*.tiff"),
            *img_dir.glob("*.jpg"),
            *img_dir.glob("*.png"),
        ]
    )
    all_rows: list[dict[str, object]] = []
    for path in tqdm(paths, desc="detect"):
        rgb = load_rgb(path)
        H, W = rgb.shape[:2]
        scene_boxes: list[tuple[float, float, float, float, float]] = []
        def starts(length: int) -> list[int]:
            last = max(length - tile, 0)
            values = list(range(0, last + 1, stride))
            if not values or values[-1] != last:
                values.append(last)
            return values

        for y0 in starts(H):
            for x0 in starts(W):
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                patch = np.zeros((tile, tile, 3), dtype=np.float32)
                patch[: y1 - y0, : x1 - x0] = rgb[y0:y1, x0:x1]
                s2 = torch.from_numpy(rgb_to_s2_l2a(patch)).unsqueeze(0).to(device)
                out = model(s2)
                for x_a, y_a, x_b, y_b, sc in decode_boxes(
                    out["center"], out["size"], args.conf, args.topk, args.nms
                )[0]:
                    scene_boxes.append(
                        (x_a + x0, y_a + y0, x_b + x0, y_b + y0, sc)
                    )
        # global NMS
        kept: list[tuple[float, float, float, float, float]] = []
        for box in sorted(scene_boxes, key=lambda t: -t[4]):
            cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
            if any(
                math.hypot(cx - 0.5 * (k[0] + k[2]), cy - 0.5 * (k[1] + k[3])) < args.nms
                for k in kept
            ):
                continue
            kept.append(box)
        # optional crop dump
        crop_dir = out_dir / "crops" / path.stem
        if args.save_crops:
            crop_dir.mkdir(parents=True, exist_ok=True)
        for i, (x1, y1, x2, y2, sc) in enumerate(kept):
            row = {
                "scene": path.name,
                "tank_id": i,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": sc,
            }
            all_rows.append(row)
            if args.save_crops:
                pad = args.crop_pad
                xa, ya = max(0, int(x1) - pad), max(0, int(y1) - pad)
                xb, yb = min(W, int(x2) + pad), min(H, int(y2) + pad)
                crop = (rgb[ya:yb, xa:xb] * 255).clip(0, 255).astype(np.uint8)
                suffix = {"tif": ".tif", "png": ".png", "jpg": ".jpg"}[
                    args.crop_format
                ]
                Image.fromarray(crop).save(crop_dir / f"tank_{i:04d}{suffix}")
        print(f"{path.name}: {len(kept)} tanks")
    pd.DataFrame(all_rows).to_csv(out_dir / "boxes.csv", index=False)
    print(f"wrote {out_dir / 'boxes.csv'}")


def _index_reference_images(root: Path | None) -> dict[str, Path]:
    if root is None:
        return {}
    paths = sorted(
        p
        for pattern in ("*.tif", "*.tiff", "*.jpg", "*.png")
        for p in root.rglob(pattern)
    )
    index: dict[str, Path] = {}
    for path in paths:
        index.setdefault(path.relative_to(root).as_posix(), path)
        index.setdefault(path.name, path)
        index.setdefault(path.stem, path)
    return index


def _find_radius_reference(
    path: Path,
    crop_dir: Path,
    reference_index: dict[str, Path],
) -> Path | None:
    if not reference_index:
        return None
    relative = path.relative_to(crop_dir)
    return (
        reference_index.get(relative.as_posix())
        or reference_index.get(relative.name)
        or reference_index.get(relative.stem)
    )


def _candidate_rank(row: dict[str, object]) -> tuple[int, int, int]:
    """Prefer a valid boundary result; otherwise use the strongest arc match."""
    return (
        int(row.get("geometry_valid") is True),
        int(row.get("shadow_method") == "legacy_boundary_scan"),
        int(row.get("outer_edge_points", 0))
        + int(row.get("inner_edge_points", 0))
        + int(row.get("outer_arc_score", 0)),
    )


def _date_dirs(data_root: Path, branches: list[str]) -> list[tuple[str, Path]]:
    dates: list[tuple[str, Path]] = []
    for branch in branches:
        branch_dir = data_root / branch
        if not branch_dir.is_dir():
            raise FileNotFoundError(branch_dir)
        for date_dir in sorted(p for p in branch_dir.iterdir() if p.is_dir()):
            if (date_dir / "no_shadow_split_tank").is_dir() and (
                date_dir / "shadow_split_tank"
            ).is_dir():
                dates.append((branch, date_dir))
    return dates


def volume_optical(args: argparse.Namespace) -> None:
    """Run the ported E:/code geometry directly on ADD_with_metadata."""
    data_root = Path(args.data_root)
    date_dirs = _date_dirs(data_root, args.branches)
    if not date_dirs:
        raise FileNotFoundError(f"no optical date folders under {data_root}")

    rows: list[dict[str, object]] = []
    total_tanks = sum(len(index_tank_pairs(path)) for _, path in date_dirs)
    progress = tqdm(total=total_tanks, desc="optical-volume")
    stop = False
    for branch, date_dir in date_dirs:
        info_dir = date_dir / "more_info"
        try:
            metadata = parse_optical_metadata(info_dir)
            extend_px = read_extend_px(info_dir)
        except (FileNotFoundError, ValueError) as exc:
            for tank_id, (no_path, shadow_candidates) in index_tank_pairs(date_dir).items():
                rows.append(
                    {
                        "branch": branch,
                        "date": date_dir.name,
                        "tank_id": tank_id,
                        "no_shadow_file": no_path.name,
                        "shadow_file": shadow_candidates[0][0].name if shadow_candidates else "",
                        "geometry_valid": False,
                        "geometry_status": f"metadata_error: {exc}",
                    }
                )
                progress.update(1)
            continue

        for tank_id, (no_path, shadow_candidates) in index_tank_pairs(date_dir).items():
            if args.limit is not None and len(rows) >= args.limit:
                stop = True
                break
            base: dict[str, object] = {
                "branch": branch,
                "date": date_dir.name,
                "tank_id": tank_id,
                "no_shadow_file": no_path.name,
                "candidate_count": len(shadow_candidates),
                "extend_px": extend_px,
                **to_dict(metadata),
            }
            try:
                circle = detect_tank_circle(read_rgb_u8(no_path))
            except (OSError, ValueError) as exc:
                circle = None
                base.update(geometry_valid=False, geometry_status=f"image_error: {exc}")

            candidate_rows: list[dict[str, object]] = []
            if circle is not None:
                circle_values = {
                    "circle_cx_px": circle.cx_px,
                    "circle_cy_no_shadow_px": circle.cy_px,
                    "circle_cy_shadow_px": circle.cy_px + extend_px,
                    "radius_px": circle.radius_px,
                    "radius_method": circle.method,
                    "circle_edge_support": circle.edge_support,
                }
                if not shadow_candidates:
                    candidate_rows.append(
                        {
                            **base,
                            **circle_values,
                            "shadow_file": "",
                            "overlap_mode": "",
                            "geometry_valid": False,
                            "geometry_status": "missing_shadow_pair",
                        }
                    )
                else:
                    for shadow_path, mode in shadow_candidates:
                        row = {
                            **base,
                            **circle_values,
                            "shadow_file": shadow_path.name,
                            "overlap_mode": mode,
                        }
                        try:
                            shadow = measure_shadow(
                                read_rgb_u8(shadow_path),
                                circle,
                                extend_px,
                                mode,
                                adjust_px=args.circle_adjust_px,
                            )
                            row.update(
                                {
                                    "Lex_px": shadow.lex_px,
                                    "Lin_px": shadow.lin_px,
                                    "shadow_method": shadow.method,
                                    "outer_edge_points": shadow.outer_edge_points,
                                    "inner_edge_points": shadow.inner_edge_points,
                                    "outer_arc_score": shadow.outer_arc_score,
                                    "inner_arc_score": shadow.inner_arc_score,
                                    "scale_used_m_per_px": metadata.row_gsd_m,
                                }
                            )
                            if shadow.valid:
                                row.update(
                                    volume_from_optical(
                                        r_m=circle.radius_px * metadata.row_gsd_m,
                                        lex_m=shadow.lex_px * metadata.row_gsd_m,
                                        lin_m=shadow.lin_px * metadata.row_gsd_m,
                                        solar_elev_deg=metadata.solar_elev_deg,
                                        sat_elev_deg=metadata.sat_elev_deg,
                                        solar_az_deg=metadata.solar_az_deg,
                                        sat_az_deg=metadata.sat_az_deg,
                                    )
                                )
                            else:
                                row.update(
                                    geometry_valid=False,
                                    geometry_status=shadow.status,
                                )
                        except (OSError, ValueError, cv2.error) as exc:
                            row.update(
                                geometry_valid=False,
                                geometry_status=f"shadow_error: {exc}",
                            )
                        candidate_rows.append(row)
            else:
                candidate_rows.append(base)

            if candidate_rows:
                rows.append(max(candidate_rows, key=_candidate_rank))
            progress.update(1)
        if stop:
            break
    progress.close()

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    valid = sum(row.get("geometry_valid") is True for row in rows)
    print(f"wrote {out}  valid={valid}/{len(rows)}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train-det", help="Train optical DetectHead on frozen OlmoEarth")
    tr.add_argument("--data-root", required=True)
    tr.add_argument("--weights", required=True)
    tr.add_argument("--out-dir", required=True)
    tr.add_argument("--size", type=int, default=1024)
    tr.add_argument("--patch-size", type=int, default=4)
    tr.add_argument("--mid-dim", type=int, default=256)
    tr.add_argument("--batch-size", type=int, default=2)
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--workers", type=int, default=2)

    de = sub.add_parser("detect", help="Sliding-window detect on full optical scenes")
    de.add_argument("--weights", required=True)
    de.add_argument("--ckpt", required=True)
    de.add_argument("--image-dir", required=True)
    de.add_argument("--out-dir", required=True)
    de.add_argument("--size", type=int, default=512)
    de.add_argument("--patch-size", type=int, default=4)
    de.add_argument("--stride", type=int, default=None)
    de.add_argument("--conf", type=float, default=0.3)
    de.add_argument("--topk", type=int, default=50)
    de.add_argument("--nms", type=float, default=24.0)
    de.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=True)
    de.add_argument("--crop-pad", type=int, default=16)
    de.add_argument(
        "--crop-format",
        choices=["tif", "png", "jpg"],
        default="tif",
        help="Lossless tif is preferred because JPEG can damage thin shadow edges",
    )

    vo = sub.add_parser(
        "volume",
        help="Run the ported optical geometry on ADD_with_metadata date folders",
    )
    vo.add_argument("--data-root", required=True)
    vo.add_argument(
        "--branches",
        nargs="+",
        default=["ADD_with_metadata", "with_metadata"],
        help="Dataset branches containing date folders (no_metadata is skipped by default)",
    )
    vo.add_argument("--out-csv", required=True)
    vo.add_argument(
        "--circle-adjust-px",
        type=int,
        default=5,
        help="Legacy circle-mask vertical correction",
    )
    vo.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.cmd == "train-det":
        train_det(args)
    elif args.cmd == "detect":
        detect_scenes(args)
    elif args.cmd == "volume":
        volume_optical(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
