"""Frozen OlmoEarth + local detail/CBAM branch for tank geometry.

Version 2 follows the supplied OilWatch paper, Fig.5 and p18 Eq.3-5.
The paper's h is floating-roof descent; stored depth is H-h. See
docs/HH-Tank-Geometry.md for migration, validation and Linux commands.

Supervised targets (need labels)::

    mask/          -> seg
    key/*.json     -> O1/O2/O3 heatmaps

NOT supervised::

    volume V  — computed after inference from geometry + imaging meta
                (Sr, incidence angle). Meta is only for this post-process,
                not for the training loss.

Pipeline::

    tokens -> DualTaskHead -> mask + kp heatmaps
    optional --meta (.xlsx/.csv) -> R,H,h,V  (eval / --save-preds only)

Example::

    # train: no meta needed
    python scripts/tools/train_hh_tank_geom.py train \\
      --data-root .../Single_Tank_Oil_Estimation_train_split \\
      --weights .../OlmoEarth-v1_2-Base \\
      --out-dir runs/tank_geom

    # eval + volume: pass meta
    python scripts/tools/train_hh_tank_geom.py eval \\
      --ckpt runs/tank_geom/best.pt --weights ... --data-root ... \\
      --meta .../test/meta.xlsx --save-preds pred_test
"""

from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class CBAM(nn.Module):
    """Channel and spatial attention inspired by OilWatch Fig. 5."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 8, 4)
        self.channel = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, channels, 1)
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.channel(F.adaptive_avg_pool2d(x, 1))
        weight = weight + self.channel(F.adaptive_max_pool2d(x, 1))
        x = x * weight.sigmoid()
        summary = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1)
        return x * self.spatial(summary).sigmoid()


# ---------------------------------------------------------------------------
# Frozen satellite features + optional local detail/CBAM + dual task decoders
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


class SharedUpsampleNeck(nn.Module):
    """Progressive upsample from ViT tokens to pixel resolution."""

    def __init__(self, in_dim: int, mid_dim: int, patch_size: int) -> None:
        if patch_size < 1 or (patch_size & (patch_size - 1)) != 0:
            raise ValueError(f"patch_size must be power of two, got {patch_size}")
        super().__init__()
        n_stages = int(math.log2(patch_size))
        layers: list[nn.Module] = [ConvBNReLU(in_dim, mid_dim)]
        ch = mid_dim
        for _ in range(n_stages):
            next_ch = max(ch // 2, 64)
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvBNReLU(ch, next_ch),
            ]
            ch = next_ch
        self.net = nn.Sequential(*layers)
        self.out_dim = ch

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens.permute(0, 3, 1, 2).contiguous())


class TaskDecoder(nn.Module):
    """A few conv blocks specialized to one task (mask or keypoints)."""

    def __init__(self, in_ch: int, out_ch: int, depth: int = 3) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        ch = in_ch
        for _ in range(depth):
            blocks.append(ConvBNReLU(ch, in_ch))
            ch = in_ch
        blocks.append(nn.Conv2d(ch, out_ch, kernel_size=1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualTaskHead(nn.Module):
    """Shared neck + mask decoder + keypoint decoder."""

    def __init__(
        self,
        in_dim: int,
        patch_size: int,
        mid_dim: int = 256,
        decoder_depth: int = 3,
        detail_branch: bool = False,
    ) -> None:
        super().__init__()
        self.neck = SharedUpsampleNeck(in_dim, mid_dim, patch_size)
        feat_ch = self.neck.out_dim
        self.detail_branch = detail_branch
        if detail_branch:
            # Preserve fine scattering cues lost by patch tokenization. This is
            # a local skip branch, not a reproduction of the paper's full U-Net.
            self.detail = nn.Sequential(
                nn.Conv2d(2, 32, 3, padding=1),
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                CBAM(32),
            )
            self.fuse = nn.Sequential(
                nn.Conv2d(feat_ch + 32, feat_ch, 3, padding=1),
                nn.GroupNorm(8, feat_ch),
                nn.ReLU(),
                CBAM(feat_ch),
            )
        self.mask_decoder = TaskDecoder(feat_ch, out_ch=2, depth=decoder_depth)
        self.kp_decoder = TaskDecoder(feat_ch, out_ch=3, depth=decoder_depth)

    def forward(
        self, tokens: torch.Tensor, images: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        feat = self.neck(tokens)
        if self.detail_branch:
            if images is None:
                raise ValueError("detail branch requires input images")
            detail = self.detail(images)
            feat = F.interpolate(
                feat, size=detail.shape[-2:], mode="bilinear", align_corners=False
            )
            feat = self.fuse(torch.cat([feat, detail], dim=1))
        return {
            "seg_logits": self.mask_decoder(feat),
            "kp_heatmaps": self.kp_decoder(feat),
        }


class HHTankGeomModel(nn.Module):
    """Frozen OlmoEarth + HH proj + DualTaskHead."""

    def __init__(
        self,
        backbone: nn.Module,
        emb_dim: int,
        patch_size: int = 4,
        mid_dim: int = 256,
        decoder_depth: int = 3,
        detail_branch: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(2, 2, kernel_size=1, bias=True)
        nn.init.eye_(self.proj.weight[:, :, 0, 0])
        nn.init.zeros_(self.proj.bias)
        self.head = DualTaskHead(
            emb_dim,
            patch_size,
            mid_dim=mid_dim,
            decoder_depth=decoder_depth,
            detail_branch=detail_branch,
        )

    def train(self, mode: bool = True):  # noqa: A003
        super().train(mode)
        self.backbone.eval()
        return self

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

        b, _, h, w = images.shape
        x = self.proj(images).permute(0, 2, 3, 1).unsqueeze(3)
        sample = MaskedOlmoEarthSample(
            sentinel1=x,
            sentinel1_mask=torch.ones(b, h, w, 1, 1, device=x.device, dtype=x.dtype)
            * MaskValue.ONLINE_ENCODER.value,
            timestamps=torch.tensor(
                [[[1, 0, 2020]]] * b, device=x.device, dtype=torch.long
            ),
        )
        # Frozen parameters still need autograd with respect to the trainable
        # input projection. eval callers already run under torch.no_grad().
        out = self.backbone.encoder(sample, fast_pass=True, patch_size=self.patch_size)
        return out["tokens_and_masks"].sentinel1.mean(dim=(3, 4))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.encode(images), images)


# ---------------------------------------------------------------------------
# Data (supervised: image + mask + keypoints only)
# ---------------------------------------------------------------------------


def _read_tif(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read()
    return arr[0] if arr.ndim == 3 else arr


def _binarize_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float32)
    return (m > 0).astype(np.float32) if m.max() > 1 else (m > 0.5).astype(np.float32)


def _fit_square_simple(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[-2], arr.shape[-1]
    pad_h, pad_w = max(0, size - h), max(0, size - w)
    if pad_h or pad_w:
        arr = np.pad(
            arr,
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
            mode="reflect",
        )
        h, w = arr.shape[-2], arr.shape[-1]
    y0, x0 = max(0, (h - size) // 2), max(0, (w - size) // 2)
    return arr[y0 : y0 + size, x0 : x0 + size]


def _origin_offset(shape: tuple[int, ...], size: int) -> tuple[float, float]:
    h, w = shape[-2], shape[-1]
    pad_h, pad_w = max(0, size - h), max(0, size - w)
    pad_top, pad_left = pad_h // 2, pad_w // 2
    hp, wp = h + pad_h, w + pad_w
    y0, x0 = max(0, (hp - size) // 2), max(0, (wp - size) // 2)
    return float(pad_top - y0), float(pad_left - x0)


def letterbox(
    arr: np.ndarray, size: int, is_mask: bool = False
) -> tuple[np.ndarray, float, float, float]:
    """Pad to a square, then resize isotropically; return scale and xy offsets.

    align_corners=False maps pixel centres as x'=(x+pad+0.5)*scale-0.5.
    Segmentation padding is ignored in the loss, never reflected into new roofs.
    """
    h, w = arr.shape
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    padded = np.pad(
        arr.astype(np.float32),
        ((top, side - h - top), (left, side - w - left)),
        mode="constant",
        constant_values=-100 if is_mask else 0,
    )
    x = torch.from_numpy(padded)[None, None]
    if is_mask:
        x = F.interpolate(x, size=(size, size), mode="nearest")
    else:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    scale = size / side
    return x[0, 0].numpy(), scale, (left + 0.5) * scale - 0.5, (top + 0.5) * scale - 0.5


def restore_mask(
    mask: np.ndarray, shape: tuple[int, int], preprocess: str
) -> np.ndarray:
    """Map a network mask to original pixels before radius fitting/export."""
    h, w = shape
    size = mask.shape[0]
    if preprocess == "letterbox":
        side = max(h, w)
        full = F.interpolate(
            torch.from_numpy(mask.astype(np.float32))[None, None],
            size=(side, side),
            mode="nearest",
        )[0, 0].numpy()
        top, left = (side - h) // 2, (side - w) // 2
        return full[top : top + h, left : left + w].astype(np.uint8)
    oy, ox = _origin_offset(shape, size)
    result = np.zeros(shape, dtype=np.uint8)
    yy, xx = np.indices(shape)
    sy, sx = yy + int(oy), xx + int(ox)
    valid = (sy >= 0) & (sx >= 0) & (sy < size) & (sx < size)
    result[valid] = mask[sy[valid], sx[valid]]
    return result


def _gaussian_heatmap(
    h: int, w: int, x: float, y: float, sigma: float = 2.0
) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    heat = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    return heat.astype(np.float32)


def _parse_key_json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pts: dict[str, tuple[float, float]] = {}
    circle: tuple[float, float, float] | None = None
    for s in data["shapes"]:
        label, st, p = s["label"], s["shape_type"], s["points"]
        if st == "point" and label in ("O1", "O2", "O3"):
            pts[label] = (float(p[0][0]), float(p[0][1]))
        elif st == "circle" and len(p) >= 2:
            cx, cy = float(p[0][0]), float(p[0][1])
            px, py = float(p[1][0]), float(p[1][1])
            r = float(math.hypot(px - cx, py - cy))
            circle = (cx, cy, r)
    return {"points": pts, "circle": circle}


class TankGeomDataset(Dataset):
    """Loads image / mask / key only. Imaging meta is NOT part of training."""

    def __init__(
        self,
        root: Path,
        split: str,
        size: int = 128,
        sigma: float = 2.0,
        preprocess: str = "letterbox",
    ) -> None:
        self.split = split
        self.image_dir = root / split / "image"
        self.mask_dir = root / split / "mask"
        self.key_dir = root / split / "key"
        self.size = size
        self.sigma = sigma
        self.preprocess = preprocess
        self.paths = sorted(self.image_dir.glob("*.tif"))
        if not self.paths:
            raise FileNotFoundError(self.image_dir)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        img_path = self.paths[idx]
        name = img_path.name
        stem = img_path.stem
        img = _read_tif(img_path).astype(np.float32)
        mask = _binarize_mask(_read_tif(self.mask_dir / name))
        key = _parse_key_json(self.key_dir / f"{stem}.json")

        if img.shape != mask.shape:
            raise ValueError(f"Image/mask shape mismatch: {name}")
        if self.preprocess == "letterbox":
            img = (img - float(img.mean())) / (float(img.std()) + 1e-6)
            img_f, scale, ox, oy = letterbox(img, self.size)
            mask_f, _, _, _ = letterbox(mask, self.size, is_mask=True)
        elif self.preprocess == "legacy":
            img_f = _fit_square_simple(img, self.size)
            mask_f = _fit_square_simple(mask, self.size)
            oy, ox = _origin_offset(img.shape, self.size)
            scale = 1.0
        else:
            raise ValueError(self.preprocess)

        heats = np.zeros((3, self.size, self.size), dtype=np.float32)
        points = []
        valid = []
        for i, lab in enumerate(("O1", "O2", "O3")):
            if lab not in key["points"]:
                # Missing annotations are unknown, not background targets.
                points.append([0.0, 0.0])
                valid.append(False)
                continue
            x, y = key["points"][lab]  # type: ignore[index]
            x, y = x * scale + ox, y * scale + oy
            points.append([x, y])
            valid.append(0 <= x < self.size and 0 <= y < self.size)
            if self.preprocess == "letterbox" and not valid[-1]:
                raise ValueError(f"Annotated {lab} outside image: {name}")
            heats[i] = _gaussian_heatmap(self.size, self.size, x, y, self.sigma)

        if key["circle"] is not None:
            r_mask_px = float(key["circle"][2])  # type: ignore[index]
        else:
            area = float(mask.sum())
            r_mask_px = math.sqrt(area / math.pi) if area > 0 else 0.0

        if self.preprocess == "legacy":
            std = float(img_f.std())
            img_f = (img_f - float(img_f.mean())) / (std + 1e-6)
        x = np.stack([img_f, img_f], axis=0)

        return {
            "image": torch.from_numpy(x),
            "mask": torch.from_numpy(mask_f).long(),
            "heatmaps": torch.from_numpy(heats),
            "r_mask_px": torch.tensor(r_mask_px, dtype=torch.float32),
            "points": torch.tensor(points, dtype=torch.float32),
            "kp_valid": torch.tensor(valid, dtype=torch.bool),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "offset": torch.tensor([ox, oy], dtype=torch.float32),
            "original_shape": torch.tensor(img.shape, dtype=torch.long),
            "name": name,
            "split": self.split,
        }


def load_meta(path: Path | None) -> dict[str, dict[str, float]]:
    """Map filename (and optional split/filename) -> imaging meta.

    Accepts ``.csv`` or ``.xlsx``. Required columns:
    ``filename``, ``pixel_resolution``, ``incidenceangle``; optional ``split``.
    Prefer ``split/filename`` when ``split`` exists (avoids train/test collisions).
    """
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suf == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"meta must be .csv/.xlsx, got {path}")
    needed = {"filename", "pixel_resolution", "incidenceangle"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"meta missing columns {sorted(missing)} in {path}")
    out: dict[str, dict[str, float]] = {}
    has_split = "split" in df.columns
    for _, row in df.iterrows():
        fn = str(row["filename"])
        entry = {
            "pixel_resolution": float(row["pixel_resolution"]),
            "incidenceangle": float(row["incidenceangle"]),
        }
        if has_split and pd.notna(row["split"]):
            sp = str(row["split"]).strip()
            out[f"{sp}/{fn}"] = entry
            out[f"{sp}/{Path(fn).stem}"] = entry
        out[fn] = entry
        out[Path(fn).stem] = entry
    return out


# backward-compatible alias
load_meta_csv = load_meta


def lookup_meta(
    meta: dict[str, dict[str, float]], split: str, name: str
) -> dict[str, float] | None:
    stem = Path(name).stem
    for key in (f"{split}/{name}", f"{split}/{stem}", name, stem):
        if key in meta:
            return meta[key]
    return None


# ---------------------------------------------------------------------------
# Geometry helpers (post-process only; not in training loss)
# ---------------------------------------------------------------------------


def volume_from_geometry(
    o1: np.ndarray,
    o2: np.ndarray,
    o3: np.ndarray,
    r_mask_px: float,
    sr: float,
    delta_deg: float,
    formula: str = "paper",
) -> dict[str, float | str | bool]:
    """OilWatch p18 Eq.3-5: h is roof descent; oil depth is H-h.

    Invalid geometry is reported as NaN volume, not silently labelled empty.
    legacy reproduces the former one-radius/clamped oil-depth calculation.
    """
    if not math.isfinite(sr) or sr <= 0 or not 0 < delta_deg < 90:
        raise ValueError("Require positive pixel spacing and 0 < incidence angle < 90")
    delta = math.radians(float(delta_deg))
    d12 = float(np.linalg.norm(o1 - o2))
    d13 = float(np.linalg.norm(o1 - o3))
    r_m = sr * r_mask_px / max(math.sin(delta), 1e-6)
    H_m = sr * d12 / max(math.cos(delta), 1e-6)
    if formula == "legacy":
        oil = max(sr * (d13 - r_mask_px) / math.cos(delta), 0.0)
        h_m = oil  # Historical column meaning, only in explicit legacy mode.
        valid = r_m > 0 and math.isfinite(oil)
        status = "legacy_formula"
    elif formula == "paper":
        h_m = sr * (d13 - 2 * r_mask_px) / math.cos(delta)
        oil = H_m - h_m
        valid = all(math.isfinite(v) for v in (r_m, H_m, h_m, oil))
        valid = valid and r_m > 0 and H_m > 0 and 0 <= h_m <= H_m
        status = "ok" if valid else "invalid_geometry"
    else:
        raise ValueError(formula)
    raw_v = math.pi * r_m * r_m * oil
    v_m3 = raw_v if valid else float("nan")
    return {
        "R_m": r_m,
        "H_m": H_m,
        "h_m": h_m,
        "oil_height_raw_m": oil,
        "V_raw_m3": raw_v,
        "V_m3": v_m3,
        "V_bbl": v_m3 / 0.158987,
        "geometry_valid": valid,
        "geometry_status": status,
        "formula": formula,
    }


def heatmaps_to_points(heat: np.ndarray) -> np.ndarray:
    pts = []
    for i in range(heat.shape[0]):
        idx = int(heat[i].reshape(-1).argmax())
        y, x = divmod(idx, heat.shape[-1])
        pts.append([x, y])
    return np.asarray(pts, dtype=np.float32)


def soft_argmax_points(
    heat_logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    b, c, h, w = heat_logits.shape
    flat = heat_logits.view(b, c, -1) / max(temperature, 1e-6)
    prob = F.softmax(flat, dim=-1)
    ys = torch.arange(h, device=heat_logits.device, dtype=heat_logits.dtype)
    xs = torch.arange(w, device=heat_logits.device, dtype=heat_logits.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=0)
    return torch.einsum("bck,dk->bcd", prob, coords)


def decode_points(logits: torch.Tensor, mode: str = "local") -> torch.Tensor:
    """Peak detector (Fig.5), refined only within a 5x5 neighbourhood.

    Local decoding avoids averaging distant peaks or a large background region.
    The spatial classification loss also makes global soft-argmax well defined.
    """
    if mode == "soft":
        return soft_argmax_points(logits)
    b, c, h, w = logits.shape
    peak = logits.reshape(b, c, -1).argmax(-1)
    x, y = peak % w, peak // w
    if mode == "argmax":
        return torch.stack([x, y], -1).to(logits.dtype)
    if mode != "local":
        raise ValueError(mode)
    yy = torch.arange(h, device=logits.device)[None, None, :, None]
    xx = torch.arange(w, device=logits.device)[None, None, None, :]
    local = (abs(xx - x[..., None, None]) <= 2) & (abs(yy - y[..., None, None]) <= 2)
    return soft_argmax_points(logits.masked_fill(~local, -1e4))


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected roof and fill holes; remove remote clutter."""
    from scipy import ndimage

    labels, n = ndimage.label(mask > 0)
    if not n:
        return np.zeros_like(mask, dtype=np.uint8)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return ndimage.binary_fill_holes(labels == counts.argmax()).astype(np.uint8)


def enforce_o123_x_order(xy: np.ndarray) -> np.ndarray:
    """SAR range prior on this dataset: O1.x < O2.x < O3.x.

    Re-label the three predicted peaks by ascending x. Shape (3,2) or (B,3,2).
    """
    single = xy.ndim == 2
    if single:
        xy = xy[None]
    out = np.empty_like(xy)
    for i in range(xy.shape[0]):
        order = np.argsort(xy[i, :, 0])
        out[i] = xy[i, order]
    return out[0] if single else out


def area_radius(mask: np.ndarray) -> float:
    area = float((mask > 0).sum())
    return float(math.sqrt(area / math.pi)) if area > 0 else 0.0


def robust_circle_radius(mask: np.ndarray) -> float:
    """Fit a circle to the main roof contour, iteratively rejecting outliers.

    A deterministic alternative to Fig.5's Hough detector; no radius shrinking
    based on keypoint distances. Fall back to area only for degenerate contours.
    """
    from scipy import ndimage

    mask = clean_mask(mask)
    boundary = (mask > 0) & ~ndimage.binary_erosion(mask > 0)
    y, x = np.nonzero(boundary)
    if len(x) < 8:
        return area_radius(mask)
    pts = np.column_stack([x, y]).astype(np.float64)
    keep = np.ones(len(pts), dtype=bool)
    radius = area_radius(mask)
    for _ in range(5):
        q = pts[keep]
        if len(q) < 8:
            break
        a = np.column_stack([2 * q[:, 0], 2 * q[:, 1], np.ones(len(q))])
        sol, _, rank, _ = np.linalg.lstsq(a, (q * q).sum(1), rcond=None)
        if rank < 3:
            return area_radius(mask)
        radius = math.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 0))
        residual = np.abs(np.linalg.norm(pts - sol[:2], axis=1) - radius)
        median = np.median(residual)
        threshold = max(1.0, median + 3 * 1.4826 * np.median(abs(residual - median)))
        updated = residual <= threshold
        if np.array_equal(updated, keep):
            break
        keep = updated
    return float(radius)


def fit_circle_radius(mask: np.ndarray) -> float:
    """Legacy unfiltered algebraic fit, retained for comparisons."""
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 8:
        return area_radius(mask)
    pad = np.pad(mask > 0, 1, mode="constant")
    edge = []
    for y, x in zip(ys, xs, strict=False):
        if pad[y : y + 3, x : x + 3].min() == 0:
            edge.append((x, y))
    if len(edge) < 8:
        return area_radius(mask)
    pts = np.asarray(edge, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, c = sol
        return float(math.sqrt(max(c + cx * cx + cy * cy, 0.0)))
    except np.linalg.LinAlgError:
        return area_radius(mask)


def resolve_radius(
    mask: np.ndarray,
    r_gt: float,
    mode: str,
    d13: float | None = None,
) -> float:
    """Choose radius without using keypoints to force a valid volume."""
    if mode == "gt":
        return float(r_gt)
    if mode == "robust":
        return robust_circle_radius(mask)
    if mode == "fit":
        r = fit_circle_radius(mask)
    elif mode == "area":
        r = area_radius(mask)
    elif mode == "auto":
        r_area = area_radius(mask)
        r_fit = fit_circle_radius(mask)
        # boundary fit often overestimates on noisy SAR masks; take the smaller.
        r = min(r_area, r_fit) if r_fit > 0 else r_area
        if d13 is not None and d13 <= r and r_area > 0:
            # last resort: shrink toward making h>=0 is wrong physically;
            # keep geometric r but caller will still clamp h.
            r = min(r, r_area)
    else:
        raise ValueError(f"unknown radius mode: {mode}")
    return float(r)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)[:, 1]
    probs = probs * (target != -100)
    tgt = (target == 1).float()
    inter = (probs * tgt).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
    return 1.0 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()


def multitask_loss(
    out: dict[str, torch.Tensor],
    mask: torch.Tensor,
    heatmaps: torch.Tensor,
    lambda_kp: float = 1.0,
    kp_channel_weights: tuple[float, float, float] = (1.0, 1.0, 2.0),
    lambda_coord: float = 1.0,
    points: torch.Tensor | None = None,
    kp_valid: torch.Tensor | None = None,
    lambda_geom: float = 1.0,
) -> torch.Tensor:
    """Spatial distribution CE + normalized coordinate/distance supervision.

    Unlike dense sigmoid MSE, background pixels cannot overwhelm a tiny peak.
    Geometry loss supervises measured distances, not an assumed x-order or
    a manufactured positive volume. No test metadata is used in training.
    """
    seg = F.cross_entropy(out["seg_logits"], mask) + dice_loss_with_logits(
        out["seg_logits"], mask
    )
    logits = out["kp_heatmaps"]
    target = heatmaps.flatten(2)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
    weight = logits.new_tensor(kp_channel_weights)[None]
    if kp_valid is not None:
        weight = weight * kp_valid
    ce = -(target * F.log_softmax(logits.flatten(2), dim=-1)).sum(-1)
    kp = (ce * weight).sum() / weight.expand_as(ce).sum().clamp_min(1)
    pred_xy = soft_argmax_points(logits)
    gt_xy = (
        points
        if points is not None
        else soft_argmax_points(heatmaps.clamp_min(1e-8).log())
    )
    size = logits.shape[-1]
    coord_err = ((pred_xy - gt_xy) / size).abs().mean(-1)
    coord = (coord_err * weight).sum() / weight.expand_as(coord_err).sum().clamp_min(1)
    pdist = torch.linalg.vector_norm(pred_xy[:, 1:] - pred_xy[:, :1], dim=-1) / size
    gdist = torch.linalg.vector_norm(gt_xy[:, 1:] - gt_xy[:, :1], dim=-1) / size
    pairs = (
        torch.ones_like(pdist)
        if kp_valid is None
        else (kp_valid[:, 1:] & kp_valid[:, :1]).float()
    )
    geom = (abs(pdist - gdist) * pairs).sum() / pairs.sum().clamp_min(1)
    return seg + lambda_kp * kp + lambda_coord * coord + lambda_geom * geom


@torch.no_grad()
def _discover_emb_dim(
    backbone: nn.Module, patch_size: int, device: torch.device
) -> int:
    m = HHTankGeomModel(backbone, 768, patch_size).to(device)
    return int(m.encode(torch.zeros(1, 2, 64, 64, device=device)).shape[-1])


def build_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[nn.Module, dict]:
    from olmoearth_pretrain.model_loader import load_model_from_path

    if not args.weights:
        raise ValueError("--weights is required (OlmoEarth checkpoint dir)")
    backbone = load_model_from_path(args.weights).to(device)
    emb_dim = _discover_emb_dim(backbone, args.patch_size, device)
    model = HHTankGeomModel(
        backbone,
        emb_dim,
        patch_size=args.patch_size,
        mid_dim=args.mid_dim,
        decoder_depth=args.decoder_depth,
        detail_branch=args.detail_branch,
    ).to(device)
    return model, {"emb_dim": emb_dim, "patch_size": args.patch_size}


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "best.pt").exists() or (out_dir / "last.pt").exists():
        raise FileExistsError(f"Use a new --out-dir to preserve checkpoints: {out_dir}")

    model, model_meta = build_model(args, device)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    train_ds = TankGeomDataset(Path(args.data_root), "train", args.size)
    val_ds = TankGeomDataset(Path(args.data_root), "test", args.size)
    split_manifest = {
        "seed": args.seed,
        "validation_split": "test",
        "train": [f"train/{p.name}" for p in train_ds.paths],
        "val": [f"test/{p.name}" for p in val_ds.paths],
    }
    (out_dir / "split.json").write_text(json.dumps(split_manifest, indent=2))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"encoder=OlmoEarth(frozen) emb_dim={model_meta['emb_dim']} "
        f"trainable={n_params:,} train={len(train_ds)} val(test)={len(val_ds)}"
    )

    best = -1e9
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_options = {
        "lambda_kp": args.lambda_kp,
        "lambda_coord": args.lambda_coord,
        "lambda_geom": args.lambda_geom,
        "kp_channel_weights": (1.0, 1.0, args.o3_weight),
    }
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            heats = batch["heatmaps"].to(device)
            out = model(images)
            loss = multitask_loss(
                out,
                masks,
                heats,
                points=batch["points"].to(device),
                kp_valid=batch["kp_valid"].to(device),
                **loss_options,
            )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=5.0
            )
            optim.step()
            running += float(loss.item()) * images.size(0)
            n += images.size(0)

        metrics = evaluate(model, val_loader, device, loss_options=loss_options)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": running / max(n, 1), **metrics})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(
            f"epoch {epoch}: train_loss={running / max(n, 1):.4f}  "
            f"val_loss={metrics['loss']:.4f}  miou={metrics['miou']:.4f}  "
            f"kp_px_err={metrics['kp_px_err']:.2f}"
        )
        ckpt = {
            "epoch": epoch,
            "size": args.size,
            "mid_dim": args.mid_dim,
            "decoder_depth": args.decoder_depth,
            "emb_dim": model_meta["emb_dim"],
            "patch_size": args.patch_size,
            "proj": model.proj.state_dict(),
            "head": model.head.state_dict(),
            "metrics": metrics,
            "format_version": 2,
            "preprocess": "letterbox",
            "detail_branch": args.detail_branch,
            "decode": "local",
            "formula": "paper",
            "loss_options": loss_options,
            "training_args": vars(args),
            "split_manifest": split_manifest,
        }
        torch.save(ckpt, out_dir / "last.pt")
        score = metrics["miou"] - 0.01 * (
            metrics["kp_px_err"] + metrics["radius_px_mae"] + metrics["d13_px_mae"]
        )
        if score > best:
            best = score
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  saved best.pt (score={best:.4f})")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    decode: str = "local",
    radius_mode: str = "robust",
    loss_options: dict | None = None,
    order_by_x: bool = False,
) -> dict[str, float]:
    """Val metrics for supervised tasks only (no volume)."""
    model.eval()
    total_loss = 0.0
    n = 0
    seg_preds, seg_labs = [], []
    kp_errs: list[float] = []
    per_point = []
    radius_errs, distance_errs = [], []
    invalid_geometry = 0
    gt_invalid_geometry = 0
    incomplete_labels = 0
    dataset = loader.dataset
    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        heats = batch["heatmaps"].to(device)
        out = model(images)
        loss = multitask_loss(
            out,
            masks,
            heats,
            points=batch["points"].to(device),
            kp_valid=batch["kp_valid"].to(device),
            **(loss_options or {}),
        )
        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

        pred = out["seg_logits"].argmax(1).cpu().numpy()
        seg_preds.extend(list(pred))
        seg_labs.extend(list(masks.cpu().numpy()))

        pred_xy = (
            decode_points(
                out["kp_heatmaps"].masked_fill((masks == -100)[:, None], -1e4), decode
            )
            .cpu()
            .numpy()
        )
        if order_by_x:
            pred_xy = enforce_o123_x_order(pred_xy)
        for i in range(bs):
            scale = float(batch["scale"][i])
            offset = batch["offset"][i].numpy()
            pp = (pred_xy[i] - offset) / scale
            gp = (batch["points"][i].numpy() - offset) / scale
            errors = np.linalg.norm(pp - gp, axis=1)
            labelled = batch["kp_valid"][i].numpy()
            errors[~labelled] = np.nan
            per_point.append(errors)
            if labelled.any():
                kp_errs.append(float(np.nanmean(errors)))
            incomplete_labels += not labelled.all()
            original_mask = restore_mask(
                pred[i], tuple(batch["original_shape"][i].tolist()), dataset.preprocess
            )
            radius = resolve_radius(
                original_mask, float(batch["r_mask_px"][i]), radius_mode
            )
            gt_radius = float(batch["r_mask_px"][i])
            radius_errs.append(abs(radius - gt_radius))
            d12, d13 = np.linalg.norm(pp[0] - pp[1]), np.linalg.norm(pp[0] - pp[2])
            gd12, gd13 = np.linalg.norm(gp[0] - gp[1]), np.linalg.norm(gp[0] - gp[2])
            if labelled[0] and labelled[2]:
                distance_errs.append(abs(d13 - gd13))
            invalid_geometry += not (radius > 0 and 0 <= d13 - 2 * radius <= d12)
            if labelled.all():
                gt_invalid_geometry += not (
                    gt_radius > 0 and 0 <= gd13 - 2 * gt_radius <= gd12
                )

    flat_p = np.concatenate([p.reshape(-1) for p in seg_preds])
    flat_l = np.concatenate([l.reshape(-1) for l in seg_labs])
    valid = flat_l != -100
    flat_p, flat_l = flat_p[valid], flat_l[valid]
    ious = []
    for c in (0, 1):
        tp = int(((flat_p == c) & (flat_l == c)).sum())
        fp = int(((flat_p == c) & (flat_l != c)).sum())
        fn = int(((flat_p != c) & (flat_l == c)).sum())
        ious.append(tp / (tp + fp + fn + 1e-8))
    return {
        "loss": total_loss / max(n, 1),
        "miou": float(np.mean(ious)),
        "kp_px_err": float(np.mean(kp_errs)) if kp_errs else 0.0,
        "O1_px_err": float(np.nanmean(np.asarray(per_point)[:, 0])),
        "O2_px_err": float(np.nanmean(np.asarray(per_point)[:, 1])),
        "O3_px_err": float(np.nanmean(np.asarray(per_point)[:, 2])),
        "radius_px_mae": float(np.mean(radius_errs)),
        "d13_px_mae": float(np.mean(distance_errs)) if distance_errs else float("nan"),
        "paper_invalid_geometry_rate": invalid_geometry / max(n, 1),
        "paper_gt_invalid_geometry_rate": gt_invalid_geometry
        / max(n - incomplete_labels, 1),
        "incomplete_keypoint_labels": incomplete_labels,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_shared(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--data-root", required=True)
        sp.add_argument("--weights", required=True, help="OlmoEarth weights dir")
        sp.add_argument("--size", type=int, default=128)
        sp.add_argument("--patch-size", type=int, default=4)
        sp.add_argument("--batch-size", type=int, default=4)
        sp.add_argument("--workers", type=int, default=2)

    tr = sub.add_parser("train", help="Train on mask+keypoints (no meta / no volume)")
    add_shared(tr)
    tr.add_argument("--out-dir", required=True)
    tr.add_argument("--mid-dim", type=int, default=256)
    tr.add_argument("--decoder-depth", type=int, default=3)
    tr.add_argument("--epochs", type=int, default=80)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--lambda-kp", type=float, default=1.0)
    tr.add_argument("--lambda-coord", type=float, default=1.0)
    tr.add_argument("--lambda-geom", type=float, default=1.0)
    tr.add_argument("--o3-weight", type=float, default=2.0)
    tr.add_argument(
        "--detail-branch", action=argparse.BooleanOptionalAction, default=True
    )
    tr.add_argument("--seed", type=int, default=42)

    ev = sub.add_parser("eval", help="Eval mask/kp; optional volume with --meta")
    add_shared(ev)
    ev.add_argument("--ckpt", required=True)
    ev.add_argument("--split", default="test", choices=("train", "test"))
    ev.add_argument(
        "--meta",
        "--meta-csv",
        dest="meta",
        default=None,
        help="Imaging meta .xlsx/.csv (Sr, incidence). Needed for R/H/h/V columns",
    )
    ev.add_argument(
        "--radius-mode",
        choices=("robust", "auto", "area", "fit", "gt"),
        default="robust",
        help="robust=main component contour; gt=diagnostic label radius (not deployable)",
    )
    ev.add_argument(
        "--order-by-x",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Re-label O1/O2/O3 by ascending x (SAR range prior)",
    )
    ev.add_argument(
        "--decode",
        choices=("local", "argmax", "soft"),
        default=None,
        help="Defaults to checkpoint mode; old checkpoints use soft for reproducibility",
    )
    ev.add_argument("--formula", choices=("paper", "legacy"), default="paper")
    ev.add_argument(
        "--save-preds",
        default=None,
        help="Dir for mask GeoTIFFs; volumes.csv gets V only if --meta is set",
    )
    return p


@torch.no_grad()
def eval_only(args: argparse.Namespace) -> None:
    import rasterio
    from olmoearth_pretrain.model_loader import load_model_from_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    backbone = load_model_from_path(args.weights).to(device)
    model = HHTankGeomModel(
        backbone,
        emb_dim=int(ckpt["emb_dim"]),
        patch_size=int(ckpt.get("patch_size", args.patch_size)),
        mid_dim=int(ckpt.get("mid_dim", 256)),
        decoder_depth=int(ckpt.get("decoder_depth", 3)),
        detail_branch=bool(ckpt.get("detail_branch", False)),
    ).to(device)
    model.proj.load_state_dict(ckpt["proj"])
    model.head.load_state_dict(ckpt["head"])
    preprocess = ckpt.get("preprocess", "legacy")
    decode = args.decode or ckpt.get("decode", "soft")
    if preprocess == "legacy":
        warnings.warn(
            "Legacy checkpoint: retaining centre crop and old normalization. "
            "Retrain for full-image letterbox and the detail branch."
        )
    if args.order_by_x:
        warnings.warn(
            "Sorting changes keypoint identities and is not a universal SAR prior."
        )

    ds = TankGeomDataset(
        Path(args.data_root),
        args.split,
        size=int(ckpt.get("size", args.size)),
        preprocess=preprocess,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    metrics = evaluate(
        model,
        loader,
        device,
        decode=decode,
        radius_mode=args.radius_mode,
        loss_options=ckpt.get("loss_options"),
        order_by_x=args.order_by_x,
    )
    print(json.dumps(metrics, indent=2))

    if args.save_preds is None:
        return

    meta = load_meta(Path(args.meta) if args.meta else None)
    pred_dir = Path(args.save_preds)
    pred_dir.mkdir(parents=True, exist_ok=True)
    if (pred_dir / "volumes.csv").exists():
        raise FileExistsError(f"Use a new --save-preds directory: {pred_dir}")
    rows: list[dict[str, object]] = []
    skipped_meta = 0
    n_zero_h = 0
    volume_errors = []
    gt_geometry_invalid = 0
    model.eval()
    for batch in tqdm(loader, desc="save"):
        images = batch["image"].to(device)
        out = model(images)
        masks = out["seg_logits"].argmax(1).cpu().numpy().astype(np.uint8)
        xy = (
            decode_points(
                out["kp_heatmaps"].masked_fill(
                    (batch["mask"].to(device) == -100)[:, None], -1e4
                ),
                decode,
            )
            .cpu()
            .numpy()
        )
        if args.order_by_x:
            xy = enforce_o123_x_order(xy)
        r_gt = batch["r_mask_px"].numpy()
        for i, name in enumerate(batch["name"]):
            shape = tuple(batch["original_shape"][i].tolist())
            m = restore_mask(masks[i], shape, preprocess)
            if args.radius_mode == "robust":
                m = clean_mask(m)
            with rasterio.open(
                pred_dir / name,
                "w",
                driver="GTiff",
                height=m.shape[0],
                width=m.shape[1],
                count=1,
                dtype="uint8",
            ) as dst:
                dst.write(m * 255, 1)

            scale = float(batch["scale"][i])
            offset = batch["offset"][i].numpy()
            o1, o2, o3 = (xy[i] - offset) / scale
            d13 = float(np.linalg.norm(o1 - o3))
            r_use = resolve_radius(m, float(r_gt[i]), args.radius_mode, d13=d13)
            row: dict[str, object] = {
                "filename": name,
                "split": args.split,
                "O1_x": float(o1[0]),
                "O1_y": float(o1[1]),
                "O2_x": float(o2[0]),
                "O2_y": float(o2[1]),
                "O3_x": float(o3[0]),
                "O3_y": float(o3[1]),
                "R_mask_px": r_use,
                "d13_px": d13,
                "coordinate_space": "original_image_pixels",
                "preprocess": preprocess,
                "decode": decode,
                "radius_mode": args.radius_mode,
            }
            mmeta = lookup_meta(meta, args.split, name) if meta else None
            if mmeta is None:
                skipped_meta += 1
                row.update(
                    {
                        "geometry_valid": False,
                        "geometry_status": "missing_meta",
                        "formula": args.formula,
                        "V_m3": float("nan"),
                    }
                )
                rows.append(row)
                continue
            geom = volume_from_geometry(
                o1,
                o2,
                o3,
                r_use,
                float(mmeta["pixel_resolution"]),
                float(mmeta["incidenceangle"]),
                formula=args.formula,
            )
            if not geom["geometry_valid"]:
                n_zero_h += 1
            row["sr"] = mmeta["pixel_resolution"]
            row["delta_deg"] = mmeta["incidenceangle"]
            row.update(geom)
            gt_points = (batch["points"][i].numpy() - offset) / scale
            gt_geom = volume_from_geometry(
                *gt_points,
                float(r_gt[i]),
                float(mmeta["pixel_resolution"]),
                float(mmeta["incidenceangle"]),
                formula=args.formula,
            )
            if not bool(batch["kp_valid"][i].all()):
                gt_geom.update({"geometry_valid": False, "V_m3": float("nan")})
            row["GT_V_m3"] = gt_geom["V_m3"]
            row["GT_geometry_valid"] = gt_geom["geometry_valid"]
            gt_geometry_invalid += not gt_geom["geometry_valid"]
            if geom["geometry_valid"] and gt_geom["geometry_valid"]:
                volume_errors.append(float(geom["V_m3"]) - float(gt_geom["V_m3"]))
            rows.append(row)

    pd.DataFrame(rows).to_csv(pred_dir / "volumes.csv", index=False)
    print(
        f"wrote {pred_dir}  radius_mode={args.radius_mode}  "
        f"order_by_x={args.order_by_x}  invalid_geometry={n_zero_h}/{len(rows)}"
    )
    metrics.update(
        {
            "n_predictions": len(rows),
            "missing_meta": skipped_meta,
            "invalid_geometry": n_zero_h,
            "gt_invalid_geometry": gt_geometry_invalid,
            "valid_volume_pairs": len(volume_errors),
            "volume_mae_valid_pairs_m3": (
                float(np.mean(np.abs(volume_errors))) if volume_errors else None
            ),
            "volume_rmse_valid_pairs_m3": (
                float(np.sqrt(np.mean(np.square(volume_errors))))
                if volume_errors
                else None
            ),
            "formula": args.formula,
            "decode": decode,
            "preprocess": preprocess,
        }
    )
    (pred_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if not meta:
        print("volumes.csv has keypoints only (pass --meta to compute R/H/h/V)")
    elif skipped_meta:
        print(f"{skipped_meta}/{len(rows)} rows missing meta")


def main() -> None:
    args = _build_parser().parse_args()
    if (
        args.size <= 0
        or args.patch_size not in (1, 2, 4, 8)
        or args.size % args.patch_size
    ):
        raise ValueError("size must be positive and divisible by patch-size (1,2,4,8)")
    if args.cmd == "train":
        if args.epochs < 1 or args.lr <= 0:
            raise ValueError("epochs and learning rate must be positive")
        train(args)
    elif args.cmd == "eval":
        eval_only(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
