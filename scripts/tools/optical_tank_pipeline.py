"""Optical floating-roof tank pipeline on frozen OlmoEarth.

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
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


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
    shadow_az_deg: float,
    proj_az_deg: float,
) -> float:
    """Denominator of paper Eq.1-2."""
    a = math.radians(solar_elev_deg)
    b = math.radians(sat_elev_deg)
    g = math.radians(shadow_az_deg)
    t = math.radians(proj_az_deg)
    ca, cb = 1.0 / max(math.tan(a), 1e-6), 1.0 / max(math.tan(b), 1e-6)
    inside = ca * ca + cb * cb - 2 * ca * cb * math.cos(g - t)
    return math.sqrt(max(inside, 1e-8))


def volume_from_optical(
    r_m: float,
    lex_m: float,
    lin_m: float,
    solar_elev_deg: float,
    sat_elev_deg: float,
    shadow_az_deg: float,
    proj_az_deg: float,
) -> dict[str, float | bool | str]:
    denom = optical_height_factor(
        solar_elev_deg, sat_elev_deg, shadow_az_deg, proj_az_deg
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


def fit_circle_radius_px(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Hough if OpenCV available, else equivalent radius from bright/tank mask."""
    g = gray
    if g.ndim == 3:
        g = (0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]).astype(np.float32)
    if mask is None:
        # crude tank: brighter central blob via Otsu-ish threshold
        flat = g.reshape(-1)
        thr = float(np.percentile(flat, 60))
        mask = (g >= thr).astype(np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8)
    if _HAS_CV2:
        edges = cv2.Canny((g * 255).clip(0, 255).astype(np.uint8), 50, 150)
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(8, min(g.shape) // 4),
            param1=100,
            param2=30,
            minRadius=max(4, min(g.shape) // 20),
            maxRadius=min(g.shape) // 2,
        )
        if circles is not None:
            return float(np.median(circles[0, :, 2]))
    area = float(mask.sum())
    return float(math.sqrt(area / math.pi)) if area > 0 else 0.0


def measure_shadow_lengths(
    rgb: np.ndarray,
    cx: float,
    cy: float,
    shadow_az_deg: float,
    r_px: float,
) -> tuple[float, float]:
    """Estimate Lex (external) and Lin (internal) in pixels along shadow azimuth.

    Simplified OilWatch idea: dark pixels along the solar-shadow direction
    outside the roof circle (Lex) and a short inward segment (Lin).
    """
    if rgb.ndim == 3:
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        gray = rgb.astype(np.float32)
    h, w = gray.shape
    # shadow direction on image (azimuth from north, clockwise → approx)
    rad = math.radians(shadow_az_deg)
    dx, dy = math.sin(rad), -math.cos(rad)
    thr = float(np.percentile(gray, 35))

    def ray_dark_len(x0: float, y0: float, max_step: int, inward: bool = False) -> float:
        sx, sy = (-dx, -dy) if inward else (dx, dy)
        length = 0.0
        for s in range(1, max_step + 1):
            x, y = x0 + sx * s, y0 + sy * s
            ix, iy = int(round(x)), int(round(y))
            if not (0 <= ix < w and 0 <= iy < h):
                break
            if gray[iy, ix] > thr:
                if length > 0:
                    break
                continue
            length += 1.0
        return length

    # start just outside / inside roof radius along shadow dir
    ox, oy = cx + dx * (r_px + 1), cy + dy * (r_px + 1)
    ix, iy = cx - dx * max(r_px * 0.3, 2), cy - dy * max(r_px * 0.3, 2)
    lex = ray_dark_len(ox, oy, max_step=int(max(h, w) * 0.5), inward=False)
    lin = ray_dark_len(ix, iy, max_step=int(r_px), inward=True)
    return lex, lin


def load_optical_meta(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None or not path.is_file():
        return {}
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    need = {
        "filename",
        "solar_elev_deg",
        "sat_elev_deg",
        "shadow_az_deg",
        "proj_az_deg",
    }
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"optical meta missing {sorted(missing)}")
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        fn = str(row["filename"])
        entry = {
            "solar_elev_deg": float(row["solar_elev_deg"]),
            "sat_elev_deg": float(row["sat_elev_deg"]),
            "shadow_az_deg": float(row["shadow_az_deg"]),
            "proj_az_deg": float(row["proj_az_deg"]),
            "pixel_resolution": float(row.get("pixel_resolution", 0.75)),
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
        for y0 in range(0, max(H - tile, 0) + 1, stride):
            for x0 in range(0, max(W - tile, 0) + 1, stride):
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
                Image.fromarray(crop).save(crop_dir / f"tank_{i:04d}.jpg")
        print(f"{path.name}: {len(kept)} tanks")
    pd.DataFrame(all_rows).to_csv(out_dir / "boxes.csv", index=False)
    print(f"wrote {out_dir / 'boxes.csv'}")


def volume_optical(args: argparse.Namespace) -> None:
    crop_dir = Path(args.crop_dir)
    meta = load_optical_meta(Path(args.meta) if args.meta else None)
    paths = sorted(
        [
            *crop_dir.rglob("*.tif"),
            *crop_dir.rglob("*.tiff"),
            *crop_dir.rglob("*.jpg"),
            *crop_dir.rglob("*.png"),
        ]
    )
    if not paths:
        raise FileNotFoundError(crop_dir)
    rows: list[dict[str, object]] = []
    skipped = 0
    for path in tqdm(paths, desc="optical-volume"):
        rgb = load_rgb(path)
        r_px = fit_circle_radius_px(rgb)
        cx, cy = rgb.shape[1] / 2.0, rgb.shape[0] / 2.0
        key = path.name
        m = meta.get(key) or meta.get(path.stem)
        # also try parent scene stem for detect crops tank_0001 under scene/
        if m is None and path.parent.name:
            m = meta.get(path.parent.name)
        if m is None:
            skipped += 1
            rows.append(
                {
                    "filename": str(path.relative_to(crop_dir)),
                    "R_mask_px": r_px,
                    "geometry_status": "missing_meta",
                }
            )
            continue
        lex_px, lin_px = measure_shadow_lengths(
            rgb, cx, cy, m["shadow_az_deg"], r_px
        )
        sr = float(m["pixel_resolution"])
        geom = volume_from_optical(
            r_m=sr * r_px,
            lex_m=sr * lex_px,
            lin_m=sr * lin_px,
            solar_elev_deg=m["solar_elev_deg"],
            sat_elev_deg=m["sat_elev_deg"],
            shadow_az_deg=m["shadow_az_deg"],
            proj_az_deg=m["proj_az_deg"],
        )
        rows.append(
            {
                "filename": str(path.relative_to(crop_dir)),
                "R_mask_px": r_px,
                "Lex_px": lex_px,
                "Lin_px": lin_px,
                "sr": sr,
                **geom,
            }
        )
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    valid = sum(1 for r in rows if r.get("geometry_valid") is True)
    print(f"wrote {out}  valid={valid}/{len(rows)}  missing_meta={skipped}")


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

    vo = sub.add_parser("volume", help="Optical shadow geometry -> V on crops")
    vo.add_argument("--crop-dir", required=True)
    vo.add_argument("--meta", required=True, help="xlsx/csv with solar/sat angles")
    vo.add_argument("--out-csv", required=True)
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
