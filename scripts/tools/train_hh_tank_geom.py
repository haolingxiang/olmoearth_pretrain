"""Frozen OlmoEarth encoder + dual-task head for tank geometry.

Supervised targets (need labels)::

    mask/          -> seg
    key/*.json     -> O1/O2/O3 heatmaps

NOT supervised::

    volume V  — computed after inference from geometry + imaging meta
                (Sr, incidence angle). Meta is only for this post-process,
                not for the training loss.

Pipeline::

    tokens -> DualTaskHead -> mask + kp heatmaps
    optional meta-csv -> R,H,h,V  (eval / --save-preds only)

Example::

    # train: no meta needed
    python scripts/tools/train_hh_tank_geom.py train \\
      --data-root .../Single_Tank_Oil_Estimation_train_split \\
      --weights .../OlmoEarth-v1_2-Base \\
      --out-dir runs/tank_geom

    # eval + volume: pass meta
    python scripts/tools/train_hh_tank_geom.py eval \\
      --ckpt runs/tank_geom/best.pt --weights ... --data-root ... \\
      --meta-csv .../tank_meta.csv --save-preds pred_test
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import load_model_from_path


# ---------------------------------------------------------------------------
# Head only: shared neck + dual task decoders (no CBAM)
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
    ) -> None:
        super().__init__()
        self.neck = SharedUpsampleNeck(in_dim, mid_dim, patch_size)
        feat_ch = self.neck.out_dim
        self.mask_decoder = TaskDecoder(feat_ch, out_ch=2, depth=decoder_depth)
        self.kp_decoder = TaskDecoder(feat_ch, out_ch=3, depth=decoder_depth)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.neck(tokens)
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
            emb_dim, patch_size, mid_dim=mid_dim, decoder_depth=decoder_depth
        )

    def train(self, mode: bool = True):  # noqa: A003
        super().train(mode)
        self.backbone.eval()
        return self

    def encode(self, images: torch.Tensor) -> torch.Tensor:
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
        with torch.no_grad():
            out = self.backbone.encoder(
                sample, fast_pass=True, patch_size=self.patch_size
            )
        return out["tokens_and_masks"].sentinel1.mean(dim=(3, 4))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.encode(images))


# ---------------------------------------------------------------------------
# Data (supervised: image + mask + keypoints only)
# ---------------------------------------------------------------------------


def _read_tif(path: Path) -> np.ndarray:
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
    ) -> None:
        self.split = split
        self.image_dir = root / split / "image"
        self.mask_dir = root / split / "mask"
        self.key_dir = root / split / "key"
        self.size = size
        self.sigma = sigma
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

        img_f = _fit_square_simple(img, self.size)
        mask_f = _fit_square_simple(mask, self.size)
        oy, ox = _origin_offset(img.shape, self.size)

        heats = np.zeros((3, self.size, self.size), dtype=np.float32)
        for i, lab in enumerate(("O1", "O2", "O3")):
            if lab not in key["points"]:
                continue
            x, y = key["points"][lab]  # type: ignore[index]
            heats[i] = _gaussian_heatmap(self.size, self.size, x + ox, y + oy, self.sigma)

        if key["circle"] is not None:
            r_mask_px = float(key["circle"][2])  # type: ignore[index]
        else:
            area = float(mask_f.sum())
            r_mask_px = math.sqrt(area / math.pi) if area > 0 else 0.0

        std = float(img_f.std())
        img_f = (img_f - float(img_f.mean())) / (std + 1e-6)
        x = np.stack([img_f, img_f], axis=0)

        return {
            "image": torch.from_numpy(x),
            "mask": torch.from_numpy(mask_f).long(),
            "heatmaps": torch.from_numpy(heats),
            "r_mask_px": torch.tensor(r_mask_px, dtype=torch.float32),
            "name": name,
            "split": self.split,
        }


def load_meta_csv(path: Path | None) -> dict[str, dict[str, float]]:
    """Map filename (and optional split/filename) -> imaging meta.

    CSV columns: filename, pixel_resolution, incidenceangle [, split]
    Prefer ``split/filename`` when ``split`` exists (avoids train/test collisions).
    """
    if path is None or not path.is_file():
        return {}
    df = pd.read_csv(path)
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
) -> dict[str, float]:
    delta = math.radians(float(delta_deg))
    d12 = float(np.linalg.norm(o1 - o2))
    d13 = float(np.linalg.norm(o1 - o3))
    r_m = sr * r_mask_px / max(math.sin(delta), 1e-6)
    h_m = max(sr * (d13 - r_mask_px) / max(math.cos(delta), 1e-6), 0.0)
    H_m = sr * d12 / max(math.cos(delta), 1e-6)
    v_m3 = math.pi * r_m * r_m * h_m
    return {"R_m": r_m, "H_m": H_m, "h_m": h_m, "V_m3": v_m3, "V_bbl": v_m3 / 0.158987}


def heatmaps_to_points(heat: np.ndarray) -> np.ndarray:
    pts = []
    for i in range(heat.shape[0]):
        idx = int(heat[i].reshape(-1).argmax())
        y, x = divmod(idx, heat.shape[-1])
        pts.append([x, y])
    return np.asarray(pts, dtype=np.float32)


def soft_argmax_points(heat_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    b, c, h, w = heat_logits.shape
    flat = heat_logits.view(b, c, -1) / max(temperature, 1e-6)
    prob = F.softmax(flat, dim=-1)
    ys = torch.arange(h, device=heat_logits.device, dtype=heat_logits.dtype)
    xs = torch.arange(w, device=heat_logits.device, dtype=heat_logits.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=0)
    return torch.einsum("bck,dk->bcd", prob, coords)


def fit_circle_radius(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 8:
        area = float(mask.sum())
        return float(math.sqrt(area / math.pi)) if area > 0 else 0.0
    pad = np.pad(mask > 0, 1, mode="constant")
    edge = []
    for y, x in zip(ys, xs, strict=False):
        if pad[y : y + 3, x : x + 3].min() == 0:
            edge.append((x, y))
    if len(edge) < 8:
        area = float(mask.sum())
        return float(math.sqrt(area / math.pi)) if area > 0 else 0.0
    pts = np.asarray(edge, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, c = sol
        return float(math.sqrt(max(c + cx * cx + cy * cy, 0.0)))
    except np.linalg.LinAlgError:
        area = float(mask.sum())
        return float(math.sqrt(area / math.pi)) if area > 0 else 0.0


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)[:, 1]
    tgt = target.float()
    inter = (probs * tgt).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
    return 1.0 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()


def multitask_loss(
    out: dict[str, torch.Tensor],
    mask: torch.Tensor,
    heatmaps: torch.Tensor,
    lambda_kp: float = 1.0,
) -> torch.Tensor:
    """Supervised loss on mask + keypoints only (no volume / no meta)."""
    seg = F.cross_entropy(out["seg_logits"], mask) + dice_loss_with_logits(
        out["seg_logits"], mask
    )
    kp = F.mse_loss(torch.sigmoid(out["kp_heatmaps"]), heatmaps)
    return seg + lambda_kp * kp


@torch.no_grad()
def _discover_emb_dim(backbone: nn.Module, patch_size: int, device: torch.device) -> int:
    m = HHTankGeomModel(backbone, 768, patch_size).to(device)
    return int(m.encode(torch.zeros(1, 2, 64, 64, device=device)).shape[-1])


def build_model(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, dict]:
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
    ).to(device)
    return model, {"emb_dim": emb_dim, "patch_size": args.patch_size}


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, model_meta = build_model(args, device)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    train_ds = TankGeomDataset(Path(args.data_root), "train", args.size)
    val_ds = TankGeomDataset(Path(args.data_root), "test", args.size)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"encoder=OlmoEarth(frozen) emb_dim={model_meta['emb_dim']} "
        f"trainable={n_params:,} train={len(train_ds)} val={len(val_ds)} "
        f"(supervise mask+kp only; volume is post-process)"
    )

    best = -1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            heats = batch["heatmaps"].to(device)
            out = model(images)
            loss = multitask_loss(out, masks, heats, args.lambda_kp)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += float(loss.item()) * images.size(0)
            n += images.size(0)

        metrics = evaluate(model, val_loader, device)
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
        }
        torch.save(ckpt, out_dir / "last.pt")
        score = metrics["miou"] - 0.01 * metrics["kp_px_err"]
        if score > best:
            best = score
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  saved best.pt (score={best:.4f})")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Val metrics for supervised tasks only (no volume)."""
    model.eval()
    total_loss = 0.0
    n = 0
    seg_preds, seg_labs = [], []
    kp_errs: list[float] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        heats = batch["heatmaps"].to(device)
        out = model(images)
        loss = multitask_loss(out, masks, heats)
        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

        pred = out["seg_logits"].argmax(1).cpu().numpy()
        seg_preds.extend(list(pred))
        seg_labs.extend(list(masks.cpu().numpy()))

        pred_xy = soft_argmax_points(out["kp_heatmaps"]).cpu().numpy()
        gt_h = heats.cpu().numpy()
        for i in range(bs):
            pp = pred_xy[i]
            gp = heatmaps_to_points(gt_h[i])
            kp_errs.append(float(np.linalg.norm(pp - gp, axis=1).mean()))

    flat_p = np.concatenate([p.reshape(-1) for p in seg_preds])
    flat_l = np.concatenate([l.reshape(-1) for l in seg_labs])
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
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--lambda-kp", type=float, default=5.0)

    ev = sub.add_parser("eval", help="Eval mask/kp; optional volume with --meta-csv")
    add_shared(ev)
    ev.add_argument("--ckpt", required=True)
    ev.add_argument("--split", default="test", choices=("train", "test"))
    ev.add_argument(
        "--meta-csv",
        default=None,
        help="Imaging meta for volume (Sr, incidence). Needed for R/H/h/V columns",
    )
    ev.add_argument(
        "--pred-radius",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit R_mask from predicted mask; else use GT circle",
    )
    ev.add_argument(
        "--save-preds",
        default=None,
        help="Dir for mask GeoTIFFs; volumes.csv gets V only if --meta-csv is set",
    )
    return p


@torch.no_grad()
def eval_only(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    backbone = load_model_from_path(args.weights).to(device)
    model = HHTankGeomModel(
        backbone,
        emb_dim=int(ckpt["emb_dim"]),
        patch_size=int(ckpt.get("patch_size", args.patch_size)),
        mid_dim=int(ckpt.get("mid_dim", 256)),
        decoder_depth=int(ckpt.get("decoder_depth", 3)),
    ).to(device)
    model.proj.load_state_dict(ckpt["proj"])
    model.head.load_state_dict(ckpt["head"])

    ds = TankGeomDataset(
        Path(args.data_root),
        args.split,
        size=int(ckpt.get("size", args.size)),
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    metrics = evaluate(model, loader, device)
    print(json.dumps(metrics, indent=2))

    if args.save_preds is None:
        return

    meta = load_meta_csv(Path(args.meta_csv) if args.meta_csv else None)
    pred_dir = Path(args.save_preds)
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    skipped_meta = 0
    model.eval()
    for batch in tqdm(loader, desc="save"):
        images = batch["image"].to(device)
        out = model(images)
        masks = out["seg_logits"].argmax(1).cpu().numpy().astype(np.uint8)
        xy = soft_argmax_points(out["kp_heatmaps"]).cpu().numpy()
        r_gt = batch["r_mask_px"].numpy()
        for i, name in enumerate(batch["name"]):
            m = masks[i]
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

            row: dict[str, object] = {
                "filename": name,
                "split": args.split,
                "O1_x": float(xy[i, 0, 0]),
                "O1_y": float(xy[i, 0, 1]),
                "O2_x": float(xy[i, 1, 0]),
                "O2_y": float(xy[i, 1, 1]),
                "O3_x": float(xy[i, 2, 0]),
                "O3_y": float(xy[i, 2, 1]),
            }
            mmeta = lookup_meta(meta, args.split, name) if meta else None
            if mmeta is None:
                skipped_meta += 1
                rows.append(row)
                continue
            r_use = fit_circle_radius(m) if args.pred_radius else float(r_gt[i])
            geom = volume_from_geometry(
                xy[i, 0],
                xy[i, 1],
                xy[i, 2],
                r_use,
                float(mmeta["pixel_resolution"]),
                float(mmeta["incidenceangle"]),
            )
            row["R_mask_px"] = r_use
            row["sr"] = mmeta["pixel_resolution"]
            row["delta_deg"] = mmeta["incidenceangle"]
            row.update(geom)
            rows.append(row)

    pd.DataFrame(rows).to_csv(pred_dir / "volumes.csv", index=False)
    if not meta:
        print(
            f"wrote masks to {pred_dir}; volumes.csv has keypoints only "
            f"(pass --meta-csv to compute R/H/h/V)"
        )
    elif skipped_meta:
        print(
            f"wrote preds to {pred_dir}; {skipped_meta}/{len(rows)} rows "
            f"missing meta (no volume for those)"
        )
    else:
        print(f"wrote preds + full volumes.csv to {pred_dir}")


def main() -> None:
    args = _build_parser().parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        eval_only(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
