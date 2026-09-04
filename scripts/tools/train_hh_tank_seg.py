"""Train / evaluate a tank-vs-background head on HH SAR using frozen OlmoEarth.

Data layout (your dataset)::

    <root>/train/image/tankXXXX.tif
    <root>/train/mask/tankXXXX.tif
    <root>/test/image/...
    <root>/test/mask/...

HH is duplicated to C=2 and fed as ``sentinel1``. Backbone stays frozen; only
the 1x1 channel proj + UNetDecoder are trained.

Examples (AutoDL)::

    python scripts/tools/train_hh_tank_seg.py train \\
      --data-root /path/to/Single_Tank_Oil_Estimation_train_split \\
      --weights /root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base \\
      --out-dir runs/tank_seg

    python scripts/tools/train_hh_tank_seg.py eval \\
      --data-root /path/to/Single_Tank_Oil_Estimation_train_split \\
      --weights /root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base \\
      --ckpt runs/tank_seg/best.pt \\
      --split test
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import load_model_from_path


class UNetDecoder(nn.Module):
    """Upsample ViT patch tokens (B,H',W',D) to per-pixel logits (B,C,H,W).

    Inlined from ``evals.finetune.unet_head`` so this script does not pull the
    full eval dependency stack (rioxarray, geobench, ...).
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        patch_size: int,
        conv_layers_per_resolution: int = 1,
    ) -> None:
        if patch_size < 1 or (patch_size & (patch_size - 1)) != 0:
            raise ValueError(f"patch_size must be a power of two, got {patch_size}")
        super().__init__()
        n_stages = int(math.log2(patch_size))

        def conv() -> list[nn.Module]:
            return [
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]

        layers: list[nn.Module] = conv()
        for _ in range(n_stages):
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            for _ in range(conv_layers_per_resolution):
                layers.extend(conv())
        layers.append(nn.Conv2d(in_dim, num_classes, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.decoder(x)


def _read_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read()
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3:
        return arr[0]
    return arr


def _binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Map arbitrary label values to {0, 1} (background / tank)."""
    m = mask.astype(np.float32)
    if m.max() > 1:
        m = (m > 0).astype(np.float32)
    else:
        m = (m > 0.5).astype(np.float32)
    return m


class TankHHDataset(Dataset):
    """Paired HH image / mask patches, resized to ``size`` x ``size``."""

    def __init__(self, root: Path, split: str, size: int = 128) -> None:
        self.image_dir = root / split / "image"
        self.mask_dir = root / split / "mask"
        self.size = size
        self.paths = sorted(self.image_dir.glob("*.tif"))
        if not self.paths:
            raise FileNotFoundError(f"no tifs under {self.image_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path = self.paths[idx]
        mask_path = self.mask_dir / img_path.name
        if not mask_path.is_file():
            raise FileNotFoundError(f"missing mask for {img_path.name}")

        img = _read_tif(img_path).astype(np.float32)
        mask = _binarize_mask(_read_tif(mask_path))

        # Center-crop / pad to square size (multiple of patch_size).
        img = _fit_square(img, self.size)
        mask = _fit_square(mask, self.size)

        # Per-image z-score (stable for mixed sensors / scales).
        std = float(img.std())
        img = (img - float(img.mean())) / (std + 1e-6)

        # HH -> fake VV/VH, layout BHWTC later: here CHW with C=2.
        x = np.stack([img, img], axis=0)  # 2, H, W
        return {
            "image": torch.from_numpy(x),
            "mask": torch.from_numpy(mask).long(),
            "name": img_path.name,
        }


def _fit_square(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[-2], arr.shape[-1]
    # Pad if smaller.
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h or pad_w:
        arr = np.pad(
            arr,
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
            mode="reflect",
        )
        h, w = arr.shape[-2], arr.shape[-1]
    # Center crop if larger.
    y0 = max(0, (h - size) // 2)
    x0 = max(0, (w - size) // 2)
    return arr[y0 : y0 + size, x0 : x0 + size]


class HHTankSegModel(nn.Module):
    """Frozen OlmoEarth encoder + trainable HH proj + UNet head."""

    def __init__(self, backbone: nn.Module, emb_dim: int, patch_size: int = 4) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.patch_size = patch_size
        # Optional learnable remix of the duplicated HH channels.
        self.proj = nn.Conv2d(2, 2, kernel_size=1, bias=True)
        nn.init.eye_(self.proj.weight[:, :, 0, 0])
        nn.init.zeros_(self.proj.bias)
        self.head = UNetDecoder(
            in_dim=emb_dim, num_classes=2, patch_size=patch_size
        )

    def train(self, mode: bool = True):  # noqa: A003
        super().train(mode)
        # Keep backbone in eval even when the wrapper is training.
        self.backbone.eval()
        return self

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 2, H, W) -> tokens (B, H', W', D)."""
        b, _, h, w = images.shape
        x = self.proj(images)  # B,2,H,W
        # -> B,H,W,T,C
        x = x.permute(0, 2, 3, 1).unsqueeze(3)  # B,H,W,1,2
        sample = MaskedOlmoEarthSample(
            sentinel1=x,
            sentinel1_mask=torch.ones(
                b, h, w, 1, 1, device=x.device, dtype=x.dtype
            )
            * MaskValue.ONLINE_ENCODER.value,
            timestamps=torch.tensor(
                [[[1, 0, 2020]]] * b, device=x.device, dtype=torch.long
            ),
        )
        with torch.no_grad():
            out = self.backbone.encoder(
                sample, fast_pass=True, patch_size=self.patch_size
            )
        tokens = out["tokens_and_masks"].sentinel1  # B,H',W',T,S,D
        return tokens.mean(dim=(3, 4))  # B,H',W',D

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.encode(images)
        return self.head(feat)  # B,2,H,W


@torch.no_grad()
def _discover_emb_dim(backbone: nn.Module, patch_size: int, device: torch.device) -> int:
    dummy = torch.zeros(1, 2, 64, 64, device=device)
    model = HHTankSegModel(backbone, emb_dim=768, patch_size=patch_size).to(device)
    feat = model.encode(dummy)
    return int(feat.shape[-1])


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft Dice on foreground class."""
    probs = F.softmax(logits, dim=1)[:, 1]
    tgt = target.float()
    inter = (probs * tgt).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
    dice = (2 * inter + 1e-6) / (union + 1e-6)
    return 1.0 - dice.mean()


def compute_seg_metrics(
    preds: list[np.ndarray], labels: list[np.ndarray]
) -> dict[str, float]:
    """Aggregate pixel metrics over a list of HxW int masks."""
    pred = np.concatenate([p.reshape(-1) for p in preds])
    lab = np.concatenate([l.reshape(-1) for l in labels])
    assert pred.shape == lab.shape

    acc = float((pred == lab).mean())
    metrics: dict[str, float] = {"pixel_acc": acc}

    ious = []
    dices = []
    for c in (0, 1):
        tp = int(((pred == c) & (lab == c)).sum())
        fp = int(((pred == c) & (lab != c)).sum())
        fn = int(((pred != c) & (lab == c)).sum())
        iou = tp / (tp + fp + fn + 1e-8)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        name = "bg" if c == 0 else "tank"
        metrics[f"iou_{name}"] = float(iou)
        metrics[f"dice_{name}"] = float(dice)
        ious.append(iou)
        dices.append(dice)
    metrics["miou"] = float(np.mean(ious))
    metrics["dice_macro"] = float(np.mean(dices))
    return metrics


@torch.no_grad()
def evaluate(
    model: HHTankSegModel, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    preds, labels = [], []
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, masks) + dice_loss_with_logits(logits, masks)
        total_loss += float(loss.item()) * images.size(0)
        n += images.size(0)
        pred = logits.argmax(dim=1).cpu().numpy()
        preds.extend(list(pred))
        labels.extend(list(masks.cpu().numpy()))
    metrics = compute_seg_metrics(preds, labels)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading backbone from {args.weights}")
    backbone = load_model_from_path(args.weights).to(device)
    emb_dim = _discover_emb_dim(backbone, args.patch_size, device)
    print(f"emb_dim={emb_dim}  device={device}")

    model = HHTankSegModel(backbone, emb_dim=emb_dim, patch_size=args.patch_size).to(
        device
    )
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_ds = TankHHDataset(Path(args.data_root), "train", size=args.size)
    try:
        val_ds = TankHHDataset(Path(args.data_root), "test", size=args.size)
        val_name = "test"
    except FileNotFoundError:
        val_ds = train_ds
        val_name = "train"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    print(f"train={len(train_ds)}  val({val_name})={len(val_ds)}")

    best_miou = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, masks) + dice_loss_with_logits(logits, masks)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += float(loss.item()) * images.size(0)
            n += images.size(0)

        train_loss = running / max(n, 1)
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f}  "
            f"val_loss={metrics['loss']:.4f}  miou={metrics['miou']:.4f}  "
            f"iou_tank={metrics['iou_tank']:.4f}  dice_tank={metrics['dice_tank']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "emb_dim": emb_dim,
            "patch_size": args.patch_size,
            "size": args.size,
            "proj": model.proj.state_dict(),
            "head": model.head.state_dict(),
            "metrics": metrics,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  saved best.pt (miou={best_miou:.4f})")

    with (out_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    print(f"done. best miou={best_miou:.4f}  artifacts in {out_dir}")


def eval_only(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    backbone = load_model_from_path(args.weights).to(device)
    model = HHTankSegModel(
        backbone,
        emb_dim=int(ckpt["emb_dim"]),
        patch_size=int(ckpt.get("patch_size", args.patch_size)),
    ).to(device)
    model.proj.load_state_dict(ckpt["proj"])
    model.head.load_state_dict(ckpt["head"])

    ds = TankHHDataset(
        Path(args.data_root), args.split, size=int(ckpt.get("size", args.size))
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    metrics = evaluate(model, loader, device)
    print(json.dumps(metrics, indent=2))

    if args.save_preds:
        pred_dir = Path(args.save_preds)
        pred_dir.mkdir(parents=True, exist_ok=True)
        model.eval()
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                logits = model(images)
                pred = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
                for name, p in zip(batch["name"], pred, strict=True):
                    out = pred_dir / name
                    with rasterio.open(
                        out,
                        "w",
                        driver="GTiff",
                        height=p.shape[0],
                        width=p.shape[1],
                        count=1,
                        dtype="uint8",
                    ) as dst:
                        dst.write(p, 1)
        print(f"wrote preds to {pred_dir}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_shared(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--data-root", type=str, required=True)
        sp.add_argument("--weights", type=str, required=True)
        sp.add_argument("--size", type=int, default=128)
        sp.add_argument("--patch-size", type=int, default=4)
        sp.add_argument("--batch-size", type=int, default=4)
        sp.add_argument("--workers", type=int, default=2)

    tr = sub.add_parser("train", help="Train proj+head, report val segmentation metrics")
    add_shared(tr)
    tr.add_argument("--out-dir", type=str, required=True)
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--weight-decay", type=float, default=1e-4)

    ev = sub.add_parser("eval", help="Evaluate a checkpoint on a split")
    add_shared(ev)
    ev.add_argument("--ckpt", type=str, required=True)
    ev.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ev.add_argument(
        "--save-preds",
        type=str,
        default=None,
        help="Optional directory to write predicted mask GeoTIFFs",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.cmd == "train":
        train(args)
    else:
        eval_only(args)


if __name__ == "__main__":
    main()
