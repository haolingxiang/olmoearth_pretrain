"""Smoke-test OlmoEarth encoder inference with synthetic inputs.

Weights download automatically from Hugging Face on first run
(allenai/<ModelID>/config.json + weights.pth). On AutoDL / China networks:

    export HF_ENDPOINT=https://hf-mirror.com

Usage:

    python scripts/tools/smoke_test_inference.py
    python scripts/tools/smoke_test_inference.py --model OlmoEarth-v1_2-Nano
    python scripts/tools/smoke_test_inference.py --no-weights   # config only, random init
"""

from __future__ import annotations

import argparse

import torch

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=str,
        default=ModelID.OLMOEARTH_V1_2_NANO.value,
        choices=[m.value for m in ModelID],
        help="HF model id (default: Nano, smallest download).",
    )
    p.add_argument(
        "--no-weights",
        action="store_true",
        help="Skip weights.pth download; random init (still needs config.json).",
    )
    p.add_argument("--h", type=int, default=64, help="Spatial height.")
    p.add_argument("--w", type=int, default=64, help="Spatial width.")
    p.add_argument("--patch-size", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = ModelID(args.model)

    print(f"device={device}")
    print(f"model={model_id.value}  load_weights={not args.no_weights}")
    print("loading (first run downloads from Hugging Face)...")

    model = load_model_from_id(model_id, load_weights=not args.no_weights)
    model.to(device).eval()
    print("model loaded")

    b, h, w, t = 1, args.h, args.w, 1
    timestamps = torch.tensor([[[15, 5, 2020]]], device=device)  # day, month0, year
    enc_mask = MaskValue.ONLINE_ENCODER.value

    # Sentinel-2 L2A: C=12, mask band-sets S=3
    s2 = torch.randn(b, h, w, t, 12, device=device)
    s2_mask = torch.ones(b, h, w, t, 3, device=device) * enc_mask

    # Sentinel-1: C=2 (vv, vh), mask band-sets S=1
    s1 = torch.randn(b, h, w, t, 2, device=device)
    s1_mask = torch.ones(b, h, w, t, 1, device=device) * enc_mask

    cases = [
        (
            "sentinel2_l2a",
            MaskedOlmoEarthSample(
                timestamps=timestamps,
                sentinel2_l2a=s2,
                sentinel2_l2a_mask=s2_mask,
            ),
        ),
        (
            "sentinel1",
            MaskedOlmoEarthSample(
                timestamps=timestamps,
                sentinel1=s1,
                sentinel1_mask=s1_mask,
            ),
        ),
    ]

    with torch.no_grad():
        for name, sample in cases:
            out = model.encoder(sample, fast_pass=True, patch_size=args.patch_size)
            tokens = getattr(out["tokens_and_masks"], name)
            pooled = tokens.mean(dim=[3, 4])
            print(f"[ok] {name}: tokens={tuple(tokens.shape)} pooled={tuple(pooled.shape)}")

    print("smoke test passed")


if __name__ == "__main__":
    main()
