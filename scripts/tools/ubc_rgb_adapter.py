"""Trainable RGB → 12-slot Sentinel-2 adapter for UBC OlmoEarth fine-tuning.

Initializes to the hand-written ``rgb-only`` / ``repeat`` mapping so the first
forward matches v2/v3, then a zero-init 3×3 residual can learn the domain gap.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

S2RgbMode = Literal["rgb-only", "repeat"]
S2_DN_MAX = 10_000.0
RGB_CHANNELS = 3
S2_CHANNELS = 12


class RgbToS2Adapter(nn.Module):
    """Map NCHW RGB in ``[0, 255]`` to normalized 12-band S2 features."""

    def __init__(
        self,
        s2_low: Tensor,
        s2_range: Tensor,
        mode: S2RgbMode = "repeat",
    ) -> None:
        super().__init__()
        if mode not in {"rgb-only", "repeat"}:
            raise ValueError(f"unsupported s2 rgb mode: {mode}")
        self.mode = mode
        low = s2_low.detach().float().reshape(-1)
        span = s2_range.detach().float().reshape(-1)
        if low.numel() != S2_CHANNELS or span.numel() != S2_CHANNELS:
            raise ValueError("expected 12 Sentinel-2 low/range channels")
        self.proj = nn.Conv2d(RGB_CHANNELS, S2_CHANNELS, kernel_size=1)
        self.residual = nn.Sequential(
            nn.Conv2d(S2_CHANNELS, S2_CHANNELS, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, S2_CHANNELS),
            nn.GELU(),
            nn.Conv2d(S2_CHANNELS, S2_CHANNELS, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.reset_to_handcrafted(low, span.clamp_min(1e-6))

    def reset_to_handcrafted(self, low: Tensor, span: Tensor) -> None:
        """Match v3 ``_sample``: BGR×10000/255, then (x-low)/range."""
        scale = S2_DN_MAX / 255.0
        weight = self.proj.weight.detach()
        bias = self.proj.bias.detach()
        weight.zero_()
        bias.zero_()
        # RGB NCHW indices: R=0, G=1, B=2. Handcrafted path uses B,G,R.
        bgr_from_rgb = (2, 1, 0)
        filled = S2_CHANNELS if self.mode == "repeat" else RGB_CHANNELS
        for channel in range(filled):
            source = bgr_from_rgb[channel % RGB_CHANNELS]
            weight[channel, source, 0, 0] = scale / span[channel]
            bias[channel] = -low[channel] / span[channel]
        self.proj.weight.data.copy_(weight)
        self.proj.bias.data.copy_(bias)

    def forward(self, rgb: Tensor) -> Tensor:
        if rgb.ndim != 4 or rgb.shape[1] < RGB_CHANNELS:
            raise ValueError("expected NCHW RGB (optionally with extra channels)")
        mapped = self.proj(rgb[:, :RGB_CHANNELS])
        mapped = mapped + self.residual(mapped)
        return mapped.permute(0, 2, 3, 1).unsqueeze(3).contiguous()
