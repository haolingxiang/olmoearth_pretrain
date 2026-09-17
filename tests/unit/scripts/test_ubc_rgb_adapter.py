"""RGB→S2 adapter matches the handcrafted UBC mapping at init."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TOOLS = next(
    parent / "scripts" / "tools"
    for parent in Path(__file__).resolve().parents
    if (parent / "scripts" / "tools" / "ubc_rgb_adapter.py").is_file()
)
sys.path.insert(0, str(TOOLS))

from ubc_rgb_adapter import S2_DN_MAX, RgbToS2Adapter  # noqa: E402


def test_adapter_init_is_handcrafted_bgr_map() -> None:
    low = torch.linspace(-1.0, 2.0, 12)
    span = torch.linspace(0.5, 3.0, 12)
    scale = S2_DN_MAX / 255.0
    for mode, filled in (("repeat", 12), ("rgb-only", 3)):
        adapter = RgbToS2Adapter(low, span, mode=mode)
        assert torch.count_nonzero(adapter.residual[-1].weight) == 0
        assert torch.count_nonzero(adapter.residual[-1].bias) == 0
        weight = adapter.proj.weight.detach()
        bias = adapter.proj.bias.detach()
        assert torch.count_nonzero(weight) == filled
        for channel in range(12):
            source = (2, 1, 0)[channel % 3]
            if channel < filled:
                expected = torch.zeros(3)
                expected[source] = scale / span[channel]
                torch.testing.assert_close(weight[channel, :, 0, 0], expected)
                torch.testing.assert_close(bias[channel], -low[channel] / span[channel])
            else:
                assert torch.count_nonzero(weight[channel]) == 0
                assert bias[channel].item() == 0.0


def test_adapter_forward_is_zero_residual_at_init() -> None:
    adapter = RgbToS2Adapter(torch.zeros(12), torch.ones(12), mode="repeat")
    rgb = torch.rand(2, 4, 8, 8) * 255.0
    mapped = adapter(rgb)
    direct = F.conv2d(rgb[:, :3], adapter.proj.weight, adapter.proj.bias)
    expected = direct.permute(0, 2, 3, 1).unsqueeze(3)
    assert mapped.shape == (2, 8, 8, 1, 12)
    torch.testing.assert_close(mapped, expected)


def test_adapter_residual_can_train() -> None:
    adapter = RgbToS2Adapter(torch.zeros(12), torch.ones(12), mode="repeat")
    loss = adapter(torch.rand(1, 3, 4, 4) * 255.0).sum()
    loss.backward()
    assert adapter.proj.weight.grad is not None
    assert adapter.residual[-1].weight.grad is not None
    assert torch.isfinite(adapter.proj.weight.grad).all()
