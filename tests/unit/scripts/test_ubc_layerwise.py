"""Layer-wise AdamW groups for adapter / pixel embed / encoder blocks / head."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

TOOLS = next(
    parent / "scripts" / "tools"
    for parent in Path(__file__).resolve().parents
    if (parent / "scripts" / "tools" / "ubc_layerwise.py").is_file()
)
sys.path.insert(0, str(TOOLS))

from ubc_layerwise import build_layerwise_param_groups  # noqa: E402


class DummyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embeddings = nn.Linear(4, 4)
        self.blocks = nn.ModuleList(nn.Linear(4, 4) for _ in range(3))
        self.norm = nn.LayerNorm(4)


class DummyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = DummyEncoder()
        self.rgb_adapter = nn.Conv2d(3, 12, kernel_size=1)
        self.stem = nn.Conv2d(4, 4, kernel_size=1)


class DummyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = DummyBackbone()
        self.roi_heads = nn.Linear(4, 8)


def _lr_of(groups: list[dict], parameter: nn.Parameter) -> float:
    matches = [
        group["lr"]
        for group in groups
        if any(candidate is parameter for candidate in group["params"])
    ]
    assert len(matches) == 1
    return matches[0]


def test_layerwise_rates_decay_from_top_block() -> None:
    model = DummyDetector()
    groups = build_layerwise_param_groups(
        model,
        head_lr=2e-4,
        backbone_lr=1e-4,
        adapter_lr=2e-4,
        pixel_embed_lr=5e-5,
        gamma=0.5,
        weight_decay=1e-4,
    )
    assigned = {id(parameter) for group in groups for parameter in group["params"]}
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert {id(parameter) for parameter in trainable} == assigned

    encoder = model.backbone.encoder
    assert _lr_of(groups, encoder.blocks[2].weight) == 1e-4
    assert _lr_of(groups, encoder.blocks[1].weight) == 5e-5
    assert _lr_of(groups, encoder.blocks[0].weight) == 2.5e-5
    assert _lr_of(groups, encoder.patch_embeddings.weight) == 5e-5
    assert _lr_of(groups, encoder.norm.weight) == 2.5e-5
    assert _lr_of(groups, model.backbone.rgb_adapter.weight) == 2e-4
    assert _lr_of(groups, model.roi_heads.weight) == 2e-4
    assert _lr_of(groups, model.backbone.stem.weight) == 2e-4
    assert _lr_of(groups, encoder.blocks[2].bias) == 1e-4
    bias_group = next(
        group
        for group in groups
        if any(parameter is encoder.blocks[2].bias for parameter in group["params"])
    )
    assert bias_group["weight_decay"] == 0.0


def test_layerwise_rejects_bad_gamma() -> None:
    model = DummyDetector()
    try:
        build_layerwise_param_groups(
            model,
            head_lr=1e-4,
            backbone_lr=1e-4,
            adapter_lr=1e-4,
            pixel_embed_lr=1e-4,
            gamma=0.0,
            weight_decay=0.0,
        )
    except ValueError as exc:
        assert "layerwise-gamma" in str(exc)
    else:
        raise AssertionError("expected ValueError for gamma=0")
