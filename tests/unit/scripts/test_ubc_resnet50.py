"""Smoke test ImageNet ResNet-50 Hybrid trainer without downloading weights."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

TOOLS = next(
    parent / "scripts" / "tools"
    for parent in Path(__file__).resolve().parents
    if (parent / "scripts" / "tools" / "train_ubc_roof_instance_resnet50.py").is_file()
)
sys.path.insert(0, str(TOOLS))

from ubc_resnet50 import ResNet50Pyramid  # noqa: E402
from ubc_hybrid_rcnn import HybridRCNN  # noqa: E402


def test_resnet50_pyramid_five_levels() -> None:
    backbone = ResNet50Pyramid(unfreeze_backbone=True, pretrained=False)
    images = torch.rand(2, 3, 64, 64) * 255.0
    feats = backbone(images)
    assert list(feats) == ["0", "1", "2", "3", "4"]
    assert feats["0"].shape[-2:] == (16, 16)
    assert feats["4"].shape[-2:] == (1, 1)
    for value in feats.values():
        assert value.shape[1] == 256


def test_resnet50_hybrid_train_eval_interface() -> None:
    backbone = ResNet50Pyramid(unfreeze_backbone=False, pretrained=False)
    model = HybridRCNN(backbone, num_classes=13, num_queries=16, num_stages=2)
    images = [torch.rand(3, 64, 64) * 255.0]
    targets = [
        {
            "boxes": torch.tensor([[8.0, 8.0, 40.0, 40.0]]),
            "labels": torch.tensor([3], dtype=torch.int64),
            "masks": torch.zeros(1, 64, 64, dtype=torch.uint8),
        }
    ]
    targets[0]["masks"][0, 8:40, 8:40] = 1
    model.train()
    losses = model(images, targets)
    assert set(losses) == {"loss_classifier", "loss_box_reg", "loss_mask"}
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    body_ids = {id(parameter) for parameter in backbone.encoder.parameters()}
    assert any(id(parameter) in body_ids for parameter in backbone.parameters())

    model.eval()
    with torch.no_grad():
        outputs = model(images)
    assert len(outputs) == 1
    assert {"boxes", "labels", "scores", "masks"} <= set(outputs[0])
