"""Smoke test Hybrid R-CNN as a Mask R-CNN replacement."""

from __future__ import annotations

from collections import OrderedDict
import sys
from pathlib import Path

import torch
from torch import nn

TOOLS = next(
    parent / "scripts" / "tools"
    for parent in Path(__file__).resolve().parents
    if (parent / "scripts" / "tools" / "ubc_hybrid_rcnn.py").is_file()
)
sys.path.insert(0, str(TOOLS))

from ubc_hybrid_rcnn import HybridRCNN  # noqa: E402


class DummyPyramid(nn.Module):
    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        maps = OrderedDict()
        x = images[:, :3]
        for index in range(5):
            if index:
                x = nn.functional.avg_pool2d(x, 2, ceil_mode=True)
            maps[str(index)] = x.mean(1, keepdim=True).expand(-1, 256, -1, -1).contiguous()
        return maps


def test_hybrid_replaces_mask_rcnn_interface() -> None:
    model = HybridRCNN(DummyPyramid(), num_classes=13, num_queries=16, num_stages=2)
    images = [torch.rand(3, 64, 64), torch.rand(3, 64, 64)]
    targets = [
        {
            "boxes": torch.tensor([[8.0, 8.0, 40.0, 40.0], [20.0, 4.0, 50.0, 30.0]]),
            "labels": torch.tensor([3, 7], dtype=torch.int64),
            "masks": torch.zeros(2, 64, 64, dtype=torch.uint8),
        },
        {
            "boxes": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.int64),
            "masks": torch.zeros(0, 64, 64, dtype=torch.uint8),
        },
    ]
    targets[0]["masks"][0, 8:40, 8:40] = 1
    targets[0]["masks"][1, 4:30, 20:50] = 1
    model.train()
    losses = model(images, targets)
    assert set(losses) == {"loss_classifier", "loss_box_reg", "loss_mask"}
    assert "loss_objectness" not in losses
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()

    model.eval()
    with torch.no_grad():
        outputs = model(images)
    assert len(outputs) == 2
    for row in outputs:
        assert {"boxes", "labels", "scores", "masks"} <= set(row)
