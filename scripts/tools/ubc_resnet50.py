"""ImageNet ResNet-50-FPN backbone for the UBC Hybrid R-CNN ablation."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor, nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNet50Pyramid(nn.Module):
    """ImageNet ResNet-50-FPN plus P6, matching Hybrid's five FPN levels."""

    out_channels = 256

    def __init__(self, unfreeze_backbone: bool = True, pretrained: bool = True) -> None:
        super().__init__()
        self.unfreeze_backbone = unfreeze_backbone
        trainable_layers = 5 if unfreeze_backbone else 0
        self.fpn = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights="DEFAULT" if pretrained else None,
            trainable_layers=trainable_layers,
            returned_layers=[1, 2, 3, 4],
        )
        self.p6 = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=2, padding=1)
        nn.init.kaiming_uniform_(self.p6.weight, a=1)
        nn.init.zeros_(self.p6.bias)
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        )

    @property
    def encoder(self) -> nn.Module:
        # Lets the v4 trainer put body weights on --backbone-lr.
        return self.fpn.body

    def forward(self, images: Tensor) -> OrderedDict[str, Tensor]:
        rgb = images[:, :3] / 255.0
        rgb = (rgb - self.mean.to(dtype=rgb.dtype)) / self.std.to(dtype=rgb.dtype)
        feats = self.fpn(rgb)
        pyramid = OrderedDict()
        pyramid["0"] = feats["0"]
        pyramid["1"] = feats["1"]
        pyramid["2"] = feats["2"]
        pyramid["3"] = feats["3"]
        pyramid["4"] = self.p6(feats["3"])
        return pyramid
