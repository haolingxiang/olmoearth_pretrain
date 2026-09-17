"""UBC Hybrid R-CNN ablation: ImageNet ResNet-50-FPN backbone.

Same protocol as ``train_ubc_roof_instance_v4.py`` (512 full image, 300 queries /
6 stages, Seesaw, no copy-paste, full val). Does not modify the OlmoEarth v4
trainer. ``--weights``, ``--patch-size``, ``--s2-rgb-mode``, and ``--detail-skip``
are accepted for command compatibility and ignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import nn

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import train_ubc_roof_instance_v4 as v4  # noqa: E402
from ubc_hybrid_rcnn import HybridRCNN  # noqa: E402
from ubc_resnet50 import ResNet50Pyramid  # noqa: E402


def build_model(
    weights: Path | None,
    task: v4.TaskName,
    tile_size: int,
    patch_size: int,
    s2_rgb_mode: v4.S2RgbMode,
    device: torch.device,
    unfreeze_backbone: bool = False,
    use_detail_skip: bool = True,
    use_seesaw: bool = True,
    class_counts: list[int] | None = None,
    num_queries: int = 300,
    num_query_stages: int = 6,
) -> HybridRCNN:
    del weights, tile_size, patch_size, s2_rgb_mode, use_detail_skip
    print(
        f"loading ImageNet ResNet-50-FPN (unfreeze={unfreeze_backbone}; "
        f"HybridRCNN queries={num_queries}, stages={num_query_stages}, seesaw={use_seesaw})"
    )
    if task == "multimodal":
        print("ResNet-50 uses RGB only; extra SAR channel is ignored")
    backbone = ResNet50Pyramid(unfreeze_backbone=unfreeze_backbone)
    return HybridRCNN(
        backbone,
        num_classes=v4.NUM_CLASSES,
        num_queries=num_queries,
        num_stages=num_query_stages,
        use_seesaw=use_seesaw,
        class_counts=class_counts,
    ).to(device)


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _relax_weights(parser: argparse.ArgumentParser) -> None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                _relax_weights(sub)
            continue
        if "--weights" in getattr(action, "option_strings", ()):
            action.required = False
            action.default = None
            action.help = (
                "ignored; ImageNet ResNet-50 is loaded from torchvision. "
                "Kept so v4 command lines still parse."
            )


def build_parser() -> argparse.ArgumentParser:
    parser = v4.build_parser()
    parser.description = __doc__
    _relax_weights(parser)
    return parser


def _install_v4_hooks() -> None:
    v4.build_model = build_model
    v4.trainable_state_dict = trainable_state_dict


def main() -> None:
    _install_v4_hooks()
    args = build_parser().parse_args()
    if getattr(args, "weights", None) in (None, ""):
        args.weights = "resnet50"
    args.backbone = "resnet50"
    v4.validate_args(args)
    if args.command == "train":
        v4.train(args)
    elif args.command == "eval":
        v4.evaluate_command(args)
    elif args.command == "eval-json":
        v4.evaluate_json_command(args)
    else:
        v4.predict_command(args)


if __name__ == "__main__":
    main()
