"""Layer-wise AdamW groups for OlmoEarth + UBC detection heads."""

from __future__ import annotations

from typing import Any

from torch import nn


def _is_no_decay(name: str, parameter: nn.Parameter) -> bool:
    return parameter.ndim <= 1 or name.endswith("bias") or "norm" in name.lower()


def _append(
    buckets: dict[float, dict[str, list[nn.Parameter]]],
    learning_rate: float,
    name: str,
    parameter: nn.Parameter,
) -> None:
    slot = buckets.setdefault(learning_rate, {"decay": [], "no_decay": []})
    key = "no_decay" if _is_no_decay(name, parameter) else "decay"
    slot[key].append(parameter)


def build_layerwise_param_groups(
    model: nn.Module,
    *,
    head_lr: float,
    backbone_lr: float,
    adapter_lr: float,
    pixel_embed_lr: float,
    gamma: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Split params into head / adapter / pixel-embed / per-block encoder groups.

    Transformer block ``l`` (0 = bottom) uses ``backbone_lr * gamma ** (L-1-l)``.
    """
    if gamma <= 0 or gamma > 1:
        raise ValueError("--layerwise-gamma must be in (0, 1]")
    backbone = model.backbone
    encoder = backbone.encoder
    blocks = list(encoder.blocks)
    adapter = getattr(backbone, "rgb_adapter", None)

    assigned: set[int] = set()
    buckets: dict[float, dict[str, list[nn.Parameter]]] = {}

    if adapter is not None:
        for name, parameter in adapter.named_parameters():
            if not parameter.requires_grad:
                continue
            _append(buckets, adapter_lr, name, parameter)
            assigned.add(id(parameter))

    for name, parameter in encoder.patch_embeddings.named_parameters():
        if not parameter.requires_grad:
            continue
        _append(buckets, pixel_embed_lr, name, parameter)
        assigned.add(id(parameter))

    depth = max(len(blocks), 1)
    for index, block in enumerate(blocks):
        learning_rate = backbone_lr * (gamma ** (depth - 1 - index))
        for name, parameter in block.named_parameters():
            if not parameter.requires_grad:
                continue
            _append(buckets, learning_rate, name, parameter)
            assigned.add(id(parameter))

    lowest_encoder_lr = backbone_lr * (gamma ** (depth - 1))
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in assigned:
            continue
        learning_rate = (
            lowest_encoder_lr if name.startswith("backbone.encoder.") else head_lr
        )
        _append(buckets, learning_rate, name, parameter)
        assigned.add(id(parameter))

    groups: list[dict[str, Any]] = []
    for learning_rate, split in sorted(buckets.items(), key=lambda item: -item[0]):
        if split["decay"]:
            groups.append(
                {"params": split["decay"], "lr": learning_rate, "weight_decay": weight_decay}
            )
        if split["no_decay"]:
            groups.append(
                {"params": split["no_decay"], "lr": learning_rate, "weight_decay": 0.0}
            )
    if not groups:
        raise RuntimeError("no trainable parameters for layer-wise AdamW")
    return groups
