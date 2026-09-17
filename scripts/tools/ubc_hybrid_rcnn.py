"""Hybrid R-CNN: query detector that replaces Mask R-CNN.

This is Sparse R-CNN + QueryInst masks, not Mask R-CNN with extra heads:

* no RPN, no dense anchors, no Cascade RoI sampling
* N learnable proposal boxes/features
* iterative Dynamic Instance Interactive Head
* dynamic mask head on the same queries (one-to-one)

Interface matches torchvision Mask R-CNN: ``model(images, targets)`` returns a
loss dict in train mode and a list of instance dicts in eval mode.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.roi_heads import (
    maskrcnn_inference,
    maskrcnn_loss,
    paste_masks_in_image,
)
from torchvision.ops import MultiScaleRoIAlign, batched_nms, generalized_box_iou

from ubc_cascade_seesaw import SeesawLoss, _sanitize_boxes

HIDDEN = 256
FEATURE_NAMES = ["0", "1", "2", "3", "4"]


class DynamicConv(nn.Module):
    """Sparse R-CNN instance-interactive 1x1 dynamic convolution."""

    def __init__(self, hidden_dim: int = HIDDEN, dynamic_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dynamic_dim = dynamic_dim
        self.param_layer = nn.Linear(hidden_dim, 2 * hidden_dim * dynamic_dim)
        self.norm1 = nn.LayerNorm(dynamic_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, roi_feat: Tensor, query_feat: Tensor) -> Tensor:
        if roi_feat.numel() == 0:
            return roi_feat.new_zeros((0, self.hidden_dim))
        batch, channels, _, _ = roi_feat.shape
        params = self.param_layer(query_feat)
        weight1 = params[:, : channels * self.dynamic_dim].reshape(
            batch, channels, self.dynamic_dim
        )
        weight2 = params[:, channels * self.dynamic_dim :].reshape(
            batch, self.dynamic_dim, channels
        )
        tokens = roi_feat.flatten(2).transpose(1, 2)
        hidden = F.relu(self.norm1(torch.bmm(tokens, weight1)))
        hidden = F.relu(self.norm2(torch.bmm(hidden, weight2)))
        pooled = hidden.mean(dim=1)
        return F.relu(self.out_norm(self.out(pooled)))


class DynamicMaskHead(nn.Module):
    """QueryInst-style mask head: query-conditioned convs keep spatial layout."""

    def __init__(self, hidden_dim: int = HIDDEN, num_classes: int = 13) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.dynamic = DynamicConv(hidden_dim)
        self.project = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.up = nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, stride=2)
        self.logits = nn.Conv2d(hidden_dim, num_classes, 1)

    def forward(self, mask_roi: Tensor, query_feat: Tensor) -> Tensor:
        if mask_roi.numel() == 0:
            return mask_roi.new_zeros((0, self.logits.out_channels, 28, 28))
        conv = self.conv(mask_roi)
        mixed = conv + self.project(self.dynamic(mask_roi, query_feat)[:, :, None, None])
        return self.logits(F.relu(self.up(mixed)))


def _assignment(cost: Tensor) -> tuple[Tensor, Tensor]:
    """Hungarian if scipy is present, otherwise greedy one-to-one."""
    num_queries, num_gt = cost.shape
    if num_gt == 0 or num_queries == 0:
        empty = cost.new_zeros((0,), dtype=torch.long)
        return empty, empty
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        used = torch.zeros(num_queries, dtype=torch.bool, device=cost.device)
        query_ids: list[int] = []
        gt_ids: list[int] = []
        for gt in cost.min(dim=0).values.argsort().tolist():
            for query in cost[:, gt].argsort().tolist():
                if not used[query]:
                    used[query] = True
                    query_ids.append(query)
                    gt_ids.append(gt)
                    break
        return (
            torch.tensor(query_ids, device=cost.device, dtype=torch.long),
            torch.tensor(gt_ids, device=cost.device, dtype=torch.long),
        )
    rows, cols = linear_sum_assignment(cost.detach().float().cpu().numpy())
    return (
        torch.as_tensor(rows, device=cost.device, dtype=torch.long),
        torch.as_tensor(cols, device=cost.device, dtype=torch.long),
    )


def _cxcywh_to_xyxy(boxes: Tensor, shape: tuple[int, int]) -> Tensor:
    height, width = shape
    cx, cy, bw, bh = boxes.unbind(-1)
    x1 = (cx - 0.5 * bw) * width
    y1 = (cy - 0.5 * bh) * height
    x2 = (cx + 0.5 * bw) * width
    y2 = (cy + 0.5 * bh) * height
    return _sanitize_boxes(torch.stack((x1, y1, x2, y2), dim=-1), shape)


def _stack_images(images: list[Tensor]) -> tuple[Tensor, list[tuple[int, int]]]:
    shapes = [(int(image.shape[-2]), int(image.shape[-1])) for image in images]
    max_h = max(height for height, _ in shapes)
    max_w = max(width for _, width in shapes)
    batch = images[0].new_zeros((len(images), images[0].shape[0], max_h, max_w))
    for index, image in enumerate(images):
        batch[index, :, : image.shape[-2], : image.shape[-1]] = image
    return batch, shapes


class HybridRCNN(nn.Module):
    """Query-based R-CNN that replaces torchvision Mask R-CNN."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        num_queries: int = 300,
        num_stages: int = 6,
        use_seesaw: bool = True,
        class_counts: list[int] | None = None,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 300,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_stages = num_stages
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.box_coder = det_utils.BoxCoder(weights=(2.0, 2.0, 1.0, 1.0))
        self.box_roi_pool = MultiScaleRoIAlign(FEATURE_NAMES, 7, 2)
        self.mask_roi_pool = MultiScaleRoIAlign(FEATURE_NAMES, 14, 2)

        self.query_feat = nn.Embedding(num_queries, HIDDEN)
        self.query_boxes = nn.Embedding(num_queries, 4)
        nn.init.trunc_normal_(self.query_feat.weight, std=0.02)
        self._init_grid_boxes()

        self.dynamic_heads = nn.ModuleList(DynamicConv() for _ in range(num_stages))
        self.cls_heads = nn.ModuleList(
            nn.Linear(HIDDEN, num_classes) for _ in range(num_stages)
        )
        self.reg_heads = nn.ModuleList(nn.Linear(HIDDEN, 4) for _ in range(num_stages))
        self.mask_heads = nn.ModuleList(
            DynamicMaskHead(HIDDEN, num_classes) for _ in range(num_stages)
        )
        for layer in self.cls_heads:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0.0)
        for layer in self.reg_heads:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.seesaw: SeesawLoss | None
        if use_seesaw:
            self.seesaw = SeesawLoss(num_classes)
            if class_counts is not None and len(class_counts) == num_classes:
                self.seesaw.cum_samples.copy_(
                    torch.tensor(class_counts, dtype=torch.float32).clamp_min(1.0)
                )
        else:
            self.seesaw = None

    def _init_grid_boxes(self) -> None:
        grid = max(int(math.ceil(math.sqrt(self.num_queries))), 1)
        ys, xs = torch.meshgrid(
            torch.linspace(0.05, 0.95, grid),
            torch.linspace(0.05, 0.95, grid),
            indexing="ij",
        )
        centers = torch.stack((xs.reshape(-1), ys.reshape(-1)), dim=1)[: self.num_queries]
        size = 0.18 * torch.ones(self.num_queries, 2)
        self.query_boxes.weight.data.copy_(torch.cat((centers, size), dim=1))

    def _init_boxes(self, image_shapes: list[tuple[int, int]]) -> list[Tensor]:
        return [
            _cxcywh_to_xyxy(self.query_boxes.weight, shape) for shape in image_shapes
        ]

    def _refine(
        self,
        deltas: Tensor,
        boxes: list[Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> list[Tensor]:
        decoded = self.box_coder.decode(deltas, boxes)
        if decoded.ndim == 3:
            decoded = decoded[:, 0]
        counts = [box.shape[0] for box in boxes]
        return [
            _sanitize_boxes(part.detach(), shape)
            for part, shape in zip(decoded.split(counts, 0), image_shapes, strict=True)
        ]

    def _stage_losses(
        self,
        logits: Tensor,
        deltas: Tensor,
        mask_logits: Tensor,
        boxes: list[Tensor],
        targets: list[dict[str, Tensor]],
        image_shapes: list[tuple[int, int]],
    ) -> dict[str, Tensor]:
        counts = [box.shape[0] for box in boxes]
        logit_parts = logits.split(counts, 0)
        delta_parts = deltas.split(counts, 0)
        mask_parts = mask_logits.split(counts, 0)
        total_cls = logits.sum() * 0.0
        total_box = deltas.sum() * 0.0
        total_mask = mask_logits.sum() * 0.0
        matched_images = 0
        for logit, delta, mask_logit, proposal, target, shape in zip(
            logit_parts, delta_parts, mask_parts, boxes, targets, image_shapes, strict=True
        ):
            gt_boxes = _sanitize_boxes(target["boxes"].to(proposal.dtype), shape)
            gt_labels = target["labels"]
            labels = torch.zeros(logit.shape[0], dtype=torch.long, device=logit.device)
            query_idx = logit.new_zeros((0,), dtype=torch.long)
            gt_idx = logit.new_zeros((0,), dtype=torch.long)
            if gt_boxes.numel():
                scores = logit.softmax(dim=-1)
                cls_cost = -scores[:, gt_labels]
                giou_cost = 1.0 - generalized_box_iou(proposal, gt_boxes)
                l1_cost = torch.cdist(proposal, gt_boxes, p=1) / max(shape[0] + shape[1], 1)
                query_idx, gt_idx = _assignment(
                    2.0 * cls_cost + 2.0 * giou_cost + 5.0 * l1_cost
                )
                labels[query_idx] = gt_labels[gt_idx]
                matched_images += 1
            if self.seesaw is None:
                total_cls = total_cls + F.cross_entropy(logit, labels)
            else:
                total_cls = total_cls + self.seesaw(logit, labels)
            if query_idx.numel() == 0:
                continue
            regression_targets = self.box_coder.encode(
                [gt_boxes[gt_idx]], [proposal[query_idx]]
            )[0]
            total_box = total_box + F.smooth_l1_loss(
                delta[query_idx], regression_targets, beta=1.0 / 9.0, reduction="mean"
            )
            pos_labels = [labels[query_idx]]
            pos_boxes = [proposal[query_idx]]
            pos_matched = [gt_idx]
            total_mask = total_mask + maskrcnn_loss(
                mask_logit[query_idx],
                pos_boxes,
                [target["masks"]],
                [target["labels"]],
                pos_matched,
            )
        count = max(len(boxes), 1)
        return {
            "loss_classifier": total_cls / count,
            "loss_box_reg": total_box / max(matched_images, 1),
            "loss_mask": total_mask / max(matched_images, 1),
        }

    def _predict(
        self,
        logits: Tensor,
        boxes: list[Tensor],
        mask_logits: Tensor,
        image_shapes: list[tuple[int, int]],
    ) -> list[dict[str, Tensor]]:
        counts = [box.shape[0] for box in boxes]
        outputs: list[dict[str, Tensor]] = []
        for logit, proposal, mask_logit, shape in zip(
            logits.split(counts, 0),
            boxes,
            mask_logits.split(counts, 0),
            image_shapes,
            strict=True,
        ):
            scores = logit.softmax(dim=-1)[:, 1:]
            score, label = scores.max(dim=-1)
            label = label + 1
            keep = score > self.score_thresh
            box = _sanitize_boxes(proposal[keep], shape)
            score = score[keep]
            label = label[keep]
            mask_logit = mask_logit[keep]
            if box.numel():
                keep_nms = batched_nms(box, score, label, self.nms_thresh)[
                    : self.detections_per_img
                ]
                box = box[keep_nms]
                score = score[keep_nms]
                label = label[keep_nms]
                mask_logit = mask_logit[keep_nms]
            if box.numel() == 0:
                outputs.append(
                    {
                        "boxes": box,
                        "labels": label,
                        "scores": score,
                        "masks": box.new_zeros((0, 1, shape[0], shape[1])),
                    }
                )
                continue
            masks = maskrcnn_inference(mask_logit, [label])[0]
            pasted = paste_masks_in_image(masks, box, shape)
            outputs.append(
                {"boxes": box, "labels": label, "scores": score, "masks": pasted}
            )
        return outputs

    def forward(
        self,
        images: list[Tensor],
        targets: list[dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        if self.training and targets is None:
            raise ValueError("HybridRCNN training requires targets")
        batch, image_shapes = _stack_images(images)
        features = self.backbone(batch)
        boxes = self._init_boxes(image_shapes)
        query_feats = [self.query_feat.weight for _ in image_shapes]
        losses = {
            "loss_classifier": batch.sum() * 0.0,
            "loss_box_reg": batch.sum() * 0.0,
            "loss_mask": batch.sum() * 0.0,
        }
        logits = None
        mask_logits = None
        for stage in range(self.num_stages):
            roi = self.box_roi_pool(features, boxes, image_shapes)
            query = torch.cat(query_feats, dim=0)
            interacted = query + self.dynamic_heads[stage](roi, query)
            logits = self.cls_heads[stage](interacted)
            deltas = self.reg_heads[stage](interacted)
            mask_roi = self.mask_roi_pool(features, boxes, image_shapes)
            mask_logits = self.mask_heads[stage](mask_roi, interacted)
            if self.training:
                assert targets is not None
                stage_losses = self._stage_losses(
                    logits, deltas, mask_logits, boxes, targets, image_shapes
                )
                scale = 1.0 / self.num_stages
                for name, value in stage_losses.items():
                    losses[name] = losses[name] + scale * value
            boxes = self._refine(deltas, boxes, image_shapes)
            counts = [box.shape[0] for box in boxes]
            query_feats = list(interacted.split(counts, 0))
        if self.training:
            return losses
        assert logits is not None and mask_logits is not None
        return self._predict(logits, boxes, mask_logits, image_shapes)
