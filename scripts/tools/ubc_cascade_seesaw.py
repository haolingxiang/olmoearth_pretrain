"""Cascade Mask R-CNN ROI heads + Seesaw classification loss for UBC v3.

Designed to replace ``torchvision`` ``MaskRCNN.roi_heads`` while keeping the same
backbone / RPN / mask head interface used by the training script.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models.detection.roi_heads import (
    RoIHeads,
    maskrcnn_inference,
    maskrcnn_loss,
)
from torchvision.ops import boxes as box_ops


class SeesawLoss(nn.Module):
    """Seesaw Cross-Entropy for long-tailed classification (CVPR 2021).

    Background class (label 0) is kept; mitigation / compensation act on the
    full logit vector including background, matching common Cascade+Seesaw
    practice for instance segmentation.
    """

    def __init__(
        self,
        num_classes: int,
        p: float = 0.8,
        q: float = 2.0,
        eps: float = 1e-2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.q = q
        self.eps = eps
        self.register_buffer("cum_samples", torch.zeros(num_classes, dtype=torch.float32))

    def forward(self, cls_score: Tensor, labels: Tensor) -> Tensor:
        if cls_score.numel() == 0:
            return cls_score.sum() * 0.0
        labels = labels.long()
        for class_id in labels.unique():
            idx = int(class_id.item())
            if 0 <= idx < self.num_classes:
                self.cum_samples[idx] += float((labels == class_id).sum().item())

        sample_ratio = self.cum_samples[:, None] / self.cum_samples.clamp_min(1.0)[None, :]
        # Mitigation: down-weight gradients of frequent negative classes.
        mitigation = sample_ratio.pow(self.p)
        mitigation = torch.where(sample_ratio < 1.0, mitigation, torch.ones_like(mitigation))
        # Compensation: boost rare true classes when their score is low.
        score = cls_score.softmax(dim=-1).detach()
        true_score = score[torch.arange(labels.numel(), device=labels.device), labels]
        compensation = (true_score.clamp_min(self.eps).unsqueeze(1) / score.clamp_min(self.eps)).pow(
            self.q
        )
        # sample_ratio[true, :] < 1 → true class rarer than column class
        compensation = torch.where(
            sample_ratio[labels] < 1.0,
            compensation,
            torch.ones_like(compensation),
        )
        weights = mitigation[labels] * compensation
        one_hot = F.one_hot(labels, self.num_classes).float()
        # Leave the true-class channel unweighted (standard Seesaw CE form).
        weights = weights * (1.0 - one_hot) + one_hot
        log_prob = F.log_softmax(cls_score, dim=-1)
        loss = -(one_hot * log_prob * weights).sum(dim=-1)
        return loss.mean()


def cascade_fastrcnn_loss(
    class_logits: Tensor,
    box_regression: Tensor,
    labels: list[Tensor],
    regression_targets: list[Tensor],
    seesaw: SeesawLoss | None,
) -> tuple[Tensor, Tensor]:
    labels_cat = torch.cat(labels, dim=0)
    regression_targets_cat = torch.cat(regression_targets, dim=0)
    if seesaw is None:
        classification_loss = F.cross_entropy(class_logits, labels_cat)
    else:
        classification_loss = seesaw(class_logits, labels_cat)

    sampled_pos_inds_subset = torch.where(labels_cat > 0)[0]
    labels_pos = labels_cat[sampled_pos_inds_subset]
    n_boxes = class_logits.shape[0]
    box_regression = box_regression.reshape(n_boxes, box_regression.size(-1) // 4, 4)
    if sampled_pos_inds_subset.numel() == 0:
        box_loss = box_regression.sum() * 0.0
    else:
        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets_cat[sampled_pos_inds_subset],
            beta=1.0 / 9.0,
            reduction="sum",
        )
        box_loss = box_loss / max(labels_cat.numel(), 1)
    return classification_loss, box_loss


class CascadeRoIHeads(RoIHeads):
    """3-stage Cascade R-CNN heads with optional Seesaw classification loss."""

    def __init__(
        self,
        base: RoIHeads,
        stage_ious: tuple[float, ...] = (0.5, 0.6, 0.7),
        use_seesaw: bool = True,
        seesaw_p: float = 0.8,
        seesaw_q: float = 2.0,
        class_counts: list[int] | None = None,
    ) -> None:
        # Copy the constructed Mask R-CNN heads, then expand box stages.
        super().__init__(
            box_roi_pool=base.box_roi_pool,
            box_head=base.box_head,
            box_predictor=base.box_predictor,
            fg_iou_thresh=stage_ious[0],
            bg_iou_thresh=stage_ious[0],
            batch_size_per_image=base.fg_bg_sampler.batch_size_per_image,
            positive_fraction=base.fg_bg_sampler.positive_fraction,
            bbox_reg_weights=None,
            score_thresh=base.score_thresh,
            nms_thresh=base.nms_thresh,
            detections_per_img=base.detections_per_img,
            mask_roi_pool=base.mask_roi_pool,
            mask_head=base.mask_head,
            mask_predictor=base.mask_predictor,
        )
        self.box_coder = base.box_coder
        self.stage_ious = tuple(stage_ious)
        self.num_stages = len(stage_ious)
        from torchvision.models.detection import _utils as det_utils

        self.stage_matchers = [
            det_utils.Matcher(iou, iou, allow_low_quality_matches=False)
            for iou in stage_ious
        ]

        self.box_heads = nn.ModuleList(
            [base.box_head] + [copy.deepcopy(base.box_head) for _ in range(self.num_stages - 1)]
        )
        self.box_predictors = nn.ModuleList(
            [base.box_predictor]
            + [copy.deepcopy(base.box_predictor) for _ in range(self.num_stages - 1)]
        )
        # Keep attribute aliases for torchvision code paths that still touch them.
        self.box_head = self.box_heads[0]
        self.box_predictor = self.box_predictors[0]

        num_classes = base.box_predictor.cls_score.out_features
        self.use_seesaw = use_seesaw
        self.seesaw: SeesawLoss | None
        if use_seesaw:
            self.seesaw = SeesawLoss(num_classes, p=seesaw_p, q=seesaw_q)
            if class_counts is not None:
                counts = torch.tensor(class_counts, dtype=torch.float32)
                if counts.numel() == num_classes:
                    self.seesaw.cum_samples.copy_(counts.clamp_min(1.0))
        else:
            self.seesaw = None

    def _select_with_matcher(
        self,
        proposals: list[Tensor],
        targets: list[dict[str, Tensor]],
        matcher: Any,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        self.check_targets(targets)
        dtype = proposals[0].dtype
        device = proposals[0].device
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        matched_idxs: list[Tensor] = []
        labels: list[Tensor] = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(
            proposals, gt_boxes, gt_labels, strict=True
        ):
            if gt_boxes_in_image.numel() == 0:
                clamped = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                match_quality = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched = matcher(match_quality)
                clamped = matched.clamp(min=0)
                labels_in_image = gt_labels_in_image[clamped].to(dtype=torch.int64)
                labels_in_image[matched == matcher.BELOW_LOW_THRESHOLD] = 0
                labels_in_image[matched == matcher.BETWEEN_THRESHOLDS] = -1
            matched_idxs.append(clamped)
            labels.append(labels_in_image)

        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        for img_id, sample in enumerate(sampled_inds):
            proposals[img_id] = proposals[img_id][sample]
            labels[img_id] = labels[img_id][sample]
            matched_idxs[img_id] = matched_idxs[img_id][sample]
            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def _refine_boxes_train(
        self,
        box_regression: Tensor,
        proposals: list[Tensor],
        labels: list[Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> list[Tensor]:
        boxes_per_image = [p.shape[0] for p in proposals]
        # torchvision BoxCoder.decode returns Tensor[N, num_classes * 4] then
        # reshapes; check actual API — decode(rel_codes, boxes) -> Tensor[N, 4]
        # when box_regression is [N, num_classes * 4], output is [N, num_classes, 4]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        if pred_boxes.ndim == 2:
            # unexpected; split directly
            parts = pred_boxes.split(boxes_per_image, 0)
            return [
                box_ops.clip_boxes_to_image(part.detach(), shape)
                for part, shape in zip(parts, image_shapes, strict=True)
            ]
        parts = pred_boxes.split(boxes_per_image, 0)
        refined = []
        for part, label, shape, proposal in zip(
            parts, labels, image_shapes, proposals, strict=True
        ):
            # part: [N, C, 4]
            idx = torch.arange(part.shape[0], device=part.device)
            class_ids = label.clamp(min=0)
            boxes = part[idx, class_ids]
            boxes = boxes.clone()
            boxes[label <= 0] = proposal[label <= 0]
            refined.append(box_ops.clip_boxes_to_image(boxes.detach(), shape))
        return refined

    def forward(
        self,
        features,  # type: ignore[no-untyped-def]
        proposals,
        image_shapes,
        targets=None,
    ):
        if targets is not None:
            for target in targets:
                if target["boxes"].dtype not in (torch.float, torch.double, torch.half):
                    raise TypeError("target boxes must be float")
                if target["labels"].dtype != torch.int64:
                    raise TypeError("target labels must be int64")

        losses: dict[str, Tensor] = {}
        result: list[dict[str, Tensor]] = []
        stage_proposals = proposals
        matched_idxs = None
        labels = None

        if self.training:
            assert targets is not None
            total_cls = features[list(features.keys())[0]].sum() * 0.0
            total_box = features[list(features.keys())[0]].sum() * 0.0
            for stage, matcher in enumerate(self.stage_matchers):
                stage_proposals, matched_idxs, labels, regression_targets = (
                    self._select_with_matcher(stage_proposals, targets, matcher)
                )
                box_features = self.box_roi_pool(features, stage_proposals, image_shapes)
                box_features = self.box_heads[stage](box_features)
                class_logits, box_regression = self.box_predictors[stage](box_features)
                loss_cls, loss_box = cascade_fastrcnn_loss(
                    class_logits,
                    box_regression,
                    labels,
                    regression_targets,
                    self.seesaw,
                )
                total_cls = total_cls + loss_cls
                total_box = total_box + loss_box
                if stage < self.num_stages - 1:
                    stage_proposals = self._refine_boxes_train(
                        box_regression, stage_proposals, labels, image_shapes
                    )
            losses = {
                "loss_classifier": total_cls / self.num_stages,
                "loss_box_reg": total_box / self.num_stages,
            }
            proposals = stage_proposals
        else:
            for stage in range(self.num_stages):
                box_features = self.box_roi_pool(features, stage_proposals, image_shapes)
                box_features = self.box_heads[stage](box_features)
                class_logits, box_regression = self.box_predictors[stage](box_features)
                if stage < self.num_stages - 1:
                    # Intermediate cascade: decode with predicted class.
                    boxes_per_image = [p.shape[0] for p in stage_proposals]
                    scores = F.softmax(class_logits, -1)
                    pred_labels = scores[:, 1:].argmax(dim=1) + 1
                    label_lists = list(pred_labels.split(boxes_per_image, 0))
                    stage_proposals = self._refine_boxes_train(
                        box_regression, stage_proposals, label_lists, image_shapes
                    )
                else:
                    boxes, scores, labels_out = self.postprocess_detections(
                        class_logits, box_regression, stage_proposals, image_shapes
                    )
                    for box, score, label in zip(boxes, scores, labels_out, strict=True):
                        result.append({"boxes": box, "labels": label, "scores": score})

        if self.has_mask():
            if self.training:
                assert labels is not None and matched_idxs is not None and targets is not None
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(len(proposals)):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                losses["loss_mask"] = maskrcnn_loss(
                    mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs
                )
            else:
                mask_proposals = [row["boxes"] for row in result]
                if self.mask_roi_pool is not None:
                    mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                    mask_features = self.mask_head(mask_features)
                    mask_logits = self.mask_predictor(mask_features)
                    labels_out = [row["labels"] for row in result]
                    masks_probs = maskrcnn_inference(mask_logits, labels_out)
                    for mask_prob, row in zip(masks_probs, result, strict=True):
                        row["masks"] = mask_prob
        return result, losses


def collect_class_counts(index: Any, num_classes: int) -> list[int]:
    """Return [bg_count_placeholder, c1, c2, ...] length num_classes."""
    counts = [0] * num_classes
    # Background gets a large prior so Seesaw does not treat it as rare.
    counts[0] = 10**6
    for annotations in index.annotations_by_image.values():
        for annotation in annotations:
            category_id = int(annotation["category_id"])
            if 0 < category_id < num_classes:
                counts[category_id] += 1
    return counts


def attach_cascade_seesaw(
    model: nn.Module,
    class_counts: list[int] | None = None,
    stage_ious: tuple[float, ...] = (0.5, 0.6, 0.7),
    use_seesaw: bool = True,
) -> nn.Module:
    model.roi_heads = CascadeRoIHeads(
        model.roi_heads,
        stage_ious=stage_ious,
        use_seesaw=use_seesaw,
        class_counts=class_counts,
    )
    return model
