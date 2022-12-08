import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads import FastRCNNOutputLayers, build_box_head
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling import ROI_HEADS_REGISTRY


def _log_classification_stats(pred_logits, gt_classes, prefix="fast_rcnn"):
    """
    Log the classification metrics to EventStorage. This function was grabbed from the Detectron2.modeling. 
    We overwrote this function to create grab the loss per iteration. Refer more to the official documentation.

    Args:
    --------
        pred_logits: Rx(K+1) logits. The last column is for background class.
        gt_classes: R labels
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)
    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)


class CustomFastRCNNOutputLayers(FastRCNNOutputLayers):
    """
    Custom loss function, this inherits the FasterRCNNOutputlayer function and changes 
    the weights in the cross entropy function. Most function is kept the same.
    """
    def __init__(self, cfg, input_shape, weight):
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        in_channels = [input_shape[f].channels for f in in_features]
        
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )

        # grabbing the weights
        self.gpu_weight = weight
        FastRCNNOutputLayers.__init__(
            self, input_shape = box_head.output_shape,
            box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS), 
            num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        )

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        losses = {
            'loss_cls': F.cross_entropy(scores, gt_classes, reduction="mean", weight=self.gpu_weight),#main function change!
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

