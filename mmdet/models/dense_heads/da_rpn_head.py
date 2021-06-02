import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.ops import nms

from ..builder import HEADS
from .guided_anchor_head import GuidedAnchorHead
from .DA_RPN import DynamicAnchorHead
from .rpn_test_mixin import RPNTestMixin


@HEADS.register_module()
class DARPNHead(RPNTestMixin, DynamicAnchorHead):
    """Guided-Anchor-based RPN head."""

    def __init__(self, in_channels, **kwargs):
        super(DARPNHead, self).__init__(1, in_channels, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        super(DARPNHead, self)._init_layers()

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.rpn_conv, std=0.01)
        super(DARPNHead, self).init_weights()

    def forward_single(self, x, scale):
        """Forward feature of a single scale level."""

        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)

        (cls_score, bbox_pred, shape_pred) = super(DARPNHead, self).forward_single(x, scale)

        return cls_score, bbox_pred, shape_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             anchor_preds,
             # quality_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        losses = super(DARPNHead, self).loss(
            cls_scores,
            bbox_preds,
            anchor_preds,
            # quality_preds,
            gt_bboxes,
            None,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'],
            loss_rpn_bbox=losses['loss_bbox'],
            loss_anchor_shape=losses['loss_anchor'])
            # loss_quality=losses["loss_quality"])

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           anchor_preds,
                           # quality_preds,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            anchor_pred = anchor_preds[idx]
            # quality_pred = quality_preds[idx]
            points = mlvl_points[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            # if no location is kept, end.


            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = rpn_cls_score.softmax(dim=1)[:, :-1]
            # filter scores, bbox_pred w.r.t. mask.
            # anchors are filtered in get_anchors() beforehand.
            # scores = scores[mask]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1,
                                                                   4)
            anchor_pred = anchor_pred.permute(1, 2, 0).reshape(-1, 2)

            # quality_pred = quality_pred.permute(1, 2, 0).reshape(-1).sigmoid()
            if scores.dim() == 0:
                rpn_bbox_pred = rpn_bbox_pred.unsqueeze(0)
                # anchors = anchors.unsqueeze(0)
                # scores = scores.unsqueeze(0)
            # filter anchors, bbox_pred, scores w.r.t. scores
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # scores, _ = (scores * quality_pred[:, None]).max(dim=1)
                _, topk_inds = scores.topk(cfg.nms_pre)
                # _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchor_pred = anchor_pred[topk_inds, :]
                points = points[topk_inds, :]
                scores = scores[topk_inds]
            # get proposals w.r.t. anchors and rpn_bbox_pred
            # anchors = torch.stack([
            #     points[:, 0] - anchor_pred[:, 0],
            #     points[:, 1] - anchor_pred[:, 1],
            #     points[:, 0] + anchor_pred[:, 0],
            #     points[:, 1] + anchor_pred[:, 1],
            # ], -1)
            anchors = torch.cat([points-anchor_pred, points+anchor_pred], dim=-1)
            proposals = self.bbox_coder.decode(
                anchors, rpn_bbox_pred, max_shape=img_shape)
            # filter out too small bboxes
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0]
                h = proposals[:, 3] - proposals[:, 1]
                valid_inds = torch.nonzero(
                    (w >= cfg.min_bbox_size) & (h >= cfg.min_bbox_size),
                    as_tuple=False).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            # NMS in current level
            proposals, _ = nms(proposals, scores, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            # NMS across multi levels
            proposals, _ = nms(proposals[:, :4], proposals[:, -1], cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals
