import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, unmap, bbox_overlaps)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead

EPS = 1e-12
INF = 1e8

@HEADS.register_module()
class ATSSHead(AnchorHead):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection.

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    https://arxiv.org/abs/1912.02424
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_loc=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_anchor=dict(type='IoULoss', loss_weight=2.0),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.strides = [8, 16, 32, 64, 128]
        self.regress_ranges = regress_ranges
        self.anchor_norm = [32, 64, 128, 256, 512]
        self.center_sampling = False
        self.norm_cfg = norm_cfg
        super(ATSSHead, self).__init__(num_classes, in_channels, **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_anchor = build_loss(loss_anchor)
        self.loss_loc = build_loss(loss_loc)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.anc_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.anc_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.atss_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.anchor_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 2, 3, padding=1)
        self.conv_loc = nn.Conv2d(self.feat_channels, 1, 1)
        self.atss_centerness = nn.Conv2d(
            self.feat_channels, self.num_anchors * 1, 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.anchor_generator.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.anc_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.atss_cls, std=0.01, bias=bias_cls)
        normal_init(self.atss_reg, std=0.01)
        normal_init(self.anchor_reg, std=0.01)
        normal_init(self.conv_loc, std=0.01)
        normal_init(self.atss_centerness, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                centerness (Tensor): Centerness for a single scale level, the
                    channel number is (N, num_anchors * 1, H, W).
        """
        cls_feat = x
        reg_feat = x
        anc_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        for anc_conv in self.anc_convs:
            anc_feat = anc_conv(anc_feat)

        cls_score = self.atss_cls(cls_feat)
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        centerness = self.atss_centerness(reg_feat)
        #print(cls_score.shape,bbox_pred.shape)
        anchor_pred = self.anchor_reg(anc_feat)
        anchor_pred = anchor_pred.exp()
        loc_pred = self.conv_loc(anc_feat)
        return cls_score, bbox_pred, anchor_pred, loc_pred, centerness

    def loss_single(self, anchors, cls_score, bbox_pred, loc_pred, centerness, labels,
                    label_weights, bbox_targets, loc_target, loc_weight,
                loc_average, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loc_pred = loc_pred.permute(0, 2, 3, 1).reshape(-1, 1)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        loc_target = loc_target.type_as(labels)
        loss_loc = self.loss_loc(loc_pred, loc_target, loc_weight, avg_factor=loc_average)
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            centerness_targets = self.centerness_target(
                pos_anchors, pos_bbox_targets)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)
            pos_decode_bbox_targets = self.bbox_coder.decode(
                pos_anchors, pos_bbox_targets)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=centerness_targets,
                avg_factor=1.0)

            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness,
                centerness_targets,
                avg_factor=num_total_samples)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, loss_loc, loss_centerness, centerness_targets.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             anchor_preds,
             loc_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        # shape (N, N_featmap, W * H * scales * ratios)
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        """added"""
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,  # 生成 5层 lvl特征图中心坐标
                                           bbox_preds[0].device)
        anchor_preds_decode = self.get_anchor_preds(all_level_points, anchor_preds, featmap_sizes)
        de_loc_preds = self.get_loc_preds(loc_preds)
        quality_targets, anchor_targets_stride, anchor_weights, loc_targets, loc_targets_lvl = self.get_anchor_targets(all_level_points, gt_bboxes, gt_labels)
        loc_weight = [torch.zeros_like(loc) for loc in loc_targets_lvl]
        loc_average = []

        for i in range(4):
            for j in range(5):
                anchor_preds_decode[i][j][loc_targets[i][j] == 0] = anchor_list[i][j][loc_targets[i][j] == 0]

        for lvl in range(5):   # 取出每张图像的anchor_preds_list 5层 []
            loc_weight[lvl][loc_targets_lvl[lvl] == 0] = 0.1
            loc_weight[lvl][loc_targets_lvl[lvl] == 1] = 1
            loc_average.append(len(loc_weight[lvl])//20)

        anchor_list = anchor_preds_decode

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, loss_loc, loss_centerness,\
            bbox_avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                loc_preds,
                centernesses,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                loc_targets_lvl,
                loc_weight,
                loc_average,
                # anchor_targets_stride,
                num_total_samples=num_total_samples)

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).item()
        if bbox_avg_factor < EPS:
            bbox_avg_factor = 1
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        #######
        flatten_quality_targets = torch.cat(quality_targets)
        flatten_anchor_targets_stride = torch.cat(anchor_targets_stride)
        flatten_weights = torch.cat(anchor_weights)
        num_imgs = cls_scores[0].size(0)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        pos_inds = torch.nonzero(flatten_quality_targets > 0).reshape(-1)
        num_pos = len(pos_inds)
        flatten_anchor_preds = [
            anchor_pred.permute(0, 2, 3, 1).reshape(-1, 2)
            for i, anchor_pred in enumerate(anchor_preds)
        ]
        flatten_anchor_preds = torch.cat(flatten_anchor_preds)
        pos_anchor_preds = flatten_anchor_preds[pos_inds]
        if num_pos > 0:
            pos_weights = flatten_weights[pos_inds]  # 正样本权重
            pos_anchor_targets_stride = flatten_anchor_targets_stride[pos_inds]
            pos_points = flatten_points[pos_inds]  # anchor正样本对应的网格坐标值
            pos_anchor_preds = torch.cat([pos_points - pos_anchor_preds, pos_points + pos_anchor_preds], 1)
            pos_anchor_targets = torch.cat(
                [pos_points - pos_anchor_targets_stride, pos_points + pos_anchor_targets_stride], 1)
            loss_anchor = self.loss_anchor(pos_anchor_preds,  # 计算anchor loss
                                           pos_anchor_targets,
                                           weight=pos_weights,
                                           avg_factor=pos_weights.sum())
        else:
            loss_anchor = pos_anchor_preds.sum()
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_centerness=loss_centerness,
            loss_loc=loss_loc,
            loss_anchor=loss_anchor)



    def centerness_target(self, anchors, bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        gts = self.bbox_coder.decode(anchors, bbox_targets)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        try:
            assert not torch.isnan(centerness).any()
        except:
            centerness[torch.isnan(centerness)] = 0.0
        return centerness

    def centerness_target_iou(self, anchors, bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        # use IOU to compute centerness
        gts = self.bbox_coder.decode(anchors, bbox_targets)
        centerness = bbox_overlaps(anchors,gts,is_aligned=True)
        assert not torch.isnan(centerness).any()
        return centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds','anchor_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   anchor_preds,
                   loc_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_anchors * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      device)
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        # loc_pred = loc_pred.sigmoid().detach()

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            loc_pred_list = [
                loc_preds[i][img_id].sigmoid().detach() for i in range(num_levels)
            ]
            anchor_pred_list = [
                anchor_preds[i][img_id].detach() * 32 * pow(2, i) for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list, anchor_pred_list, loc_pred_list,
                                                mlvl_anchors,
                                                centerness_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale,
                                                with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           anchor_preds,
                           loc_preds,
                           mlvl_anchors,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_anchors * 1, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_anchors = []
        mlvl_centerness = []
        for cls_score, bbox_pred, anchor_pred, loc_pred, centerness, points, anchors in zip(
                cls_scores, bbox_preds,anchor_preds, loc_preds, centernesses, mlvl_points, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchor_pred = anchor_pred.permute(1, 2, 0).reshape(-1, 2)
            loc_pred = loc_pred.permute(1, 2, 0).reshape(-1, 2)
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                anchor_pred = anchor_pred[topk_inds, :]
                loc_pred = loc_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            anchor_preds = torch.stack([
                points[:, 0] - anchor_pred[:, 0],
                points[:, 1] - anchor_pred[:, 1],
                points[:, 0] + anchor_pred[:, 0],
                points[:, 1] + anchor_pred[:, 1],
            ], -1)

            bboxes = self.bbox_coder.decode(
                anchor_preds, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                mlvl_bboxes,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_centerness)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_centerness

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of postive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if hasattr(self, 'bbox_coder'):
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                # used in VFNetHead
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points of a single scale level."""
        h, w = featmap_size
        x_range = torch.arange(w, dtype=dtype, device=device)
        y_range = torch.arange(h, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),  # 拼接並加上偏移
                             dim=-1) + stride // 2
        return points

    def get_points(self, featmap_sizes, dtype, device, flatten=False):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self._get_points_single(featmap_sizes[i], self.strides[i],
                                        dtype, device, flatten))
        return mlvl_points

    def get_anchor_preds(self, all_level_points, anchor_preds, featmap_sizes):
        assert len(all_level_points) == len(anchor_preds)

        num_lvl = len(all_level_points)
        num_imgs = anchor_preds[0].shape[0]
        all_imgs_anchor_preds = []
        for img_id in range(num_imgs):   # 取出每张图像的anchor_preds_list 5层 []
            all_level_anchor_preds = [
                anchor_pred[img_id].permute(1, 2, 0).reshape(-1, 2) * 32 * pow(2, i)   # 将归一化的预测值转换为真正值
                for i, anchor_pred in enumerate(anchor_preds)
            ]
            for i in range(num_lvl):
                all_level_anchor_preds[i] = torch.cat([all_level_points[i] - all_level_anchor_preds[i],
                                                      all_level_anchor_preds[i] + all_level_points[i]], 1)
            all_imgs_anchor_preds.append(all_level_anchor_preds)
        return all_imgs_anchor_preds

    def get_loc_preds(self, loc_preds):
        num_imgs = loc_preds[0].shape[0]
        all_imgs_loc_preds = []
        for img_id in range(num_imgs):
            all_level_loc_preds = [
                loc_pred[img_id].permute(1, 2, 0).reshape(-1, 1) for loc_pred in loc_preds
            ]
            all_imgs_loc_preds.append(all_level_loc_preds)
        return all_imgs_loc_preds


    def get_anchor_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)                      # 5层特征图
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(  # .new_tensor(tensor) 创建一个与给定tensor一致的新tensor
                points[i]) for i in range(num_levels)                      # tensor1.expand_as（tensor）将tensor1扩充复制
        ]                                                                  # 成新tensor2，其size与给定的tensor size一致
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]     # 每层特征图的中心点数量 [15200, 3800, 950, 247, 70]

        # get labels and bbox_targets of each image
        # labels_list, bbox_targets_list, anchor_targets_list, quality_targets_list, weights_list = multi_apply(   # 为每一张图像构造预测目标
        #     self._get_target_single,
        #     gt_bboxes_list,
        #     gt_labels_list,
        #     points=concat_points,
        #     regress_ranges=concat_regress_ranges,
        #     num_points_per_lvl=num_points)
        anchor_targets_list, quality_targets_list, weights_list, loc_targets_list = multi_apply(
            # 为每一张图像构造预测目标
            self._get_anchor_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        # labels_list = [labels.split(num_points, 0) for labels in labels_list]   # 对每张图像的预测，划分出每一层特征图的预测
        # bbox_targets_list = [                                                   # 同上
        #     bbox_targets.split(num_points, 0)
        #     for bbox_targets in bbox_targets_list
        # ]
        anchor_targets_list = [
            anchor_targets.split(num_points, 0)
            for anchor_targets in anchor_targets_list
        ]
        quality_targets_list = [
            quality_targets.split(num_points, 0)
            for quality_targets in quality_targets_list
        ]
        weight_targets_list = [
            weight.split(num_points, 0)
            for weight in weights_list
        ]
        loc_targets_list = [
            loc_targets.split(num_points, 0)
            for loc_targets in loc_targets_list
        ]

        # concat per level image

        concat_lvl_anchor_targets_stride = []
        concat_lvl_quality_targets = []
        concat_lvl_weight = []
        concat_lvl_loc_targets = []
        for i in range(num_levels):   # 5
            # concat_lvl_labels.append(                            # concat batch图像的每层征征图的targets
            #     torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_quality_targets.append(
                torch.cat([quality_targets[i] for quality_targets in quality_targets_list])
            )

            # bbox_targets = torch.cat(
            #     [bbox_targets[i] for bbox_targets in bbox_targets_list])
            anchor_targets = torch.cat(
                [anchor_targets[i] for anchor_targets in anchor_targets_list]
            )
            # concat_lvl_anchor_targets_stride.append(anchor_targets)
            # if self.norm_on_bbox:
            #     bbox_targets = bbox_targets / self.strides[i]

            anchor_targets = anchor_targets / self.anchor_norm[i]
            concat_lvl_anchor_targets_stride.append(anchor_targets)
            # concat_lvl_bbox_targets.append(bbox_targets)
            #concat_lvl_anchor_targets.append(anchor_targets)
            concat_lvl_weight.append(
                torch.cat([weight_targets[i] for weight_targets in weight_targets_list])
            )
            concat_lvl_loc_targets.append(
                torch.cat([loc_target[i] for loc_target in loc_targets_list])
            )
        return concat_lvl_quality_targets, concat_lvl_anchor_targets_stride, concat_lvl_weight, loc_targets_list,concat_lvl_loc_targets
        # return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_quality_targets, concat_lvl_anchor_targets_stride, concat_lvl_weight

    def _get_anchor_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)          # 20267 5层特征图的特征点数
        num_gts = gt_labels.size(0)          # gt框的数量2
        if num_gts == 0:                     # 图像中无待预测目标
            return gt_bboxes.new_zeros((num_points, 2)), \
                   gt_bboxes.new_zeros((num_points, 1)), \
                   gt_bboxes.new_zeros((num_points, 1))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (       # 计算gt_bbox的面积
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)     # areas 20267*6
        areas = areas[None].repeat(num_points, 1)             # repeat()函数可以对张量进行复制,第一个参数表示复制行数，第二个参数表示复制列数
        regress_ranges = regress_ranges[:, None, :].expand(         # 20267*6*2
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)  # 20267*6*4
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)        # 20267*6
        ys = ys[:, None].expand(num_points, num_gts)        # 20267*6

        left = xs - gt_bboxes[..., 0]                       # 计算left
        right = gt_bboxes[..., 2] - xs                      # 计算right
        top = ys - gt_bboxes[..., 1]                        # 计算top
        bottom = gt_bboxes[..., 3] - ys                     # 计算 bottom
        bbox_targets = torch.stack((left, top, right, bottom), -1)    #计算所有点与所有gt框的left、right、top、bottom
        # .......................................................
        ws = torch.stack((left, right), -1)
        hs = torch.stack((top, bottom), -1)
        max_w = ws.max(-1)[0]
        max_h = hs.max(-1)[0]
        anchor_targets = torch.stack((max_w, max_h), -1)

        if self.center_sampling:             # 后续处理该部分
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl): # [15200, 3800, 950, 247, 70]
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox                       # 条件1 中心点在gt框中
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0    # 取出最后一个维度上的最小值 与0进行比较

        # condition2: limit the regression range for each location # 条件2 满足回归范围要求
        max_regress_distance = bbox_targets.max(-1)[0]             # 取出最后一个维度上的最大值
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])       # 与每个点的回归范围做比较
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location, # 存在一个点落在多个gts中的情况，取gt面积最小的那个
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF                      # 将不符合条件1，2的面积赋值INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)                 # 取出面积最小的值与索引

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        #bbox_targets = bbox_targets[range(num_points), min_area_inds]
        # ...........................................................

        gt_bboxes = gt_bboxes[range(num_points), min_area_inds]
        anchor_targets = anchor_targets[range(num_points), min_area_inds]

        # weights = bbox_targets[range(num_points), min_area_inds]
        # weights = self.centerness_target(weights)

        weights = torch.zeros_like(gt_bboxes[:, 0].squeeze())       # 计算
        pos_inds = torch.nonzero(labels != self.num_classes).squeeze()  # 正样本索引
        neg_inds = torch.nonzero(labels == self.num_classes).squeeze()  # 负样本索引
        pos_bbox_targets = bbox_targets[range(num_points), min_area_inds]
        weight_pos = self.centerness_target_(pos_bbox_targets)
        # weights[pos_inds] = weight_pos
        # left_x = points[..., 0] - anchor_targets[..., 0]
        # top_y = points[..., 1] - anchor_targets[..., 1]
        # right_x = points[..., 0] + anchor_targets[..., 0]
        # bottom_y = points[..., 1] + anchor_targets[..., 1]
        # boxes = torch.stack((left_x, top_y, right_x, bottom_y), -1)
        anchor_boxes = torch.cat((points - anchor_targets, points + anchor_targets), 1)
        quality_targets = self.BboxOverlaps(anchor_boxes, gt_bboxes)
        quality_targets[neg_inds] = 0          # 属于背景的特征点 quality 赋值为0
        loc_targets = quality_targets
        loc_targets[pos_inds] = 1
        # bbox_targets = self.bbox_coder.encode(boxes, gt_bboxes)      # GT坐标转化为偏移值
        # bbox_targets = gt_bboxes
        return anchor_targets, quality_targets, weight_pos, loc_targets
        # return labels, gt_bboxes, anchor_targets, quality_targets, weights

    def centerness_target_(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (                                           # 计算中心度
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def BboxOverlaps(self, bboxes1, bboxes2, mode='iou', eps=1e-6):
        assert bboxes1.size(-1) in [0, 4]
        assert bboxes2.size(-1) in [0, 4]
        assert bboxes1.size(-1) == bboxes2.size(-1)

        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (     # 计算面积
                bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
                bboxes2[:, 3] - bboxes2[:, 1])
        lt = torch.max(bboxes1[:, :2],                  # 左上角坐标
                       bboxes2[:, :2])
        rb = torch.min(bboxes1[:,2:],                   # 右下角坐标
                       bboxes2[:,2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]
        union = area1[:] + area2[:] - overlap
        eps = union.new_tensor([eps])
        union = torch.max(union, eps)
        ious = overlap / union
        return ious
