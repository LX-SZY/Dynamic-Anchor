import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale, normal_init
from mmcv.runner import force_fp32
import math
from mmdet.core import distance2bbox, multi_apply, multiclass_nms
from ..builder import HEADS, build_loss
from .Dynamic_Free_Head import PseudoAnchorHead
from mmdet.core import build_bbox_coder


INF = 1e8

@HEADS.register_module()
class Dynamic_Anchor_Head(PseudoAnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=True,
                 center_sample_radius=2,
                 quality_on_reg=True,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_quality=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_anchor=dict(type='IoULoss', loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=(.0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0)),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling          # tricks
        self.center_sample_radius = center_sample_radius
        self.quality_on_reg = quality_on_reg      # tricks
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            **kwargs)
        self.loss_quality = build_loss(loss_quality)
        self.loss_anchor = build_loss(loss_anchor)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.anchor_norm = [64, 128, 256, 512, 1024]

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_quality = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
    def init_weights(self):
        """Initialize weights of the head."""
        super().init_weights()
        normal_init(self.conv_quality, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                    quality
        """
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """

        cls_score, bbox_pred, anchor_pred, cls_feat, reg_feat = super().forward_single(x)
        # ...................................................................

        if self.quality_on_reg:
            quality = self.conv_quality(reg_feat)
        else:
            quality = self.conv_quality(cls_feat)


        anchor_pred = scale(anchor_pred).float()
        anchor_pred = anchor_pred.exp()
        return cls_score, bbox_pred, anchor_pred, quality

    @force_fp32(apply_to=('cls_scores', 'bbox_preds','anchor_preds', 'quality_preds'))
    def loss(self,                                     # 计算损失
             cls_scores,
             bbox_preds,
             anchor_preds,
             quality_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            anchor_preds (list[Tensor]):
            quality_preds (list[Tensor]):
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(anchor_preds)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]   # 特征图尺寸：[100,152][50,76][25,38][13,19][7,10]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,  # 生成 5层 lvl特征图中心坐标
                                           bbox_preds[0].device)
        inside_flag_list = self.get_inside_flag(featmap_sizes, img_metas,  # 计算每张图的inside_flag
                                           cls_scores[0].device)
        # 计算 anchor targets以及weights  anchor_targets已归一化
        quality_targets, anchor_targets_stride, anchor_weights = self.get_targets(all_level_points,
                                                                                  gt_bboxes,
                                                                                  gt_labels)

        # get decoded anchor preds [x0,y0,x1,y1]
        anchor_preds_decode = self.get_anchor_preds(all_level_points, anchor_preds)

        # get decoded anchor  targets
        label_channels = self.cls_out_channels
        cls_reg_targets, anchor_preds_lvl = self.get_clc_reg_targets(anchor_preds_decode,
                                                                     inside_flag_list,
                                                                     gt_bboxes,
                                                                     img_metas,
                                                                     gt_bboxes_ignore_list=gt_bboxes_ignore,
                                                                     gt_labels_list=gt_labels,
                                                                     label_channels=label_channels)

        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness   #
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_anchor_preds_decode = [
            anchor_pred.reshape(-1, 4)
            for i, anchor_pred in enumerate(anchor_preds_lvl)
        ]
        flatten_anchor_preds = [
            anchor_pred.permute(0, 2, 3, 1).reshape(-1, 2)
            for i, anchor_pred in enumerate(anchor_preds)
        ]
        flatten_quality_preds = [
            quality_pred.permute(0, 2, 3, 1).reshape(-1)
            for quality_pred in quality_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)         # 将5层预测值concat 162136*80
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_anchor_preds = torch.cat(flatten_anchor_preds)
        flatten_anchor_preds_decode = torch.cat(flatten_anchor_preds_decode)
        flatten_quality_preds = torch.cat(flatten_quality_preds)

        flatten_labels = torch.cat([labels.reshape(-1) for labels in labels_list])                          # 将5层targets concat 162136*80
        flatten_labels_weights = [
            label_weights.reshape(-1)
            for label_weights in label_weights_list
        ]
        flatten_labels_weights = torch.cat(flatten_labels_weights)

        flatten_bbox_targets = torch.cat([bbox_targets.reshape(-1,4) for bbox_targets in bbox_targets_list])
        flatten_bbox_weights = [
            bbox_weights[:, :, 0].reshape(-1)
            for bbox_weights in bbox_weights_list
        ]
        flatten_bbox_weights = torch.cat(flatten_bbox_weights)

        # flatten_anchor_targets = torch.cat(anchor_targets)
        flatten_quality_targets = torch.cat(quality_targets)
        flatten_anchor_targets_stride = torch.cat(anchor_targets_stride)
        flatten_weights = torch.cat(anchor_weights)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes     # 80

        # pos_inds = ((flatten_labels >= 0)  # 用cls_labels得到正样本索引
        #             & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        pos_inds = torch.nonzero(flatten_quality_targets > 0).reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(                                   # 分类损失计算
            flatten_cls_scores, flatten_labels,
            weight=flatten_labels_weights,
            avg_factor=num_total_samples + num_imgs)  # avoid num_pos is 0

        pos_anchor_preds = flatten_anchor_preds[pos_inds]           # 预测的正样本anchor
        pos_quality_preds = flatten_quality_preds[pos_inds]         # 预测的正样本quality

        decoded_bbox_preds = self.bbox_coder.decode(flatten_anchor_preds_decode, flatten_bbox_preds)

        if num_pos > 0:
            pos_quality_targets = flatten_quality_targets[pos_inds]  # 正样本quality预测目标
            # 计算quality 的交叉熵损失函数
            loss_quality = self.loss_quality(pos_quality_preds, pos_quality_targets, avg_factor=num_pos)


            pos_weights = flatten_weights[pos_inds]                  # 正样本权重
            pos_anchor_targets_stride = flatten_anchor_targets_stride[pos_inds]   # 正样本anchor预测目标 归一化

            pos_points = flatten_points[pos_inds]                   # anchor正样本对应的网格坐标值
            pos_anchor_preds = torch.cat([pos_points - pos_anchor_preds, pos_points + pos_anchor_preds], 1)

            pos_anchor_targets = torch.cat([pos_points - pos_anchor_targets_stride, pos_points + pos_anchor_targets_stride], 1)

            # 计算anchor iou loss
            loss_anchor = self.loss_anchor(pos_anchor_preds,     # 计算anchor loss
                                           pos_anchor_targets,
                                           weight=pos_weights,
                                           avg_factor=pos_weights.sum())

            # 将预测的anchor和预测的偏移值 转化为预测的bbox (左上右下)
            # decoded_bbox_preds = self.bbox_coder.decode(flatten_anchor_preds_decode, flatten_bbox_preds)
            loss_bbox = self.loss_bbox(decoded_bbox_preds,   # 利用预测的bbox与gt_bbox计算IOUloss
                                       flatten_bbox_targets,
                                       weight=flatten_bbox_weights,
                                       avg_factor=num_total_samples + num_imgs)
        else:
            loss_bbox = flatten_bbox_preds[pos_inds].sum()
            loss_quality = pos_quality_preds.sum()
            loss_anchor = pos_anchor_preds.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_anchor=loss_anchor,
            loss_quality=loss_quality)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'quality_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   anchor_preds,
                   quality_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W).
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
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):          # 取出一張圖像的5層預測結果
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            anchor_pred_list = [
                anchor_preds[i][img_id].detach()*64*pow(2, i) for i in range(num_levels)
            ]
            quality_pred_list = [
                quality_preds[i][img_id].detach() for i in range(num_levels)
            ]

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(
                cls_score_list, bbox_pred_list, anchor_pred_list, quality_pred_list,
                mlvl_points, img_shape, scale_factor, cfg, rescale, with_nms)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           anchor_preds,
                           quality_preds,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
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
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_anchors = []
        mlvl_qualities = []
        for cls_score, bbox_pred, anchor_pred, quality_pred, points in zip(
                cls_scores, bbox_preds, anchor_preds, quality_preds, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            quality_pred = quality_pred.permute(1, 2, 0).reshape(-1).sigmoid()
            anchor_pred = anchor_pred.permute(1, 2, 0).reshape(-1, 2)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * quality_pred[:, None]).max(dim=1)
                # max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                anchor_pred = anchor_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                quality_pred = quality_pred[topk_inds]
            anchor_gen = torch.stack([
                                     points[:, 0] - anchor_pred[:, 0],
                                     points[:, 1] - anchor_pred[:, 1],
                                     points[:, 0] + anchor_pred[:, 0],
                                     points[:, 1] + anchor_pred[:, 1],
                                    ], -1)
            bboxes = self.bbox_coder.decode(anchor_gen, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_anchors.append(anchor_gen)
            mlvl_scores.append(scores)
            mlvl_qualities.append(quality_pred)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_anchors = torch.cat(mlvl_anchors)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_anchors /= mlvl_anchors.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_qualities = torch.cat(mlvl_qualities)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                mlvl_bboxes,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_qualities)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_qualities

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),  # 拼接並加上偏移
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
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
        anchor_targets_list, quality_targets_list, weights_list = multi_apply(
            # 为每一张图像构造预测目标
            self._get_target_single,
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

        # concat per level image
        #concat_lvl_labels = []
        #concat_lvl_bbox_targets = []
        #concat_lvl_anchor_targets = []
        concat_lvl_anchor_targets_stride = []
        concat_lvl_quality_targets = []
        concat_lvl_weight = []
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
        return concat_lvl_quality_targets, concat_lvl_anchor_targets_stride, concat_lvl_weight
        # return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_quality_targets, concat_lvl_anchor_targets_stride, concat_lvl_weight

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
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
        weight_pos = self.centerness_target(pos_bbox_targets)
        # weights[pos_inds] = weight_pos
        # left_x = points[..., 0] - anchor_targets[..., 0]
        # top_y = points[..., 1] - anchor_targets[..., 1]
        # right_x = points[..., 0] + anchor_targets[..., 0]
        # bottom_y = points[..., 1] + anchor_targets[..., 1]
        # boxes = torch.stack((left_x, top_y, right_x, bottom_y), -1)
        anchor_boxes = torch.cat((points - anchor_targets, points + anchor_targets), 1)
        quality_targets = self.BboxOverlaps(anchor_boxes, gt_bboxes)
        quality_targets[neg_inds] = 0          # 属于背景的特征点 quality 赋值为0
        # bbox_targets = self.bbox_coder.encode(boxes, gt_bboxes)      # GT坐标转化为偏移值
        # bbox_targets = gt_bboxes
        return anchor_targets, quality_targets, weight_pos
        # return labels, gt_bboxes, anchor_targets, quality_targets, weights

    def centerness_target(self, pos_bbox_targets):
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
