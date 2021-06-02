from abc import abstractmethod

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                            build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from mmdet.core import multi_apply, anchor_inside_flags, unmap, images_to_levels
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
import numpy as np
from mmcv.ops import DeformConv2d


class FeatureAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deform_groups=4):
        super(FeatureAttention, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            2, deform_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deform_groups=deform_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption, std=0.01)

    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.conv_adaption(x, offset))
        return x


@HEADS.register_module()
class PseudoAnchorHead(BaseDenseHead, BBoxTestMixin):
    """Anchor-free head (FCOS, Fovea, RepPoints, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        stacked_convs (int): Number of stacking convs of the head.
        strides (tuple): Downsample factor of each feature map.
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
    """  # noqa: W605

    _version = 1

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 reg_decoded_bbox=True,
                 stacked_convs=4,
                 strides=(8, 16, 32, 64, 128),
                 dcn_on_last_conv=True,
                 deform_groups=4,
                 conv_bias=True,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 # loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=(.0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0)),
                 conv_cfg=None,
                 norm_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(PseudoAnchorHead, self).__init__()
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.reg_decoded_bbox = reg_decoded_bbox
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False
        self.deform_groups = deform_groups
        # self.len_convs = 2
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        # self._init_cls_convs()
        # self._init_reg_convs()
        # .....................
        self._init_anchor_convs()
        # .....................
        self._init_predictor()
        # .....................

        self.feature_attention = FeatureAttention(
            self.in_channels,
            self.feat_channels,
            kernel_size=3,
            deform_groups=self.deform_groups)

    # def _init_cls_convs(self):
    #     """Initialize classification conv layers of the head."""
    #     self.cls_convs = nn.ModuleList()
    #     for i in range(self.len_convs):
    #         chn = self.in_channels if i == 0 else self.feat_channels
    #         if self.dcn_on_last_conv and i == self.len_convs - 1:
    #             conv_cfg = dict(type='DCNv2')
    #         else:
    #             conv_cfg = self.conv_cfg
    #         self.cls_convs.append(
    #             ConvModule(
    #                 chn,
    #                 self.feat_channels,
    #                 3,
    #                 stride=1,
    #                 padding=1,
    #                 conv_cfg=conv_cfg,
    #                 norm_cfg=self.norm_cfg,
    #                 bias=self.conv_bias))

    # def _init_reg_convs(self):
    #     """Initialize bbox regression conv layers of the head."""
    #     self.reg_convs = nn.ModuleList()
    #     for i in range(self.len_convs):
    #         chn = self.in_channels if i == 0 else self.feat_channels
    #         if self.dcn_on_last_conv and i == self.len_convs - 1:
    #             conv_cfg = dict(type='DCNv2')
    #         else:
    #             conv_cfg = self.conv_cfg
    #         self.reg_convs.append(
    #             ConvModule(
    #                 chn,
    #                 self.feat_channels,
    #                 3,
    #                 stride=1,
    #                 padding=1,
    #                 conv_cfg=conv_cfg,
    #                 norm_cfg=self.norm_cfg,
    #                 bias=self.conv_bias))

    # .............................................................
    def _init_anchor_convs(self):
        self.anchor_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.anchor_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
    # .............................................................

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.decline_channels = 128
        self.fusion_conv_x = nn.Conv2d(self.feat_channels, self.decline_channels, 1, padding=0)
        self.fusion_conv_y = nn.Conv2d(self.feat_channels, self.decline_channels, 1, padding=0)
        self.fusion = nn.Conv2d(self.feat_channels, 256, 1, padding=0)
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        # ................................................................
        self.conv_anchor = nn.Conv2d(self.feat_channels, 2, 3, padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        # for m in self.cls_convs:
        #     if isinstance(m.conv, nn.Conv2d):
        #         normal_init(m.conv, std=0.01)
        # for m in self.reg_convs:
        #     if isinstance(m.conv, nn.Conv2d):
        #         normal_init(m.conv, std=0.01)
    # .........................................
        for m in self.anchor_convs:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
    # ........................................
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)
    # .........................................................
        normal_init(self.conv_anchor, std=0.01)

        normal_init(self.fusion_conv_x, std=0.01)
        normal_init(self.fusion_conv_y, std=0.01)
        normal_init(self.fusion, std=0.01)

        self.feature_attention.init_weights()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Hack some keys of the model state dict so that can load checkpoints
        of previous version."""
        version = local_metadata.get('version', None)
        if version is None:
            # the key is different in early versions
            # for example, 'fcos_cls' become 'conv_cls' now
            bbox_head_keys = [
                k for k in state_dict.keys() if k.startswith(prefix)
            ]
            ori_predictor_keys = []
            new_predictor_keys = []
            # e.g. 'fcos_cls' or 'fcos_reg'
            for key in bbox_head_keys:
                ori_predictor_keys.append(key)
                key = key.split('.')
                conv_name = None
                if key[1].endswith('cls'):
                    conv_name = 'conv_cls'
                elif key[1].endswith('reg'):
                    conv_name = 'conv_reg'
                elif key[1].endswith('anchor'):
                    conv_name = 'conv_anchor'
                elif key[1].endswith('centerness'):
                    conv_name = 'conv_centerness'
                else:
                    assert NotImplementedError
                if conv_name is not None:
                    key[1] = conv_name
                    new_predictor_keys.append('.'.join(key))
                else:
                    ori_predictor_keys.pop(-1)
            for i in range(len(new_predictor_keys)):
                state_dict[new_predictor_keys[i]] = state_dict.pop(
                    ori_predictor_keys[i])
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def forward(self, feats):                                      # 前向計算入口
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually contain classification scores and bbox predictions.
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * 4.
        """
        return multi_apply(self.forward_single, feats)[:2]

    def forward_single(self, x):                                 #
        """Forward features of a single scale levle.

        Args:
            x (Tensor): FPN feature maps of the specified stride.

        Returns:
            tuple: Scores for each class, bbox predictions, features
                after classification and regression conv layers, some
                models needs these features like FCOS.
        """
        anchor_feat = x

        for anchor_layer in self.anchor_convs:
            anchor_feat = anchor_layer(anchor_feat)
        anchor_pred = self.conv_anchor(anchor_feat)

        x = self.feature_attention(x, anchor_pred)
        x = self.fusion_conv_x(x)
        y = self.fusion_conv_y(anchor_feat)
        feat = torch.cat((x, y), dim=1)
        feat = self.fusion(feat)
        reg_feat = feat
        cls_score = self.conv_cls(feat)
        bbox_pred = self.conv_reg(reg_feat)
        return cls_score, bbox_pred, anchor_pred, anchor_feat, reg_feat
        # .........................................................

    @abstractmethod
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
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
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """

        raise NotImplementedError

    @abstractmethod
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space
        """

        raise NotImplementedError

    @abstractmethod
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
        """
        raise NotImplementedError

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
        return y, x

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

    def get_inside_flag(self, featmap_sizes, img_metas, device='cuda', num_base_anchors=1):

        inside_flag_list = []

        for img_id, img_meta in enumerate(img_metas):    # 遍历4张图
            multi_level_flags = []
            for i in range(len(featmap_sizes)):  # 遍历5层 lvl   # i=0
                anchor_stride = self.strides[i]  # 8
                feat_h, feat_w = featmap_sizes[i]  # torch.Size([136, 100])
                h, w = img_meta['pad_shape'][:2]  # 1088，800
                valid_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_w = min(int(np.ceil(w / anchor_stride)), feat_w)

                assert valid_h <= feat_h and valid_w <= feat_w

                valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
                valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
                valid_x[:valid_w] = 1
                valid_y[:valid_h] = 1
                valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
                valid = valid_xx & valid_yy
                valid = valid[:, None].expand(valid.size(0),  # 将一个特征点的一个锚框标志扩展为多个【多个anchor】
                                              num_base_anchors).contiguous().view(-1)
                multi_level_flags.append(valid)
            inside_flag_list.append(multi_level_flags)
        return inside_flag_list

    def _meshgrid(self, x, y, row_major=True):
        """Generate mesh grid of x and y.

        Args:
            x (torch.Tensor): Grids of x dimension.
            y (torch.Tensor): Grids of y dimension.
            row_major (bool, optional): Whether to return y grids first.
                Defaults to True.

        Returns:
            tuple[torch.Tensor]: The mesh grids of x and y.
        """
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def get_anchor_preds(self, all_level_points, anchor_preds):
        assert len(all_level_points) == len(anchor_preds)

        num_lvl = len(all_level_points)
        num_imgs = anchor_preds[0].shape[0]
        all_imgs_anchor_preds = []
        for img_id in range(num_imgs):   # 取出每张图像的anchor_preds_list 5层 []
            all_level_anchor_preds = [
                anchor_pred[img_id].permute(1, 2, 0).reshape(-1, 2) * 64 * pow(2, i)   # 将归一化的预测值转换为真正值
                for i, anchor_pred in enumerate(anchor_preds)
            ]
            for i in range(num_lvl):
                all_level_anchor_preds[i] = torch.cat([all_level_points[i] - all_level_anchor_preds[i],
                                                      all_level_anchor_preds[i] + all_level_points[i]], 1)
            all_imgs_anchor_preds.append(all_level_anchor_preds)

        return all_imgs_anchor_preds

    def get_clc_reg_targets(self,
                            anchor_list,                                # list[img1_anchor, img2_anchor, ----]
                            valid_flag_list,
                            gt_bboxes_list,
                            img_metas,
                            gt_bboxes_ignore_list=None,
                            gt_labels_list=None,
                            label_channels=1,
                            unmap_outputs=True,
                            return_sampling_results=False
                            ):

        num_imgs = len(img_metas)  # 4

        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]  # 每层特征图的锚框数量
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):  # 将每一张图像中的锚框收集到一个list中
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))
        anchor_preds_lvl = images_to_levels(concat_anchor_list, num_level_anchors)
        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_clc_reg_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list,)
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results), anchor_preds_lvl

    def _get_clc_reg_targets_single(self,                         # 获取一张图片每层特征图的预测目标
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,     # 判断锚框否否处于允许边界内
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():         # any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]  # 取出在边界内的锚框

        assign_result = self.assigner.assign(    # IOU分配真实框
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,   # 采样正样本和负样本
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds     # 正锚框的索引
        neg_inds = sampling_result.neg_inds     # 负锚框的索引
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(         # 正锚框的预测目标编码为偏移值
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
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
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)












    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)
