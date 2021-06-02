_base_ = './DA_RetinaNet_r50_caffe_fpn_4x4_1x_coco.py'
model = dict(
    pretrained='open-mmlab://detectron/resnet101_caffe',
    backbone=dict(depth=101))
