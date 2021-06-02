# Dynamic anchor: A Feature-Guided Anchor Strategy for Object Detection
we achieve RetinaNet with dynamic anchor based on mmdetection
## Introduction
Code details of dynamic anchor
## Installation
This DA-RetinaNet implementation is based on [FCOS](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos) and [mmdetection](https://github.com/open-mmlab/mmdetection) and the installation is the same as mmdetection. Please check [INSTALL.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) for installation instructions.
## Inference
Once the installation is done, you can download DA-RetinaNet_r50.pth from Google or Baidu. The following command line will inference on coco minival split and please replace `root` with your `root directory` at first (for example: my root directory is "/home/yht/Demo1/mmdetection/") :
```
python tools/test.py root/configs/DA-RetinaNet/DA-RetinaNet_r50_caffe_fpn_4x4_1x_coco.py root/weights/DA-RetinaNet_r50_FPN_1x.pth  --eval bbox  --show-dir root/show_results/DA-RetinaNet_r50/
```
Please note that: <br>
>>1. `Config file` path and `weight file` path are best to use absolute paths.
>>2. `--show-dir` represents saving painted images with detection results.
>>3. `--eval` represents evaluating performance.
## Training
 The following command line will train DA-RetinaNet_r50_FPN_1x on 2 GPUs with Synchronous Stochastic Gradient Descent (SGD) and please replace the `root` with your `root directory` at first (for example: my root directory is "/home/yht/Demo1/mmdetection/") :
```
python tools/dist_train.sh  root/configs/DA-RetinaNet/DA-RetinaNet_r50_caffe_fpn_4x4_1x_coco.py  2
```
## Results and Models
we provide the following trained models. The AP All models are trained with 8 images in a mini-batch on 2 RTX 3090 GPUs. The AP is evaluated on coco test_dev split.<br>
|    Method     |   Backbone        |        Style     |         Lr schd    |    box AP         |        GFLPs       |  log      | Download     |
|:-----------:  |  :-------------:  |    :----------:  |    :-----------:   |    :-----------:  |  :----------:       |:-----------: | :---------:  |
| DA-RetinaNet  |     R-50-FPN      |     caffe        |       1x           |    38.0           |    141.79           |   [log](https://pan.baidu.com/s/10GzGxUuPTKFLnqVOePFooQ)/key:`2ahy`           |    [weight](https://pan.baidu.com/s/1j8YUSZruKpqiMoFdQjfT7Q)/key:`w787`         |
| DA-RetinaNet  |     R-101-FPN     |     caffe        |       1x           |    40.0           |    217.86           |   [log](https://pan.baidu.com/s/1VBdij73sNpYDYZhq3roewQ)/key:`sp8a`           |   [weight](https://pan.baidu.com/s/1_x3EfR8kgU7gqw9eiSUT6w)/key:`b4ay`         |
| DA-RetinaNet  |     R-50-FPN      |     caffe        |       2x           |                   |    141.79           |              |              |
| DA-RetinaNet  |     R-101-FPN     |     caffe        |       2x           |                   |    217.86           |              |              |
 
