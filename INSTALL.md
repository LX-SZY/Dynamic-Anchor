# Prerequisites
Requirements: <br>
* Linux
* Python 3.6+
* PyTorch 1.3+
* CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
* GCC 5+
* [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

## Installation
### stpe 1 ：`installing mmdetcetion`
Install the mmdetcetion toolbox with GPU by following [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md)
### step 2 : `download mmdet file and config file`
Replace the `config` file and `mmdet` file  with the our [config](https://github.com/LX-SZY/dynamicanchor/tree/main/configs) and [mmdet](https://github.com/LX-SZY/dynamicanchor/tree/main/mmdet) 
### step 3 : `prepare the dataset`
download [COCO dataset](https://cocodataset.org/#download) <br>
Store in the form of the following directory：<br>
![](https://github.com/LX-SZY/Dynamic-Anchor/blob/main/data/img.png)  
