import cv2
import numpy as np
from pycocotools.coco import COCO
import os.path as osp
import os
import json
import matplotlib.pyplot as plt
# cocoRoot = "/home/yht/Demo1/mmdetection/data/coco"
# dataType = "val2017"
# annFile = osp.join(cocoRoot, f'annotations/instances_{dataType}.json')
# coco = COCO(annFile)
root = "/home/yht/Demo1/mmdetection/out_imgs"
file_list = os.listdir(root)
dit = {}
for i in range(3):
    dit[i] = {0: 0, 1: 0, 2: 0}
for file in file_list:
    file = os.path.join(root, file)
    with open(file, "r") as f:
        info = json.load(f)
        for i in range(3):
            for j in range(3):
                dit[i][j] += info[f"{i}"][f"{j}"]
print(dit)
plt.figure()
large = [dit[0][0], dit[0][1], dit[0][2]]
medium = [dit[1][0], dit[1][1], dit[1][2]]
small = [dit[2][0], dit[2][1], dit[2][2]]
large = np.array(large)/sum(large)
medium = np.array(medium)/sum(medium)
small = np.array(small)/sum(small)
name_list = ["level 0", "level 1", "level 2"]
x = [0, 1, 2]
total_width, n = 0.9, 3
width = total_width / n
plt.bar(x, large, width=width, label='large', fc='y')
x = [t+width for t in x]
plt.bar(x, medium, width=width, label='medium', tick_label=name_list, fc='lightsteelblue')
x = [t+width for t in x]
plt.bar(x, small, width=width, label='small', fc='lightseagreen')
plt.legend()
plt.savefig("/home/yht/Demo1/mmdetection/out/statistic.pdf")
plt.show()

# path = '/home/yht/Demo1/mmdetection/data/coco/val2017/000000226111.jpg'
# img = cv2.imread(path)
# cv2.imshow("img", img)
# cv2.waitKey(0)
# bbox = np.array([[3.8584e+02,  7.3369e+01,  4.9918e+02,  3.5313e+02,  9.9333e-01],
#         [ 2.9246e+01,  3.4254e+02,  1.0212e+02,  3.8442e+02,  9.2755e-01],
#         [ 6.1848e+01,  2.8899e+02,  1.3423e+02,  3.3052e+02,  8.8990e-01],
#         [ 2.5917e-01,  2.5610e+02,  6.0181e+01,  3.0550e+02,  6.8982e-01],
#         [ 1.5488e+02,  1.6816e+02,  1.8368e+02,  1.8309e+02,  6.8501e-01],
#         [ 5.2446e+02,  3.8076e+01,  5.4000e+02,  1.1971e+02,  5.5068e-01],
#         [ 4.9545e+02,  1.9980e+02,  6.0736e+02,  2.2473e+02,  3.9282e-01],
#         [ 4.8873e+02,  4.3318e+01,  4.9795e+02,  9.4423e+01,  3.7379e-01],
#         [ 1.4306e+02,  2.6602e+02,  1.7353e+02,  3.0316e+02,  3.6242e-01],
#         [ 1.3591e+02,  2.4897e+02,  1.5270e+02,  2.7658e+02,  3.5737e-01],
#         [ 5.3498e+02,  4.2342e+01,  5.5113e+02,  1.2117e+02,  3.5541e-01],
#         [ 4.3467e+02,  3.6118e+01,  4.5066e+02,  7.1239e+01,  2.6619e-01],
#         [ 5.1100e+02,  4.4324e+01,  5.2334e+02,  9.4564e+01,  2.1062e-01],
#         [ 9.2182e+01,  3.6488e+02,  1.2016e+02,  3.8760e+02,  1.8395e-01],
#         [ 3.9665e+02,  7.9561e+01,  4.0458e+02,  1.3346e+02,  1.6298e-01],
#         [ 1.5611e+02,  1.1475e+02,  1.7493e+02,  1.2805e+02,  1.3636e-01],
#         [ 4.8013e+02,  4.5039e+01,  4.8622e+02,  9.5673e+01,  1.2547e-01],
#         [ 4.9545e+02,  1.9980e+02,  6.0736e+02,  2.2473e+02,  1.1644e-01],
#         [ 9.2182e+01,  3.6488e+02,  1.2016e+02,  3.8760e+02,  1.1285e-01],
#         [ 9.2182e+01,  3.6488e+02,  1.2016e+02,  3.8760e+02,  1.0611e-01],
#         [ 1.2127e+02,  2.7241e+02,  1.4282e+02,  3.0736e+02,  9.9402e-02],
#         [ 2.5318e+02,  2.6431e+02,  2.9295e+02,  2.8099e+02,  9.6699e-02],
#         [ 2.5216e+02,  1.5920e+02,  3.8797e+02,  2.1623e+02,  9.6070e-02],
#         [-1.7682e+01,  1.9488e+02,  3.4870e+02,  4.0911e+02,  9.2109e-02],
#         [-2.0562e+00,  3.0566e+02,  8.5979e+01,  3.6079e+02,  9.0277e-02],
#         [ 4.5037e+02,  3.8270e+01,  4.6403e+02,  7.1690e+01,  8.1524e-02],
#         [ 4.0597e+02,  7.0781e+01,  4.1486e+02,  1.2424e+02,  7.9123e-02],
#         [ 4.3645e-01,  3.1326e+02,  8.4598e+01,  3.6498e+02,  7.4388e-02],
#         [ 2.4596e+02,  3.1972e+01,  2.8259e+02,  9.1241e+01,  6.8527e-02],
#         [ 5.5538e+02,  4.9439e+01,  5.6512e+02,  1.0887e+02,  6.4137e-02],
#         [ 2.2591e+02,  1.9161e+02,  3.5866e+02,  3.7283e+02,  5.9637e-02],
#         [ 3.4025e+01,  2.0835e+02,  1.9285e+02,  2.8971e+02,  5.5302e-02]])
# for i in range(bbox.shape[0]):
#     box = bbox[i, :-1]
#     cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
#     cv2.imshow("img", img)
#
#     cv2.waitKey(0)
# id = 226111
# anns_id = coco.getAnnIds(imgIds=id)
# anns = coco.loadAnns(anns_id)
# for ans in anns:
#     bbox = ans["bbox"]
#     print(ans["category_id"])
#     cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 255, 0), 2)
# cv2.imshow("img", img)
# cv2.waitKey(0)
