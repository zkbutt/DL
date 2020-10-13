import os
from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.models.detection.faster_rcnn
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch

from f_tools.datas.data_factory import CreVOCDataset
from f_tools.pic.f_show import show_od4pil
from object_detection.simple_faster_rcnn_pytorch_master.data.dataset import Dataset
from object_detection.simple_faster_rcnn_pytorch_master.utils.config import opt

if __name__ == '__main__':
    path_datas = os.path.join('M:\AI\datas\m_VOC2007', 'trainval')
    dataset = Dataset(opt)

    for img, bbox, label, scale in dataset:
        pool = torchvision.ops.roi_pool(torch.tensor(img).reshape(1, *img.shape), [torch.tensor(bbox)], 7, spatial_scale=1.0)

        print(pool)
        show_od4pil(np.transpose(img, (2, 1, 0)), bbox)
        show_od4pil(np.transpose(np.transpose(torch.squeeze(pool).numpy(), (2, 1, 0)), (2, 1, 0)), bbox)
        # resize_img_keep_ratio()

    # image, target = iter(train_data_set).__next__()
