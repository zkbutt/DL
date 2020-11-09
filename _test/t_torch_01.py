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

from f_tools.GLOBAL_LOG import flog

# input = torch.randn(2, 4, requires_grad=True)
# for i in input.split(1, 1):
#     print(i.shape)

# y.backward(torch.FloatTensor([1, 0.1, 0.01]))  # 自动求导
# print(x.grad)  # 求对x的梯度


# idxs_img = torch.arange(5).type(torch.float)
# print(idxs_img.repeat(2))

# input = torch.randn(3, 5)
# (5) -> (1,5)
# idxs_img = idxs_img.view(-1, 1).repeat(1, 5)
# print(idxs_img.view(idxs_img.shape[0],-1))
# np.set_printoptions(suppress=True)  # 关闭科学计数


# tensor = torch.tensor([-1, 3, 99, -7, 0, 1.])
# print(torch.sigmoid(input))
# print(torch.sigmoid(tensor))

# ones = torch.rand(4, 3, 4)
# mask = ones > 0.5
# mask_ = ones[mask]
# print(mask_.shape)
print(torchvision.__version__)