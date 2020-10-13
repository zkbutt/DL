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


input = torch.randn(2, 5, requires_grad=True)
print(input)
input = (input + 8) * 2
print(input)

# y.backward(torch.FloatTensor([1, 0.1, 0.01]))  # 自动求导
# print(x.grad)  # 求对x的梯度
