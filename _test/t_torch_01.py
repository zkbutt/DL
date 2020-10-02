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


class MySquares():

    def __init__(self) -> None:
        self.num = 0

    def __iter__(self):  # 返回自身的迭代器
        return self

    def __next__(self):
        flog.debug('__next__ %s',self.num )
        # ret = torch.rand((300, 300))
        ret = self.num
        # 这里是预加载
        if self.num ==5:
            raise StopIteration()
        # self.next_input, self.next_target = next(self.loader)  # 加载到显存
        self.num += 1
        return ret


tt = MySquares()
for i in iter(tt):
    print(i)



# y.backward(torch.FloatTensor([1, 0.1, 0.01]))  # 自动求导
# print(x.grad)  # 求对x的梯度
