import datetime
import glob
import json
import os
import random
import re

import h5py
import numpy as np
import pandas as pd

import time

import numpy as np
import time
import torch.nn.functional as F
import torch
import xmltodict

import sys

import logging
from tqdm import trange

from f_tools.fun_od.f_boxes import bbox_iou4one

# a = torch.randn((6, 5))
# print('a', a)
# # ind 是降维的索引
# val, ind = a.topk(2, dim=0)
# a[ind, torch.arange(val.shape[1])] = 999
# # val, ind = a.topk(2, dim=1)
# print(val)
# print(ind)
# # a[ ind] = 999
# print(val.shape)
# print(a)

# [3,4] * [4,1]
# a = torch.arange(2).reshape(1, 2).type(torch.float)
# a = F.softmax(a, dim=-1)
# # b = torch.arange(2).type(torch.float)
# b = torch.tensor([1,2]).type(torch.float)
# print(a, a.shape)
# print(b[None], b[None].shape)
#
# # [1,2] ^^ [1,2]
# print(F.linear(a, b[None]))
import os.path as osp
import subprocess
import sys
from collections import defaultdict

import cv2
import torch
import torchvision

# gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
# gcc = gcc.decode('utf-8').strip()
# print(gcc)
a = torch.tensor([
    [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
    [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]])
print(a.shape)
print(a[..., 1])
a[:, :, 1] = 999
print(a)
