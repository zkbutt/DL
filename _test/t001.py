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

from f_tools.fun_od.f_boxes import bbox_iou4one_2d

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

# for i in range(0, 32):
# for i in range(33, 41):
# for i in range(42, 50):
# for i in range(51, 59):
# for i in range(60, 67):
# for i in range(68, 76):
for i in range(77, 97):
    print('[%s,%s],' % (str(i + 1), str(i + 2)))

a=3
assert a==3,'adf'