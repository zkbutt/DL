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

a = torch.tensor([132.37500, 201.37500, 192.37500, 276.50000]).view(1, -1)
b = torch.tensor([91.13490, 111.55963, 237.08778, 250.51987]).view(1, -1)
print(1-bbox_iou4one_2d(a, b))
