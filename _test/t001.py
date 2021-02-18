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

import torch
import xmltodict

import sys

import logging
from tqdm import trange

from f_tools.fun_od.f_boxes import bbox_iou4one

a = torch.randn((6, 5))
print('a', a)
# ind 是降维的索引
val, ind = a.topk(2, dim=0)
a[ind, torch.arange(val.shape[1])] = 999
# val, ind = a.topk(2, dim=1)
print(val)
print(ind)
# a[ ind] = 999
print(val.shape)
print(a)
