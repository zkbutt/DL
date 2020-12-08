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

# for i in trange(100):
#     # do something
#     time.sleep(0.5)
#     pass
import argparse

from collections import OrderedDict

from f_tools.GLOBAL_LOG import flog


def labels2onehot4ts(labels, num_class):
    batch = labels.shape[0]
    labels.resize_(batch, 1)  # labels[:,None]
    zeros = torch.zeros(batch, num_class)
    onehot = zeros.scatter_(1, labels, 1)  # dim,index,value
    return onehot


if __name__ == '__main__':
    '''
    usage: t001.py [-h] [--name NAME] -f FAMILY t4 integers [integers ...]
    '''
    import torch.nn.functional as F

    # random_ = torch.LongTensor(5).random_() % 4
    # random_ = torch.tensor([1])
    # print(labels2onehot4ts(random_, 4))
    ious1 = torch.tensor([[2, 1, 3], [2, 1, 1]], dtype=torch.float)
    ious2 = torch.tensor([[2, 1, 3], [2, 1, 6]], dtype=torch.float)
    # ious = torch.tensor([[2], [1], [3]])
    print(ious1[[0, 1], :])
    # print(ious.max(dim=1))
    print(F.mse_loss(ious1, ious2, reduction='none'))

    # nB, nA, nG, nG = 4, 2, 6, 6
    # obj_mask = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    # noobj_mask = torch.ByteTensor(nB, nA, nG, nG).fill_(1)
    # class_mask = torch.FloatTensor(nB, nA, nG, nG).fill_(0)
    # iou_scores = torch.FloatTensor(nB, nA, nG, nG).fill_(0)
    # print(torch.zeros(nB, nA, nG, nG,dtype=torch.float).shape)
    # print(obj_mask.shape)
