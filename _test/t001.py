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
    # random_ = torch.LongTensor(5).random_() % 4
    # random_ = torch.tensor([1])
    # print(labels2onehot4ts(random_, 4))
    bbox_index = torch.tensor([1, 1, 1])
    anc_index = torch.tensor([0, 0])
    _ids = torch.arange(0, anc_index.size(0), dtype=torch.int64).to(bbox_index)
    bbox_index[anc_index[_ids]] = _ids
    print(bbox_index)

    # nB, nA, nG, nG = 4, 2, 6, 6
    # obj_mask = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    # noobj_mask = torch.ByteTensor(nB, nA, nG, nG).fill_(1)
    # class_mask = torch.FloatTensor(nB, nA, nG, nG).fill_(0)
    # iou_scores = torch.FloatTensor(nB, nA, nG, nG).fill_(0)
    # print(torch.zeros(nB, nA, nG, nG,dtype=torch.float).shape)
    # print(obj_mask.shape)
#
