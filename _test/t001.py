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

if __name__ == '__main__':
    '''
    usage: t001.py [-h] [--name NAME] -f FAMILY t4 integers [integers ...]
    '''
    # box = '11.01,11.02,11.03,11.05'
    # split = box.split(',')
    # # print(map(int, split))
    # print(list(map(int, map(float, split))))
    print(glob.glob('*.py'))
    # parser = argparse.ArgumentParser(description='命令行中传入一个数字')
    # parser.add_argument('t4', type=str, help='传入数字')  # 必须填
    # parser.add_argument('integers', type=str, nargs='+', help='传入数字')  # 至少传一个
    # parser.add_argument('--name', type=str, help='传入姓名', )  # --表示可选参数, required必填
    # parser.add_argument('-f', '--family', default='张三的家', type=str, help='传入姓名', required=True)  # --表示可选参数
    # args = parser.parse_args()
    # print(args)
    # print(args.integers)
    # print(args.name)
    # print(args.family)
    # r = range(3, 6)
    # for i in r:
    #     print(i)
    # print(sys.float_info.min)

    print(random.randint(1,3))
    # a = np.arange(1, 5).reshape(2, 2)
    # print(a)
    # print(np.repeat(a, 2))
    # print(np.repeat(a, 2, axis=0))
    # print(np.repeat(a, 2, axis=1))  # 看着个体
    # print(np.concatenate([a, a], axis=0))  # 看着整体扩展,默认为 axis=0
    # print(np.concatenate([a, a], axis=1))  # 看着整体
    # b = torch.tensor(a)
    # print(b.repeat(1, 2))  # 看着整体扩展
    # print(torch.cat([b, b], dim=0))  # 与上面等价,看着整体扩展 默认为 dim=0
    # print(torch.cat([b, b], dim=1))  # 与上面等价,看着整体扩展
    # print(torch.cat([b[:, :1].repeat(1, 2), b[:, 1:2].repeat(1, 3)], dim=1))  # 交替扩展
