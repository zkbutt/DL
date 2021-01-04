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

s1 = [1, 2, 3]
s2 = [4, 3, 2]
print(np.true_divide(np.array(s1), np.array(s2)))
tensor_s1 = torch.tensor(s1)
tensor_s2 = torch.tensor(s2)
res = torch.true_divide(tensor_s1, tensor_s2)
print(res)
print(round(res[0].item(), 2))
res___ = res[res == 0.25]
res___[0] = 99
view = res.view(1, 1, -1)
view[0, 0, 1] = 9999
print(view)
print(res)
print(res___)
