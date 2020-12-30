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
print(torch.true_divide(torch.tensor(s1), torch.tensor(s2)))
