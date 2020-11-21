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
    a = torch.arange(0, 4).reshape(2, 2)
    b = torch.arange(0, 6).reshape(3, 2)
    a = [0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9]
    a = np.array(a)
    flog.debug('123%s', '444')
