"""
tools to convert specified type
"""
import torch as t
import numpy as np


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t.Tensor):
        return data.detach().cpu().numpy()


def totensor(data):
    if isinstance(data, np.ndarray):
        tensor = t.from_numpy(data)
    if isinstance(data, t.Tensor):
        tensor = data.detach()
    return tensor


def scalar(data):
    ''' 输入 np或tensor 变成标量'''
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, t.Tensor):
        return data.item()