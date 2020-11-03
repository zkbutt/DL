import numpy as np


def ecludDist(x, y):
    '''欧式距离'''
    return np.sqrt(sum(np.square(np.array(x) - np.array(y))))


def manhattanDist(x, y):
    '''曼哈顿距离'''
    return np.sum(np.abs(x - y))


def cos(x, y):
    '''夹角余弦'''
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
