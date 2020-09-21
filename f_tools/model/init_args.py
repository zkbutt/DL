import torch

'''
卷积参数的初始化
'''


def init1():
    for layer in self.children():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)
