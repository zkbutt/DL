import os
import torch
import random
import numpy as np
import torch.nn as nn


# set_seed(1)  # 设置随机种子


class MLP(nn.Module):  # 建立全连接模型
    def __init__(self, neural_num, layers):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.neural_num = neural_num

    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            # x = torch.tanh(x) # 加上tanh激活后可能会梯度消失

            print("layer:{}, std:{}".format(i, x.std()))
            if torch.isnan(x.std()):  # 如果出现inf nan，则停止
                print("output is nan in {} layers".format(i))
                break
        return x


def initialize(self):  # 初始化模型参数
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data)
        # 针对 Linear, 采用mean=0，标准差std=sqrt(1/n) n是神经元个数 解决Linear
        # nn.init.normal_(m.weight.data, std=np.sqrt(1/self.neural_num))

        # Xavier初始化
        # a = np.sqrt(6 / (self.neural_num + self.neural_num))  # Xavier初始化方法
        # tanh_gain = nn.init.calculate_gain('tanh')
        # a *= tanh_gain
        # nn.init.uniform_(m.weight.data, -a, a)
        # pytorch实现
        # nn.init.xavier_uniform_(w, gain=tanh_gain)


layer_nums = 100
neural_nums = 256
batch_size = 16

net = MLP(neural_nums, layer_nums)
net.initialize()

inputs = torch.randn((batch_size, neural_nums))  # normal: mean=0, std=1

output = net(inputs)
print(output)

'''
Xavier均匀分布；
	nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)
Xavier正态分布；
Kaiming均匀分布；
Kaiming正态分布； 前向传播选择fan_in, 反向传播选择fan_out 'relu', 'leaky_relu'
	nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')
均匀分布；
正态分布；
常数分布；
正交矩阵初始化；
单位矩阵初始化；
稀疏矩阵初始化；
'''

# 计算方差比
x = torch.randn(10000)
out = torch.tanh(x)

gain = x.std() / out.std()  # 输入的方差和输出的方差
print('gain:{}'.format(gain))

tanh_gain = nn.init.calculate_gain('tanh')
print('tanh_gain in PyTorch:', tanh_gain)
