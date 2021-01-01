import torch
import numpy as np


def f布尔同维1():
    # 布尔同维 选出满足条件的行或列
    zeros = torch.zeros((3, 4))
    zeros[:, ::2] = 1
    print(zeros)
    axis1, axis2 = torch.where(zeros == 1)
    print(axis1, torch.unique(axis2))  # tensor([0, 0, 1, 1, 2, 2]) tensor([0, 2])
    t1 = torch.arange(6).reshape(3, 2).type(torch.float)

    zeros[:, torch.unique(axis2)] = t1

    print(zeros)


def t_axis():
    a = np.arange(1, 5).reshape(2, 2)
    b = torch.tensor(a)
    print(a)
    print('单体复制')
    print(np.repeat(a, 2))
    print(torch.repeat_interleave(b.view(-1), 2, dim=0))
    print(np.repeat(a, 2, axis=0))
    print(torch.repeat_interleave(b, 2, dim=0))  # 默认dim=0
    print(np.repeat(a, 2, axis=1))
    print(torch.repeat_interleave(b, 2, dim=1))
    print(np.repeat(a, [2, 3], axis=0))
    print(torch.repeat_interleave(b, [2, 3], dim=0))  # 不支持数组

    print('整体复制')
    print(np.tile(a, (2, 1)))
    print(b.repeat(2, 1))

    print('其它')
    print(np.concatenate([a, a], axis=0))  # 看着整体扩展,默认为 axis=0
    print(np.concatenate([a, a], axis=1))  # 看着整体

    print(b.repeat(1, 2))  # 看着整体扩展
    print(torch.cat([b, b], dim=0))  # 与上面等价,看着整体扩展 默认为 dim=0
    print(torch.cat([b, b], dim=1))  # 与上面等价,看着整体扩展
    print(torch.cat([b[:, :1].repeat(1, 2), b[:, 1:2].repeat(1, 3)], dim=1))  # 交替扩展


# torch.repeat ~~~ np.tile  np.repeat ~~~ torch.repeat_interleave

def t_交叉运算():
    a = torch.arange(0, 4).reshape(2, 2)
    b = torch.arange(0, 6).reshape(3, 2)
    # 2,1,2 *** 3,2 -> 2,3,2  交叉运算
    none_b = a[:, None, :] - b
    print(none_b)
    print(none_b.shape)


def x_高级1(labels):
    random_ = torch.LongTensor(5).random_() % 4
    zeros = torch.zeros(5, 20)
    onehot = zeros.scatter_(1, labels, 1)  # dim,index,value
    return onehot


def x_高级2(labels):
    a = np.arange(8).reshape(2, 2, 2)
    print('原数组：')
    print(a)
    print('--------------------')
    print(np.swapaxes(a, 2, 0))  # 交换数组
    print('--------------------')
    print(np.rollaxis(a, 2))  # 向后滚动特定的轴到一个特定位置
    return labels


def f数据高级():
    input = [
        [2, 3, 4, 5, 0, 0],
        [1, 4, 3, 0, 0, 0],
        [4, 2, 2, 5, 7, 0],
        [1, 0, 0, 0, 0, 0]
    ]
    input = torch.tensor(input)
    # 4,6 ^ 4,1 = 4,1
    idx = torch.LongTensor([3, 2, 4, 0])
    idx.unsqueeze_(-1)
    # 数据索引取数
    out = torch.gather(input, dim=1, index=idx)
    print(out)  # tensor([[5],    [3],    [7],    [1]])

    input = [
        [[2, 3, 4, 5, 0, 0],
         [1, 4, 3, 0, 0, 0],
         [4, 2, 2, 5, 7, 0],
         [1, 0, 0, 0, 0, 0]],
        [[2, 3, 4, 5, 0, 0],
         [1, 4, 3, 0, 0, 0],
         [4, 2, 2, 5, 7, 0],
         [1, 0, 0, 0, 0, 0]],
    ]
    input = torch.tensor(input)
    print(input.shape)
    # 多维数组索引 [2, 4, 6] 只要每批的  第0行和1行 选择的在第0维, 最后一维需要复制匹配
    idx = torch.LongTensor([0, 2])
    idx.unsqueeze_(-1).unsqueeze_(-1)
    idx = idx.repeat(1, 1, input.shape[-1])
    out = torch.gather(input, dim=1, index=idx)
    print(out)

if __name__ == '__main__':
    # t1 = torch.arange(8)
    # t2 = t1.reshape(2, 4)
    # t2[:, 1] = 55 # 降维索引
    # t2[:, 1:2] = torch.zeros((t2.shape[0], 1))  # 同维索引
    # print(t2[:, [1, 2]])  # 同维索引

    # t2[t2 > 2] = 999  # 布尔降维索引
    # t2[torch.where(t2 > 2)] = 888  # 布尔降维索引

    # f布尔同维1()
    # 反序
    # arr = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    # tensor = torch.tensor(arr)
    # print(tensor.numpy())
    # x_高级2(1)

    # t_交叉运算()
    f数据高级()
