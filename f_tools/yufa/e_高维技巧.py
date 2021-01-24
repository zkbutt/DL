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
    a_np = np.arange(1, 5).reshape(2, 2)
    b_ts = torch.tensor(a_np)
    print(a_np)
    print('单体复制')
    print(np.repeat(a_np, 2))
    print(torch.repeat_interleave(b_ts.view(-1), 2, dim=0))
    print(np.repeat(a_np, 2, axis=0))
    print(torch.repeat_interleave(b_ts, 2, dim=0))  # 默认dim=0
    print(np.repeat(a_np, 2, axis=1))
    print(torch.repeat_interleave(b_ts, 2, dim=1))
    print(np.repeat(a_np, [2, 3], axis=0))
    print(torch.repeat_interleave(b_ts, [2, 3], dim=0))  # 不支持数组

    print('整体复制')
    print(np.tile(a_np, (2, 1)))
    print(b_ts.repeat(2, 1))

    print('其它')
    print(np.concatenate([a_np, a_np], axis=0))  # 看着整体扩展,默认为 axis=0
    print(np.concatenate([a_np, a_np], axis=1))  # 看着整体

    print(b_ts.repeat(1, 2))  # 看着整体扩展
    print(torch.cat([b_ts, b_ts], dim=0))  # 与上面等价,看着整体扩展 默认为 dim=0
    print(torch.cat([b_ts, b_ts], dim=1))  # 与上面等价,看着整体扩展
    print(torch.cat([b_ts[:, :1].repeat(1, 2), b_ts[:, 1:2].repeat(1, 3)], dim=1))  # 交替扩展


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


def f_gather():
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
    dim0, dim1, dim2 = input.shape
    print(dim0, dim1, dim2)  # [2, 4, 6]
    # 定义在要选的上一维
    idx1 = torch.LongTensor([0, 1, 2, 5])  # 定义在前1维
    idx = idx.reshape(1, idx1.shape[0], 1)  # 扩维   要选的为维度为1 其它同维
    idx = idx.repeat(dim0, 1, 1)  # 一致
    out = torch.gather(input, dim=-1, index=idx)  # 其它都要匹配 选择那维单独处理

    # idx = torch.LongTensor([0, 3])  # 选[2, 4, 6]第1维则第1维匹配维度为1 定义在前一维0维长度
    # idx.unsqueeze_(-1).unsqueeze_(-1)
    # idx = idx.reshape(len(idx), 1, 1)
    # idx = idx.reshape(idx.shape[0], 1, 1)
    # idx = idx.repeat(1, 1, dim2)
    # print(idx.shape)  # [2, 1, 6]
    # out = torch.gather(input, dim=1, index=idx)  # 其它都要匹配 选择那维单独处理
    print(out)

    fill = input.index_fill(-1, idx1, 99)  # 直接使用1维索引
    print(fill)

    print(input.scatter_(-1, idx, 55))  # 与 gather 相同


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
    f_gather()
