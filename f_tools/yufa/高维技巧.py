import torch


def f布尔同维1():
    # 布尔同维 选出满足条件的行或列
    zeros = torch.zeros((3, 4))
    zeros[:, ::2] = 1
    print(zeros)
    axis1, axis2 = torch.where(zeros == 1)
    print(axis1, torch.unique(axis2))
    t1 = torch.arange(6).reshape(3, 2).type(torch.float)
    zeros[:, torch.unique(axis2)] = t1

    print(zeros)


if __name__ == '__main__':
    t1 = torch.arange(8)
    t2 = t1.reshape(2, 4)
    # t2[:, 1] = 55 # 降维索引
    # t2[:, 1:2] = torch.zeros((t2.shape[0], 1))  # 同维索引
    # print(t2[:, [1, 2]])  # 同维索引

    # t2[t2 > 2] = 999  # 布尔降维索引
    # t2[torch.where(t2 > 2)] = 888  # 布尔降维索引

    f布尔同维1()
