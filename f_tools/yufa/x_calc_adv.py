import numpy as np
import torch


def x_select_1():
    '''
    输入:两个科目3个人对应的分数
    输出:选出的人ID
    要求:
        超过75分的人
        每个科目的最高分的人要选出 (即使没有超阀值也要选出来)
        没有对应关系
    '''
    # 选人不分类: 两个科目 (数学,语文) 有三个人,两科的分数,  每个老要选一个成绩最好的 (即使没有超阀值也要选出来) 和超过75的
    s = np.array([0, 1, 2])  # id为1,2,3的人
    fen = torch.tensor([
        [94, 55, 88],  # 数学
        [97, 78, 69],  # 语文
    ])
    # print(iou)
    # 降维运算 2,3 -> 3    每个学生所得的最高分, 最高分的科目
    s1, i1 = fen.max(dim=0)  # 98,78,88   0,1,0
    # 2,3 -> 2    每个科目最好的分数学生,最好学生的编号
    s2, i2 = fen.max(dim=1)  # 98,97   0,0
    # s1.index_fill_(0, i2, 999)
    s1[i2] = 999  # 每个科目最好的学生保留  让他的分超过阀值
    # 这f??
    # _ids = torch.arange(0, i2.shape[0], dtype=torch.int64)
    # i1[i2[_ids]] = _ids
    mask = s1 > 75
    print(s[mask])  # 选出的人ID
    # print(mask * 1.)
    print(i1[mask])  # 对应的科目


def x_select_2():
    '''
    输入:1个科目3个人对应的分数
    输出:选出的人ID
    要求:
        超过75分的人
        每个科目的最高分的人要选出 (即使没有超阀值也要选出来)
        没有对应关系
    '''
    # 选人不分类: 两个科目 (数学,语文) 有三个人,两科的分数,  每个老要选一个成绩最好的 (即使没有超阀值也要选出来) 和超过75的
    s = np.array([0, 1, 2])  # id为1,2,3的人
    fen = torch.tensor([94, 55, 99])
    # print(iou)
    # 降维运算 2,3 -> 3    每个学生所得的最高分, 最高分的科目
    s1, i1 = fen[None].max(dim=1)  # 96   1
    fen[i1] = 999  # 每个科目最好的学生保留  让他的分超过阀值
    # 这f??
    # _ids = torch.arange(0, i2.shape[0], dtype=torch.int64)
    # i1[i2[_ids]] = _ids
    mask = fen > 75
    print(s[mask])  # 选出的人ID


def batch_offset(boxes, idxs):
    # 根据最大的一个值确定每一类的偏移
    max_coordinate = boxes.max()  # 选出每个框的 坐标最大的一个值
    # idxs 的设备和 boxes 一致 , 真实类别index * (1+最大值) 则确保同类框向 左右平移 实现隔离
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    # boxes 加上对应层的偏移量后，保证不同类别之间boxes不会有重合的现象
    boxes_for_nms = boxes + offsets[:, None]


def x_select_max_num(labels_neg):
    _, labels_idx = labels_neg.sort(dim=1, descending=True)  # descending 倒序
    # 得每一个图片batch个 最大值索引排序
    _, labels_rank = labels_idx.sort(dim=1)  # 两次索引排序 用于取最大的n个布尔索引
    # 计算每一层的反例数   [batch] -> [batch,1]  限制正样本3倍不能超过负样本的总个数  基本不可能超过总数
    neg_num = torch.clamp(self.neg_ratio * pos_num, max=mask_pos.size(1)).unsqueeze(-1)
    mask_neg = labels_rank < neg_num  # 选出最大的n个的mask  Tensor [batch, 8732]
    # 正例索引 + 反例索引 得1 0 索引用于乘积筛选
    mask_z = mask_pos.float() + mask_neg.float()
    loss_labels = (loss_labels * (mask_z)).sum(dim=1)


def f_mershgrid(row, col, is_rowcol=True, num_repeat=1):
    '''

    :param row:  y 需要加 row
    :param col:  x 需要加 col
    :param is_rowcol:
    :param num_repeat:
    :return:
    '''
    a = torch.arange(row)  # 3行 4列
    b = torch.arange(col)
    x, y = torch.meshgrid(a, b)  # row=3 col=5
    '''
    x ([[0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2]])
    y ([[0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4]])
    '''

    # for i in range(3):
    #     for j in range(4):
    # print("(", x[i, j], ",", y[i, j], ")")
    # print(x)
    # print(y)

    if is_rowcol:
        stack = torch.stack((x, y), dim=2)
    else:
        stack = torch.stack((y, x), dim=2)
    stack = torch.repeat_interleave(stack.view(-1, 2), num_repeat, dim=0)
    return stack


def t_ids2bool():
    # topk索引
    a = torch.arange(9).reshape(3, 3).type(torch.float)
    a[1, 2] = 1
    print(a)

    b = torch.zeros_like(a, dtype=torch.bool)
    print(b)

    # 3,3 -> 2,3
    topk_val, topk_index = a.topk(2, dim=0)
    # print(topk_val)
    print(topk_index)

    res = a[topk_index, torch.arange(3)]  # 3,3 ^^ [[2,3],[ngt]] 降维

    print(res)
    print(res.mean(0))
    print(res.shape)
    b[topk_index, torch.arange(3)] = 1
    print(b)


if __name__ == '__main__':
    # x_select_1()
    # x_select_2()
    # print(f_mershgrid(3, 4, num_repeat=1, is_rowcol=True))
    # a = torch.arange(5)
    # b = torch.arange(5, 10)
    # print(torch.stack([a, b], dim=1))  # 升级连接
    t_ids2bool()
