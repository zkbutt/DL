import torch


def f_get_rowcol_index(row, col):
    '''
    先行再列
    :param row:
    :param col:
    :return:
        (nn,[row_index,col_index])
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

    # 升维组装 每个元素取一个 3,4 ^^ 3,4 = 3,4,2
    rowcol_index = torch.stack((x, y), dim=2)
    # print(stack.view(-1, 2))  # 升维
    return rowcol_index.view(-1, 2)


if __name__ == '__main__':
    print(f_get_rowcol_index(3, 4))
