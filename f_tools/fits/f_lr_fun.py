import torch


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    '''

    :param optimizer:
    :param warmup_iters: 迭代次数最大1000  warmup_iters = min(1000, len(data_loader) - 1)
    :param warmup_factor: 迭代值起始值  warmup_factor = 5.0 / 10000  # 0.0005
    :return: 学习率倍率 从 设定值 warmup_factor -> 1
    '''

    def f(x):
        """根据step数返回一个学习率倍率因子"""
        if x >= warmup_iters:  # 当迭代数大于给定的warmup_iters时，倍率因子为1
            return 1
        alpha = float(x) / warmup_iters  # 随每一步变大最大1
        # 迭代过程中倍率因子从 设定值 warmup_factor -> 1
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def update_lr(optimizer, lr):
    '''
        curr_lr=0.001
        if (epoch + 1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)

        lambda_G = lambda epoch : 0.5 ** (epoch // 30)
    :param optimizer:
    :param lr:
    :return:

    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    lambda_G = lambda epoch: 0.5 ** (epoch // 30)
    # 29表示从epoch = 30开始
    schduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer.parameters(), lambda_G, last_epoch=29)

    schduler_G = torch.optim.lr_scheduler.StepLR(optimizer.parameters(), step_size=30, gamma=0.1, last_epoch=29)
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer):  # 获取优化器lr
    for param_group in optimizer.param_groups:
        return param_group['lr']


def lr_example():
    # 在发现loss不再降低或者acc不再提高之后，降低学习率
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)


def op_example():
    # 权重衰减(如L2惩罚)(默认: 0)
    optimizer = optim.Adam(model.parameters(), 1e-3, weight_decay=5e-4)
