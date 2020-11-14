import math
import torch
from torch import optim
from torchvision import models


def f_cos_diminish(optimizer, start_epoch, epochs, lrf_scale):
    '''
    自定义调整策略
    :param optimizer:
    :param start_epoch:
    :param epochs: 总迭代次数
    :return:
    '''
    # cos渐减小学习率 余弦值首先缓慢下降吗然后加速下降, 再次缓慢下降 从  初始 ~ 0.1
    fun = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf_scale) + lrf_scale

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fun)
    scheduler.last_epoch = start_epoch  # 指定从哪个epoch开始
    return scheduler


def lr_example(optimizer):
    lr = 1e-3
    '''
    监控指标，当指标不再变化则调整 2次不降低则 LR变为原来的一半
        • mode：min（对应损失值）/max（对应精确度） 两种模式
        • factor：调整系数（相当于之前的lamda）
        • patience：“耐心”，接受几次不变化
        • cooldown：“冷却时间”，停止监控一段时间
        • verbose：是否打印日志
        • min_lr：学习率下限（到达下限就不再监视调整了）
        • eps：学习率衰减最小值
    '''
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    '''
    步按比例gamma  等间隔调整学习率
    0<epoch<30, lr = 0.05
    30<=epoch<60, lr = 0.005
    60<=epoch<90, lr = 0.0005
    '''
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    '''
    指定区间按比例下降 按给定间隔调整学习率
    lr = 0.05     if epoch < 30
    lr = 0.005    if 30 <= epoch < 80
    lr = 0.0005   if epoch >= 80
    '''
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 80], 0.1)

    # 按指数减 gamma：指数的底
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    '''
    余弦周期调整学习率 30轮后降为0
    T_max：下降周期，就是学习率从最大下降到最小经过的epoch数
    eta_min：学习率下限
    '''
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0, )

    return scheduler


def op_example():
    lr0 = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr0, weight_decay=5e-4)  # 权重衰减(如L2惩罚)(默认: 0)
    optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.937, weight_decay=0.0005, nesterov=True)
    optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.9, weight_decay=0.005)


def f_show_scheduler(scheduler, epochs):
    import matplotlib.pyplot as plt
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LambdaLR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    model = models.resnext50_32x4d(pretrained=True)
    lr0 = 1e-3
    optimizer = optim.Adam(model.parameters(), lr0)
    lrf_scale = 0.01
    start_epoch = 0
    epochs = 10

    scheduler = f_cos_diminish(optimizer, start_epoch, epochs, lrf_scale)
    # scheduler = lr_example(optimizer)
    f_show_scheduler(scheduler, 100)
    pass
