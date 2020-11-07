import math
import numpy as np

import os

import torch
from torch import optim

from f_tools.GLOBAL_LOG import flog
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_lossfun import LossOD_K, LossYOLO, LossYOLOv1
from f_tools.fun_od.f_anc import Anchors
from object_detection.f_yolov1.CONFIG_YOLO1 import CFG
from object_detection.f_yolov1.process_fun import init_model, data_loader, train_eval

'''
txt格式：[图片名字 目标个数 左上角坐标x 左上角坐标y 右下角坐标x 右下角坐标y 类别]


416x416x3  13x13x(20+1+4)
c,x,y,w,h
    直接将每一项进行MSE求误差
        值域不一致需要权重
    只对有目标的计算损失
'''

if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    # np.set_printoptions(suppress=True)  # 关闭科学计数
    # torch.set_printoptions(linewidth=320, precision=5, profile='long')
    # np.set_printoptions(suppress=True, linewidth=320,
    #                     formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
    # matplotlib.rc('font', **{'size': 11})

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(CFG.PATH_SAVE_WEIGHT):
        try:
            os.makedirs(CFG.PATH_SAVE_WEIGHT)
        except Exception as e:
            flog.error(' %s %s', CFG.PATH_SAVE_WEIGHT, e)
    CFG.SAVE_FILE_NAME = os.path.basename(__file__)

    device = torch.device('cuda:%s' % 0 if torch.cuda.is_available() else "cpu")
    flog.info('模型当前设备 %s', device)

    if CFG.DEBUG:
        # device = torch.device("cpu")
        CFG.PRINT_FREQ = 1
        CFG.PATH_SAVE_WEIGHT = None
        CFG.BATCH_SIZE = 5
        CFG.DATA_NUM_WORKERS = 0
        pass
    else:
        torch.multiprocessing.set_sharing_strategy('file_system')  # 多进程开文件

    '''------------------模型定义---------------------'''
    model = init_model(CFG)

    model.to(device)  # 这个不需要加等号
    model.train()

    losser = LossYOLOv1(
        grid=CFG.GRID,
        l_noobj=CFG.L_NOOBJ,
        l_coord=CFG.L_COORD,
        num_cls=CFG.NUM_CLASSES
    )

    # 最初学习率
    lr0 = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr0, weight_decay=5e-4)  # 权重衰减(如L2惩罚)(默认: 0)
    # optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.937, weight_decay=0.0005, nesterov=True)

    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    # start_epoch = load_weight(CFG.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device)
    # # 权重初始模式
    # # state_dict = torch.load(CFG.FILE_FIT_WEIGHT, map_location=device)
    # # model_dict = model.state_dict()
    # # keys_missing, keys_unexpected = model.load_state_dict(state_dict)
    start_epoch = 0

    # '''---------------数据加载及处理--------------'''
    loader_train, loader_val = data_loader(CFG, device)

    flog.debug('---训练开始---epoch %s', start_epoch + 1)
    # v1不需要anchor
    train_eval(CFG, start_epoch, model, losser, optimizer, lr_scheduler,
               loader_train=loader_train, loader_val=loader_val, device=device,
               )

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
