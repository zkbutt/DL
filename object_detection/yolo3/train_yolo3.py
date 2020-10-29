import numpy as np

import os

import torch
from torch import optim

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_lossfun import LossOD_K, LossOD
from f_tools.fun_od.f_anc import Anchors
from object_detection.yolo3.utils.process_fun import init_model, data_loader, train_eval
from object_detection.yolo3.CONFIG_YOLO3 import CFG

'''
416x416x3  13x13x(20+1+4)
'''

if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    np.set_printoptions(suppress=True)  # 关闭科学计数
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
        CFG.DATA_NUM_WORKERS = 1
        pass
    else:
        torch.multiprocessing.set_sharing_strategy('file_system')  # 多进程开文件

    # CFG.FILE_FIT_WEIGHT = None

    '''------------------模型定义---------------------'''
    model = init_model(CFG)

    model.to(device)  # 这个不需要加等号
    model.train()

    anchors = Anchors(CFG.IMAGE_SIZE, CFG.ANCHORS_SIZE, CFG.FEATURE_MAP_STEPS, CFG.ANCHORS_CLIP).get_anchors()
    anchors = anchors.to(device)
    losser = LossOD(anchors, CFG.LOSS_WEIGHT, CFG.NEGATIVE_RATIO)
    # # 权重衰减(如L2惩罚)(默认: 0)
    optimizer = optim.Adam(model.parameters(), 1e-3)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # start_epoch = load_weight(CFG.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device)
    # # 权重初始模式
    # # state_dict = torch.load(CFG.FILE_FIT_WEIGHT, map_location=device)
    # # model_dict = model.state_dict()
    # # keys_missing, keys_unexpected = model.load_state_dict(state_dict)
    start_epoch = 0
    #
    # '''---------------数据加载及处理--------------'''
    loader_train, loader_val = data_loader(CFG, device)

    flog.debug('---训练开始---epoch %s', start_epoch + 1)
    train_eval(CFG, start_epoch, model, anchors, losser, optimizer, lr_scheduler,
               loader_train=loader_train, loader_val=loader_val, device=device,
               )

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))