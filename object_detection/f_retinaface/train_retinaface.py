import numpy as np

import os

import torch
from torch import optim

from f_tools.GLOBAL_LOG import flog
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_lossfun import LossOD_K
from f_tools.fun_od.f_anc import AnchorsFound
from object_detection.f_retinaface.CONFIG_F_RETINAFACE import CFG
from object_detection.f_retinaface.utils.process_fun import init_model, data_loader, train_eval

'''/home/win10_sys/AI/weights/feadre/train_retinaface.py-40_6.929875373840332.pth
eta: 0:00:00.730305  lr: 0.000500  loss_total: 6.8880 (7.2228)  loss_bboxs: 1.0246 (1.0967)  loss_labels: 1.5412 (1.5934)  loss_keypoints: 2.1876 (2.3392)  time: 0.5443  data: 0.0541  剩余时间: 0  max mem: 3893

/home/win10_sys/AI/weights/feadre/train_retinaface.py-10_7.423844337463379.pth
lr: 0.000250  loss_total: 6.9989 (7.4330)  loss_bboxs: 1.0570 (1.0892)  loss_labels: 1.4724 (1.5348)  loss_keypoints: 2.5386 (2.6307)  time: 0.6397  data: 0.1441  剩余时间: 0  max mem: 389

6个图片，显存6125 time: 0.5513
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

    anchors = AnchorsFound(CFG.IMAGE_SIZE, CFG.ANCHORS_SIZE, CFG.FEATURE_MAP_STEPS, CFG.ANCHORS_CLIP).get_anchors()
    anchors = anchors.to(device)

    losser = LossOD_K(anchors, CFG.LOSS_WEIGHT, CFG.NEGATIVE_RATIO)
    # 权重衰减(如L2惩罚)(默认: 0)
    optimizer = optim.Adam(model.parameters(), 1e-3, weight_decay=5e-4)
    # 在发现loss不再降低或者acc不再提高之后，降低学习率
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    start_epoch = load_weight(CFG.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device)
    # 权重初始模式
    # state_dict = torch.load(CFG.FILE_FIT_WEIGHT, map_location=device)
    # model_dict = model.state_dict()
    # keys_missing, keys_unexpected = model.load_state_dict(state_dict)
    # start_epoch = 0

    '''---------------数据加载及处理--------------'''
    loader_train, loader_val = data_loader(CFG, device)

    flog.debug('---训练开始---epoch %s', start_epoch + 1)
    train_eval(CFG, start_epoch, model, anchors, losser, optimizer, lr_scheduler,
               loader_train=loader_train, loader_val=loader_val, device=device,
               )

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
