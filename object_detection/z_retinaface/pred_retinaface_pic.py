import json
import os

import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_loader import cre_transform4resize
from f_tools.fits.fitting.f_fit_eval_base import f_prod_pic4keypoints, f_prod_pic
from object_detection.z_retinaface.CONFIG_RETINAFACE import CFG
from object_detection.z_retinaface.train_retinaface import init_model, train_eval_set

if __name__ == '__main__':
    '''

    '''
    '''------------------系统配置---------------------'''
    cfg = CFG
    train_eval_set(cfg)

    device = torch.device('cpu')
    flog.info('模型当前设备 %s', device)

    # 这里是原图
    dataset_val = cfg.FUN_EVAL_DATA(cfg)
    ids_classes = dataset_val.ids_classes
    labels_lsit = list(ids_classes.values())  # index 从 1开始 前面随便加一个空
    labels_lsit.insert(0, None)  # index 从 1开始 前面随便加一个空
    flog.debug('测试类型 %s', labels_lsit)

    '''------------------模型定义---------------------'''
    model, optimizer, lr_scheduler, start_epoch = init_model(cfg, device, id_gpu=None)
    model.eval()

    data_transform = cre_transform4resize(cfg)

    # 这里是原图
    for img, _ in dataset_val:
        f_prod_pic(img, model, labels_lsit, data_transform)

    # for name in file_names:
    #     '''---------------数据加载及处理--------------'''
    #     file_img = os.path.join(path_img, name)
    #     f_prod_pic4keypoints(file_img, model, labels_lsit, data_transform)
    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
