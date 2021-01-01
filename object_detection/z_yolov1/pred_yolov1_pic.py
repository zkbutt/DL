import json
import os
import random

import torch
import numpy as np
from PIL import Image

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_loader import cre_transform4resize
from f_tools.f_general import get_path_root
from f_tools.fits.fitting.f_fit_eval_base import f_prod_pic
from f_tools.pic.f_show import f_plot_od4pil, f_show_od4pil
# 这里要删除
from object_detection.z_yolov1.CONFIG_YOLOV1 import CFG
from object_detection.z_yolov1.train_yolov1 import init_model, train_eval_set

if __name__ == '__main__':
    '''

    '''
    '''------------------系统配置---------------------'''
    cfg = CFG
    train_eval_set(cfg)

    device = torch.device('cpu')
    flog.info('模型当前设备 %s', device)
    # json_file = open(os.path.join(cfg.PATH_DATA_ROOT, 'ids_classes_voc_proj.json'), 'r', encoding='utf-8')
    # ids_classes = json.load(json_file, encoding='utf-8')  # json key是字符

    _ = cfg.FUN_LOADER_DATA(cfg, is_mgpu=False, )  # 用于加载配置
    
    ids_classes = cfg.IDS_CLASSES
    labels_lsit = list(ids_classes.values())  # index 从 1开始 前面随便加一个空
    labels_lsit.insert(0, None)  # index 从 1开始 前面随便加一个空
    flog.debug('测试类型 %s', labels_lsit)

    '''------------------模型定义---------------------'''
    model, optimizer, lr_scheduler, start_epoch = init_model(cfg, device, id_gpu=None)
    model.eval()

    path_img = cfg.PATH_IMG_TRAIN
    # path_img = os.path.join(get_path_root(), '_test_pic')
    file_names = os.listdir(path_img)
    random.seed(20201215)
    random.shuffle(file_names)  # 随机打乱
    data_transform = cre_transform4resize(cfg)

    for name in file_names:
        '''---------------数据加载及处理--------------'''
        file_img = os.path.join(path_img, name)
        f_prod_pic(file_img, model, labels_lsit, data_transform)
    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
