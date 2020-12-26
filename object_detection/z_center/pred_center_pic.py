import json
import os
import random

import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.fitting.f_fit_eval_base import f_prod_pic
# 这里要删除
from object_detection.z_center.CONFIG_CENTER import CFG
from object_detection.z_center.process_fun import cre_data_transform, init_model

if __name__ == '__main__':
    '''

    '''
    '''------------------系统配置---------------------'''
    cfg = CFG
    cfg.IMAGE_SIZE = [640, 640]
    cfg.NUM_CLASSES = 1
    # cfg.THRESHOLD_PREDICT_CONF = 0.4
    device = torch.device('cpu')
    flog.info('模型当前设备 %s', device)
    json_file = open(os.path.join(cfg.PATH_HOST, r'AI\datas\raccoon200\classes_ids_raccoon.json'), 'r', encoding='utf-8')
    ids_classes = json.load(json_file, encoding='utf-8')  # json key是字符
    labels_lsit = list(ids_classes.values())  # index 从 1开始 前面随便加一个空
    labels_lsit.insert(0, None)  # index 从 1开始 前面随便加一个空
    flog.debug('测试类型 %s', labels_lsit)

    '''------------------模型定义---------------------'''
    model, optimizer, lr_scheduler, start_epoch = init_model(cfg, device, id_gpu=None)
    model.eval()

    # path_img = cfg.PATH_DATA_ROOT + '/test/JPEGImages'
    path_img = cfg.PATH_HOST + r'\AI\datas\raccoon200\VOCdevkit\JPEGImages'
    # path_img = os.path.join(get_path_root(), '_test_pic')
    file_names = os.listdir(path_img)
    random.seed(20201215)
    random.shuffle(file_names)  # 随机打乱
    data_transform = cre_data_transform(cfg)

    for name in file_names:
        '''---------------数据加载及处理--------------'''
        file_img = os.path.join(path_img, name)
        # 这里需要修复
        f_prod_pic(file_img, model, labels_lsit, data_transform, is_keeep=cfg.IS_KEEP_SCALE, cfg=cfg)
    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
