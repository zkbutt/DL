import json
import os

import torch
from f_tools.GLOBAL_LOG import flog
from f_tools.device.f_device import init_video
from f_tools.fits.fitting.f_fit_eval_base import f_prod_vodeo
from object_detection.z_yolov1.CONFIG_YOLOV1 import CFG

if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    cfg = CFG
    json_file = open(os.path.join(cfg.PATH_DATA_ROOT, 'ids_classes_voc_proj.json'), 'r', encoding='utf-8')
    ids_classes = json.load(json_file, encoding='utf-8')  # json key是字符
    labels_lsit = list(ids_classes.values())  # index 从 1开始 前面随便加一个空
    labels_lsit.insert(0, None)  # index 从 1开始 前面随便加一个空
    flog.debug('测试类型 %s', labels_lsit)

    device = torch.device('cpu')
    flog.info('模型当前设备 %s', device)

    # 调用摄像头
    cap = init_video()
    '''------------------模型定义---------------------'''
    model, optimizer, lr_scheduler, start_epoch = init_model(cfg, device, id_gpu=None)
    model.eval()

    '''---------------预测开始--------------'''
    data_transform = cre_data_transform(cfg)

    # 使用视频预测
    f_prod_vodeo(cap, data_transform, model, labels_lsit)

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
