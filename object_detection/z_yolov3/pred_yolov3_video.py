import json
import os

import torch
import cv2
import time

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.f_coco.convert_data.coco_dataset import CustomCocoDataset4cv
from f_tools.device.f_device import init_video
from f_tools.fits.fitting.f_fit_class_base import Predicted_Pic
from f_tools.fits.fitting.f_fit_eval_base import f_prod_vodeo
from f_tools.pic.enhance.f_data_pretreatment4np import cre_transform_resize4np
from object_detection.z_center.train_center import train_eval_set, init_model
from object_detection.z_yolov3.CONFIG_YOLO3 import CFG

if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    cfg = CFG

    pred_pic = Predicted_Pic(cfg,train_eval_set,init_model)

    # 这里是原图
    dataset_test = CustomCocoDataset4cv(
        file_json=cfg.FILE_JSON_TEST,
        path_img=cfg.PATH_IMG_EVAL,
        mode=cfg.MODE_COCO_EVAL,
        transform=None,
        is_mosaic=False,
        is_mosaic_keep_wh=False,
        is_mosaic_fill=False,
        is_debug=cfg.DEBUG,
        cfg=cfg
    )

    data_transform = cre_transform_resize4np(cfg)['val']
    ids_classes = dataset_test.ids_classes
    labels_lsit = list(ids_classes.values())  # index 从 1开始 前面随便加一个空
    labels_lsit.insert(0, None)  # index 从 1开始 前面随便加一个空
    flog.debug('测试类型 %s', labels_lsit)

    device = torch.device('cpu')
    flog.info('模型当前设备 %s', device)

    '''------------------模型定义---------------------'''
    model, _, _, _ = init_model(cfg, device, id_gpu=None)
    model.eval()

    # 调用摄像头
    cap = init_video()

    '''---------------预测开始--------------'''

    # 使用视频预测
    f_prod_vodeo(cap, data_transform, model, labels_lsit, is_keeep=cfg.IS_KEEP_SCALE)

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
