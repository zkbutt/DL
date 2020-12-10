import json
import os

import torch
import numpy as np
from PIL import Image

from f_tools.GLOBAL_LOG import flog
from f_tools.f_general import get_path_root
from f_tools.pic.f_show import f_plot_od4pil, f_show_od4pil
from object_detection.z_yolov1.CONFIG_YOLOV1 import CFG
from object_detection.z_yolov1.process_fun import init_model, cre_data_transform

if __name__ == '__main__':
    '''

    '''
    '''------------------系统配置---------------------'''
    cfg = CFG
    device = torch.device('cpu')
    flog.info('模型当前设备 %s', device)
    json_file = open(os.path.join(cfg.PATH_DATA_ROOT, 'ids_classes_voc_proj.json'), 'r', encoding='utf-8')
    ids_classes = json.load(json_file, encoding='utf-8')  # json key是字符
    labels_lsit = list(ids_classes.values())  # index 从 1开始 前面随便加一个空
    labels_lsit.insert(0, None)  # index 从 1开始 前面随便加一个空
    flog.debug('测试类型 %s', labels_lsit)

    '''------------------模型定义---------------------'''
    model, optimizer, lr_scheduler, start_epoch = init_model(cfg, device, id_gpu=None)
    model.eval()

    path_img = os.path.join(get_path_root(), '_test_pic')
    files = os.listdir(path_img)
    for file in files:
        '''---------------数据加载及处理--------------'''
        img_pil = Image.open(os.path.join(path_img, file)).convert('RGB')
        w, h = img_pil.size
        # 用于恢复bbox及ke
        szie_scale4bbox = torch.Tensor([w, h] * 2)
        # szie_scale4landmarks = torch.Tensor([w, h] * 5)
        data_transform = cre_data_transform(cfg)
        img_ts = data_transform['val'](img_pil)[0][None]

        '''---------------预测开始--------------'''
        ids_batch, p_boxes_ltrb, p_labels, p_scores = model(img_ts)
        if p_boxes_ltrb is not None:
            p_boxes = p_boxes_ltrb * szie_scale4bbox
            img_pil = f_plot_od4pil(img_pil, p_boxes, p_scores, p_labels, labels_lsit)
            f_show_od4pil(img_pil, p_boxes, p_scores, p_labels, labels_lsit)
    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
