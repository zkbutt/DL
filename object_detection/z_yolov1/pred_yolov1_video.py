import json
import os
import sys

import torch
import cv2
import time

from PIL import Image
import numpy as np
from f_tools.GLOBAL_LOG import flog
from f_tools.pic.f_show import f_plot_od4pil
from object_detection.z_yolov1.CONFIG_YOLOV1 import CFG
from object_detection.z_yolov1.process_fun import init_model, cre_data_transform

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
    cap = cv2.VideoCapture(0)  # capture=cv2.VideoCapture("1.mp4")
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    '''------------------模型定义---------------------'''
    model, optimizer, lr_scheduler, start_epoch = init_model(cfg, device, id_gpu=None)
    model.eval()

    '''---------------预测开始--------------'''
    fps = 0.0
    count = 0
    while True:
        start_time = time.time()
        '''---------------数据加载及处理--------------'''
        ref, img_np = cap.read()  # 读取某一帧 ref是否成功
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # 格式转变，BGRtoRGB
        img_pil = Image.fromarray(img_np, mode="RGB")

        w, h = img_pil.size
        szie_scale4bbox = torch.Tensor([w, h] * 2)
        # szie_scale4landmarks = torch.Tensor([w, h] * 5)

        data_transform = cre_data_transform(cfg)
        img_ts = data_transform['val'](img_pil)[0][None]

        '''---------------预测开始--------------'''
        ids_batch, p_boxes_ltrb, p_labels, p_scores = model(img_ts)
        if p_boxes_ltrb is not None:
            p_boxes = p_boxes_ltrb * szie_scale4bbox
            img_pil = f_plot_od4pil(img_pil, p_boxes, p_scores, p_labels, labels_lsit)

        img_np = np.array(img_pil)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        # print("fps= %.2f" % (fps))
        count += 1
        img_np = cv2.putText(img_np, "fps= %.2f count=%s" % (fps, count), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                             (0, 255, 0), 2)
        fps = (fps + (1. / max(sys.float_info.min, time.time() - start_time))) / 2
        cv2.imshow("video", img_np)

        c = cv2.waitKey(1) & 0xff  # 输入esc退出
        if c == 27:
            cap.release()
            break
