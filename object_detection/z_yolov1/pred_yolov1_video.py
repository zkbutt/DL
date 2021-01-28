import os

import torch
from f_tools.GLOBAL_LOG import flog
from f_tools.fits.fitting.f_fit_class_base import Predicted_Video
from object_detection.z_yolov1.CONFIG_YOLOV1 import CFG
from object_detection.z_yolov1.train_yolov1 import train_eval_set, init_model

if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    cfg = CFG
    device = torch.device('cpu')
    flog.info('模型当前设备 %s', device)
    cfg.THRESHOLD_PREDICT_CONF = 0.4  # 用于预测的阀值
    cfg.THRESHOLD_PREDICT_NMS = 0.5  # 提高 conf 提高召回, 越小框越少

    predicted_video = Predicted_Video(cfg, train_eval_set, init_model, device)
    predicted_video.f_run()

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
