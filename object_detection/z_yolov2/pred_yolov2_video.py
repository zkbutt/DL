import os

import torch

from f_tools.GLOBAL_LOG import flog
from object_detection.z_yolov2.CONFIG_YOLOV2 import CFG

if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    cfg = CFG
    device = torch.device('cpu')
    flog.info('模型当前设备 %s', device)
    cfg.THRESHOLD_PREDICT_CONF = 0.4  # 用于预测的阀值
    cfg.THRESHOLD_PREDICT_NMS = 0.5  # 提高 conf 提高召回, 越小框越少

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
