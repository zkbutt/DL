import os

import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.fitting.f_fit_class_base import Predicted_Pic
from object_detection.z_yolov2.CONFIG_YOLOV2 import CFG
from object_detection.z_yolov2.train_yolov2 import train_eval_set, init_model

if __name__ == '__main__':
    '''

    '''
    '''------------------系统配置---------------------'''
    cfg = CFG
    device = torch.device('cpu')
    flog.info('模型当前设备 %s', device)
    cfg.THRESHOLD_PREDICT_CONF = 0.4  # 用于预测的阀值
    cfg.THRESHOLD_PREDICT_NMS = 0.5  # 提高 conf 提高召回, 越小框越少
    eval_start = 20
    is_test_dir = True  # 测试dataset 或目录
    path_img = r'D:\tb\tb\ai_code\DL\_test_pic\dog_cat_bird'

    predicted_pic = Predicted_Pic(cfg, train_eval_set, init_model, device,
                                  eval_start=eval_start,
                                  is_test_dir=is_test_dir,
                                  path_img=path_img)

    predicted_pic.f_run()

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
