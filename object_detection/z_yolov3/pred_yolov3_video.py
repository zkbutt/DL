import os

from f_tools.GLOBAL_LOG import flog
from object_detection.z_yolov3.CONFIG_YOLO3 import CFG

if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    cfg = CFG

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
