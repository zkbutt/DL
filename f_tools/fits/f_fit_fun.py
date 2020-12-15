import math
import os

import cv2
import matplotlib
import torch
import numpy as np

from f_tools.GLOBAL_LOG import flog
import socket


def init_od():
    # -----------通用系统配置----------------
    torch.set_printoptions(linewidth=320, sci_mode=False, precision=5, profile='long')
    np.set_printoptions(linewidth=320, suppress=True,
                        formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
    matplotlib.rc('font', **{'size': 11})

    # 防止OpenCV进入多线程(使用PyTorch DataLoader)
    cv2.setNumThreads(0)


def custom_set(cfg):
    flog.info('当前主机: %s 及主数据路径: %s ' % (cfg.host_name, cfg.PATH_HOST))


def base_set(cfg):
    cfg.DATA_NUM_WORKERS = min([os.cpu_count(), cfg.DATA_NUM_WORKERS])

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(cfg.PATH_SAVE_WEIGHT):
        try:
            os.makedirs(cfg.PATH_SAVE_WEIGHT)
        except Exception as e:
            flog.error(' %s %s', cfg.PATH_SAVE_WEIGHT, e)

    # cfg.SAVE_FILE_NAME = os.path.basename(__file__)
    if torch.cuda.is_available():
        device = torch.device('cuda:%s' % 0)
    else:
        device = torch.device("cpu")
        cfg.IS_MIXTURE_FIX = False
    flog.info('模型当前设备 %s', device)

    if cfg.DEBUG:
        flog.warning('debug模式')
        # device = torch.device("cpu")
        cfg.PRINT_FREQ = 1
        # cfg.PATH_SAVE_WEIGHT = None
        cfg.BATCH_SIZE = 5
        cfg.DATA_NUM_WORKERS = 0
        pass
    else:
        torch.multiprocessing.set_sharing_strategy('file_system')  # 多进程开文件

    # if cfg.IS_MOSAIC:
    # cfg.IMAGE_SIZE = [512, 512]
    # cfg.ANC_SCALE = list(np.array(cfg.ANC_SCALE,dtype=np.float32) / 2)
    # pass

    # cfg.FILE_FIT_WEIGHT = None

    '''-----多尺度训练-----'''
    # if cfg.IS_MULTI_SCALE:
    #     # 动态输入尺寸选定 根据预设尺寸  0.667~1.5 之间 满足32的倍数
    #     imgsz_min = cfg.IMAGE_SIZE[0] // cfg.MULTI_SCALE_VAL[1]
    #     imgsz_max = cfg.IMAGE_SIZE[0] // cfg.MULTI_SCALE_VAL[0]
    #     # 将给定的最大，最小输入尺寸向下调整到32的整数倍
    #     grid_min, grid_max = imgsz_min // down_sample, imgsz_max // down_sample
    #     imgsz_min, imgsz_max = int(grid_min * down_sample), int(grid_max * down_sample)
    #     sizes_in = []
    #     for i in range(imgsz_min, imgsz_max + 1, down_sample):
    #         sizes_in.append(i)
    #     # imgsz_train = imgsz_max  # initialize with max size
    #     # img_size = random.randrange(grid_min, grid_max + 1) * gs
    #     flog.info("输入画像的尺寸范围为[{}, {}] 可选尺寸为{}".format(imgsz_min, imgsz_max, sizes_in))
    return device, cfg


if __name__ == '__main__':
    host_name = socket.gethostname()
    flog.info('当前主机 %s', host_name)
