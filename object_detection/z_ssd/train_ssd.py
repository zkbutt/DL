import os
import sys

'''用户命令行启动'''
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from object_detection.z_ssd.CONFIG_SSD import CFG
from object_detection.z_ssd.nets.net_ssd import SSD, Backbone
from f_tools.fits.fitting.f_fit_class_base import Train_1gpu

from torch import optim
from f_tools.datas.data_loader import cfg_raccoon, cfg_type3, DataLoader2, cfg_type4, cfg_voc

from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_gpu.f_gpu_api import model_device_init, mgpu_process0_init

from f_tools.GLOBAL_LOG import flog
from torchvision import models

'''

linux用这个   python /AI/temp/tmp_pycharm/DL/object_detection/z_ssd/train_ssd.py
tensorboard --host=192.168.0.199 --logdir=/AI/temp/tmp_pycharm/DL/object_detection/z_ssd/runs_type3
'''


def train_eval_set(cfg):
    # 基本不动
    cfg.TB_WRITER = True
    cfg.LOSS_EPOCH = False
    cfg.USE_MGPU_EVAL = True  # 一个有一个没得会卡死
    cfg.IS_MULTI_SCALE = False  # 关多尺度训练
    cfg.FILE_NAME_WEIGHT = '123' + '.pth'  # 重新开始

    # batch = 64  # type
    batch = 32  # type
    # batch = 2  # type
    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        batch *= 2
        cfg.IS_COCO_EVAL = False

    size = (300, 300)  # 多尺寸时这个用于预测
    cfg_type3(cfg, batch=batch, image_size=size)  # 加载数据基础参数
    # cfg_type4(cfg, batch=batch, image_size=size)  # 加载数据基础参数
    # cfg_voc(cfg, batch=batch, image_size=size)  # 加载数据基础参数
    # cfg_raccoon(cfg, batch=batch, image_size=size)  # 加载数据基础参数

    cfg.FEATURE_SIZES = [38, 19, 10, 5, 3, 1]  # 共有30个anc
    cfg.NUMS_ANC = [4, 6, 6, 6, 4, 4]
    cfg.ANCS_SCALE = [[0.028, 0.034],
                      [0.028, 0.064],
                      [0.044, 0.05],
                      [0.062, 0.04],
                      [0.046, 0.072],
                      [0.038, 0.108],
                      [0.07, 0.078],
                      [0.106, 0.052],
                      [0.054, 0.158],
                      [0.084, 0.114],
                      [0.146, 0.092],
                      [0.078, 0.21],
                      [0.118, 0.154],
                      [0.15, 0.204],
                      [0.106, 0.294],
                      [0.242, 0.13],
                      [0.212, 0.234],
                      [0.162, 0.382],
                      [0.422, 0.198],
                      [0.298, 0.288],
                      [0.23, 0.412],
                      [0.47, 0.31],
                      [0.34, 0.444],
                      [0.274, 0.62187],
                      [0.792, 0.268],
                      [0.57806, 0.42],
                      [0.432, 0.6],
                      [0.843, 0.49],
                      [0.64, 0.732],
                      [0.942, 0.662], ]

    '''特有参数'''
    cfg.MODE_TRAIN = 1  # base
    # cfg.MODE_TRAIN = 2 # 去conf
    # cfg.MODE_TRAIN = 3 # 任意分布
    # cfg.MODE_TRAIN = 4  # IOU 损失及预测

    cfg.NUMS_EVAL = {10: 10, 100: 3, 160: 2}
    # cfg.NUMS_EVAL = {100: 3, 160: 2}
    # cfg.NUM_SAVE_INTERVAL = 100

    # cfg.KEEP_SIZE = True
    cfg.KEEP_SIZE = False

    # 稳定训练用这个
    # cfg.START_EVAL = 50  # 这个要 + EVAL_INTERVAL
    # cfg.EVAL_INTERVAL = 3  #

    cfg.NUM_ANC = 1

    # cfg.FILE_NAME_WEIGHT = 'zz/t_yolo1_type3_res18c0.01-110_4.47_p72.4_r46.2' + '.pth'  # conf-0.01 nms-0.5
    # cfg.FILE_NAME_WEIGHT = 'nvidia_ssdpyt_fp32_190826' + '.pt'  # conf-0.01 nms-0.5

    # cfg.FILE_NAME_WEIGHT = 't_ssd_type3c0.05-141_2.21_p64.7_r41.1' + '.pth'  # conf-0.01 nms-0.5
    cfg.MAPS_VAL = [0.6371928080996411, 0.40522891081043455]

    # cfg.LR0 = 1e-3/2
    cfg.LR0 = 0.0005
    cfg.TB_WRITER = True
    cfg.DEL_TB = False
    cfg.IS_FORCE_SAVE = False  # 强制记录


def init_model(cfg, device, id_gpu=None):
    # model = models.resnet50(pretrained=True)
    # model = ModelOut4Resnet50(model)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_res50'

    # model = darknet19(pretrained=True)
    # model = ModelOut4DarkNet19(model)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_d19'

    model = Backbone()
    model = SSD(backbone=model, cfg=cfg, device=device)
    # f_look_model(model, input=(1, 3, *cfg.IMAGE_SIZE))

    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        for name, param in model.backbone.named_parameters():
            param.requires_grad_(False)

    model, is_mgpu = model_device_init(model, device, id_gpu, cfg)

    # ------------------------自定义backbone完成-------------------------------
    pg = model.parameters()
    # optimizer = optim.Adam(pg, cfg.LR0)
    optimizer = optim.SGD(pg, lr=cfg.LR0, momentum=0.9, weight_decay=0.0005)
    # optimizer = optim.Adam(pg, cfg.LR0, weight_decay=5e-4)
    # optimizer = optim.SGD(pg, lr=cfg.LR0, momentum=0.937, weight_decay=5e-4, nesterov=True)
    # optimizer = optim.SGD(pg, lr=cfg.LR0, momentum=0.9, weight_decay=5e-4)

    # 两次不上升，降低一半
    lr_scheduler = None

    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 80, 100], 0.1)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30], 0.1)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=1, verbose=True)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 80, 100], 0.1)

    # start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, None, lr_scheduler, device, is_mgpu=is_mgpu)

    ''' 定制逻辑 '''
    def ffun(pretrained_dict):
        dd = {}
        s1 = 'net.backbone.'
        s2 = 'net.'
        for k, v in pretrained_dict.items():
            split_key = k.split(".")
            if 'additional_blocks' in split_key:
                dd[s2 + k] = v
            elif 'loc' in split_key:
                dd[s2 + k] = v
            elif 'feature_extractor' in split_key:
                k = '.'.join(split_key[1:])
                dd[s1 + k] = v

        return dd

    start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device, is_mgpu=is_mgpu,
                              ffun=None)
    # start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, None, None, device, is_mgpu=is_mgpu,
    #                           ffun=ffun)

    model.cfg = cfg
    return model, optimizer, lr_scheduler, start_epoch


if __name__ == '__main__':
    cfg = CFG
    '''这个要加/ 这里要改'''
    path_project_root = '/AI/temp/tmp_pycharm/DL/object_detection/z_ssd'
    # cfg.LR0 = 1e-3

    train = Train_1gpu(cfg, train_eval_set, init_model, path_project_root)
    train.f_run()

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
