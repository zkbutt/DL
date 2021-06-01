import os
import sys

from object_detection.fcos.CONFIG_FCOS import CFG
from object_detection.fcos.net.net_fcos import Fcos

'''用户命令行启动'''

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from f_tools.datas.data_loader import cfg_type3, cfg_type4
from f_tools.fits.fitting.f_fit_class_base import Train_1gpu
from f_tools.GLOBAL_LOG import flog

from torch import optim
from torchvision import models

from f_pytorch.tools_model.f_layer_get import ModelOuts4Mobilenet_v2, ModelOuts4Resnet
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_gpu.f_gpu_api import model_device_init

'''
tensorboard --host=192.168.0.199 --logdir=/AI/temp/tmp_pycharm/DL/object_detection/fcos/log/runs_type3/2021-05-23_12_52_11

'''


def train_eval_set(cfg):
    # 基本不动
    cfg.TB_WRITER = True
    cfg.LOSS_EPOCH = False
    cfg.USE_MGPU_EVAL = True  # 一个有一个没得会卡死
    cfg.IS_MULTI_SCALE = False  # 关多尺度训练
    cfg.FILE_NAME_WEIGHT = '123' + '.pth'  # 重新开始

    # batch = 8  # type
    batch = 16  # type
    # batch = 2  # type
    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        batch *= 2
        cfg.IS_COCO_EVAL = False

    # cfg.MODE_TRAIN = 1  # base  r18+fpn4
    # size = (320, 320)  # type

    cfg.MODE_TRAIN = 2  # 论文标准实现
    size = (512, 512)  # size 和 cfg.STRIDES 必须成倍
    cfg.USE_BASE4NP = True  # 这个用于测试

    cfg_type3(cfg, batch=batch, image_size=size)  # 加载数据基础参数
    # cfg_type4(cfg, batch=batch, image_size=size)  # 加载数据基础参数

    cfg.NUMS_EVAL = {10: 10, 100: 3, 160: 2}
    # cfg.NUM_SAVE_INTERVAL = 100

    # 原装是416 输出为5层 (0,64,128,256,512,inf)

    # type3 dark19
    # cfg.FILE_NAME_WEIGHT = 'zz/t_yolo2_type3_dark19c0.01-137_3.94_p73.5_r49.8' + '.pth'  # conf-0.01 nms-0.5
    # cfg.FILE_NAME_WEIGHT = 't_fcos_type3_res50-10_1.763' + '.pth'  # conf-0.01 nms-0.5
    cfg.MAPS_VAL = [0.74, 0.56]  # 最高

    cfg.LR0 = 1e-3
    cfg.TB_WRITER = True
    # cfg.DEL_TB = True 已弃用
    cfg.IS_FORCE_SAVE = False  # 强制记录


def init_model(cfg, device, id_gpu=None):
    if cfg.MODE_TRAIN == 1:
        model = models.resnet18(pretrained=True)
        dims_out = (128, 256, 512)
        model = ModelOuts4Resnet(model, dims_out)
        cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_res18'
        cfg.STRIDES = [8, 16, 32, 64]  # 特有参数下采样步距
        cfg.SCALE_THRESHOLDS = [0, 49, 98, 196, 10000000000.0]  # 用于确保每一个特图预测相应大小的框,且一个GT只在一个层进行匹配
    elif cfg.MODE_TRAIN == 2:
        model = models.resnet50(pretrained=True)
        dims_out = (512, 1024, 2048)
        model = ModelOuts4Resnet(model, dims_out)
        cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_res50'
        cfg.STRIDES = [8, 16, 32, 64, 128]  # 待定
        cfg.SCALE_THRESHOLDS = [0, 64, 128, 256, 512, 10000000000.0]
    else:
        raise Exception('cfg.MODE_TRAIN 出错 cfg.MODE_TRAIN=%s' % (cfg.MODE_TRAIN))

    model = Fcos(model, cfg, device)

    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        for name, param in model.backbone.named_parameters():
            param.requires_grad_(False)

    model, is_mgpu = model_device_init(model, device, id_gpu, cfg)

    # ------------------------自定义backbone完成-------------------------------
    pg = model.parameters()
    optimizer = optim.Adam(pg, lr=cfg.LR0)
    # 两次不上升，降低一半
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 80, 100], 0.1)
    start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device, is_mgpu=is_mgpu)

    model.cfg = cfg
    return model, optimizer, lr_scheduler, start_epoch


if __name__ == '__main__':
    cfg = CFG
    path_project_root = '/AI/temp/tmp_pycharm/DL/object_detection/fcos'
    # cfg.LR0 = 1e-3

    train = Train_1gpu(cfg, train_eval_set, init_model, path_project_root)
    train.f_run()

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
