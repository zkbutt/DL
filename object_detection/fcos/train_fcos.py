import os
import sys

from object_detection.fcos.CONFIG_FCOS import CFG
from object_detection.fcos.net.net_fcos import Fcos

'''用户命令行启动'''

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from f_tools.datas.data_loader import cfg_type3
from f_tools.fits.fitting.f_fit_class_base import Train_1gpu
from f_tools.GLOBAL_LOG import flog

from torch import optim
from torchvision import models

from f_pytorch.tools_model.f_layer_get import ModelOuts4Mobilenet_v2, ModelOuts4Resnet
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_gpu.f_gpu_api import model_device_init


def train_eval_set(cfg):
    # 基本不动
    cfg.TB_WRITER = True
    cfg.LOSS_EPOCH = False
    cfg.USE_MGPU_EVAL = True  # 一个有一个没得会卡死
    cfg.IS_MULTI_SCALE = False  # 关多尺度训练
    cfg.FILE_NAME_WEIGHT = '123' + '.pth'  # 重新开始

    # batch = 32  # type
    # batch = 16  # type
    batch = 2  # type
    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        batch *= 2
        cfg.IS_COCO_EVAL = False

    size = (320, 320)  # type
    cfg_type3(cfg, batch=batch, image_size=size)  # 加载数据基础参数

    cfg.NUMS_EVAL = {10: 10, 100: 3, 160: 2}
    # cfg.NUM_SAVE_INTERVAL = 100
    cfg.STRIDES = [8, 16, 32, 64]  # 特有参数下采样步距
    # 原装是(0,64,128,256,512,inf)
    cfg.SCALE_THRESHOLDS = [0, 49, 98, 196, 10000000000.0]  # 用于确保每一个特图预测相应大小的框,且一个GT只在一个层进行匹配
    cfg.USE_BASE4NP = True  # 特有参数下采样步距

    # type3 dark19
    # cfg.FILE_NAME_WEIGHT = 'zz/t_yolo2_type3_dark19c0.01-137_3.94_p73.5_r49.8' + '.pth'  # conf-0.01 nms-0.5
    # cfg.FILE_NAME_WEIGHT = 't_retina_type3-10_303.063' + '.pth'  # conf-0.01 nms-0.5
    # cfg.FILE_NAME_WEIGHT = 't_retina_type3-80_135.118' + '.pth'  # conf-0.01 nms-0.5
    # cfg.FILE_NAME_WEIGHT = 't_retina_type3-40_49.265' + '.pth'  # conf-0.01 nms-0.5
    cfg.MAPS_VAL = [0.58, 0.38]  # 最高

    cfg.LR0 = 1e-3
    cfg.TB_WRITER = True
    cfg.DEL_TB = True
    cfg.IS_FORCE_SAVE = False  # 强制记录


def init_model(cfg, device, id_gpu=None):
    model = models.resnet18(pretrained=True)
    dims_out = (128, 256, 512)
    model = ModelOuts4Resnet(model, dims_out)

    # cfg.NUMS_ANC = [9, 9, 9, 9, 9]
    # cfg.NUMS_ANC = [1, 1, 1, 1, 1]
    # cfg.FEATURE_MAP_STEPS = [8, 16, 32, 64, 128]
    # cfg.ANCS_SCALE = [[0.078, 0.07775],
    #                   [0.174, 0.164],
    #                   [0.324, 0.336],
    #                   [0.578, 0.466],
    #                   [0.698, 0.674]]
    model = Fcos(model, cfg, device)

    # cfg.NUMS_ANC = [3, 3, 3]
    # cfg.FEATURE_MAP_STEPS = [8, 16, 32]
    # model = Retina(model, cfg, device)

    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        for name, param in model.backbone.named_parameters():
            param.requires_grad_(False)

    model, is_mgpu = model_device_init(model, device, id_gpu, cfg)

    # ------------------------自定义backbone完成-------------------------------
    pg = model.parameters()
    optimizer = optim.Adam(pg, lr=cfg.LR0)
    # 两次不上升，降低一半
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [150, 200, 250], 0.1)
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
