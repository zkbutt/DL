import os
import sys

'''用户命令行启动'''
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
# sys.path.insert(0, '.') # 这个能否使用待测试
from object_detection.z_retina.nets.net_retina import Retina2, Retina3
from f_tools.datas.data_loader import cfg_type3, cfg_voc, cfg_type4
from f_tools.fits.fitting.f_fit_class_base import Train_1gpu
from f_tools.GLOBAL_LOG import flog

from torch import optim
from torchvision import models

from f_pytorch.tools_model.f_layer_get import ModelOuts4Mobilenet_v2, ModelOuts4Resnet
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_gpu.f_gpu_api import model_device_init

from object_detection.z_retina.CONFIG_RETINAFACE import CFG

'''
tensorboard --host=192.168.0.199 --logdir=/AI/temp/tmp_pycharm/DL/object_detection/z_retina/runs_voc
正反例  3614：1(未减正例)

'''


def train_eval_set(cfg):
    # 基本不动
    cfg.TB_WRITER = True
    cfg.LOSS_EPOCH = False
    cfg.USE_MGPU_EVAL = True  # 一个有一个没得会卡死
    cfg.IS_MULTI_SCALE = False  # 关多尺度训练
    cfg.FILE_NAME_WEIGHT = '123' + '.pth'  # 重新开始

    batch = 64  # type
    # batch = 16  # type
    # batch = 5  # type
    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        batch *= 2
        cfg.IS_COCO_EVAL = False

    size = (416, 416)  # type
    # cfg_type3(cfg, batch=batch, image_size=size)  # 加载数据基础参数
    cfg_type4(cfg, batch=batch, image_size=size)  # 加载数据基础参数
    # cfg_voc(cfg, batch=batch, image_size=size)  # 加载数据基础参数

    cfg.NUMS_EVAL = {10: 5, 100: 3, 150: 1}

    '''特有参数'''
    cfg.MODE_TRAIN = 1  # base  带conf 使用3倍难例挖掘
    # cfg.MODE_TRAIN = 2  # 不带conf
    # cfg.MODE_TRAIN = 3  # 统一归一化计算
    cfg.NUM_REG = 1  # 这个是必须
    cfg.KEEP_SIZE = False  # 有anc建议用这个
    cfg.variances = (0.1, 0.2)

    # type3 dark19
    # cfg.FILE_NAME_WEIGHT = 'zz/t_yolo2_type3_dark19c0.01-137_3.94_p73.5_r49.8' + '.pth'  # conf-0.01 nms-0.5
    # cfg.FILE_NAME_WEIGHT = 't_retina_type3-10_366.704' + '.pth'  # conf-0.01 nms-0.5
    # cfg.FILE_NAME_WEIGHT = 't_retina_type3_res50-10_6.695' + '.pth'
    cfg.MAPS_VAL = [0.672, 0.48095764814812036]  # 最高

    # cfg.LR0 = 1e-3 / 2
    cfg.LR0 = 0.0005
    cfg.TB_WRITER = True
    # cfg.DEL_TB = True 已弃用
    cfg.IS_FORCE_SAVE = False  # 强制记录


def init_model(cfg, device, id_gpu=None):
    # model = models.densenet121(pretrained=True)
    # ret_name_dict = {'denseblock2': 1, 'denseblock3': 2, 'denseblock4': 3}
    # dims_out = [512, 1024, 1024]
    # model = ModelOuts4Densenet121(model, 'features', ret_name_dict, dims_out)
    # cfg.FEATURE_MAP_STEPS = [8, 16, 32]  # 特图的步距 下采倍数
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'densenet121'

    # model = models.resnet50(pretrained=True)
    # dims_out = (512, 1024, 2048)
    # model = ModelOuts4Resnet(model, dims_out)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_res50'
    # cfg.FEATURE_MAP_STEPS = [8, 16, 32]  # 特图的步距 下采倍数

    model = models.resnet18(pretrained=True)
    dims_out = (128, 256, 512)
    model = ModelOuts4Resnet(model, dims_out)

    cfg.FEATURE_MAP_STEPS = [8, 16, 32, 64, 128]

    # 每层特图5个anc
    cfg.NUMS_ANC = [1, 1, 1, 1, 1]
    cfg.ANCS_SCALE = [[0.078, 0.07775],
                      [0.174, 0.164],
                      [0.324, 0.336],
                      [0.578, 0.466],
                      [0.698, 0.674]]

    # cfg.NUMS_ANC = [3, 3, 3, 3, 3]
    # cfg.ANCS_SCALE = [[0.052, 0.046],
    #                   [0.088, 0.092],
    #                   [0.152, 0.106],
    #                   [0.184, 0.186],
    #                   [0.224, 0.32],
    #                   [0.38, 0.242],
    #                   [0.338, 0.424],
    #                   [0.572, 0.32],
    #                   [0.34, 0.624],
    #                   [0.484, 0.527],
    #                   [0.7, 0.44],
    #                   [0.642, 0.656],
    #                   [0.836, 0.573],
    #                   [0.61, 0.882],
    #                   [0.92, 0.74231]]
    # cfg.LOSS_WEIGHT = [1, 1, 1, 1, 1]  # conf_pos conf_neg cls loss_txty  loss_twth
    model = Retina2(model, cfg, device)
    # model = Retina3(model, cfg, device)

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
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 70, 100], 0.1)
    start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device, is_mgpu=is_mgpu)

    model.cfg = cfg
    return model, optimizer, lr_scheduler, start_epoch


if __name__ == '__main__':
    cfg = CFG
    path_project_root = '/AI/temp/tmp_pycharm/DL/object_detection/z_retina'
    # cfg.LR0 = 1e-3

    train = Train_1gpu(cfg, train_eval_set, init_model, path_project_root)
    train.f_run()

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
