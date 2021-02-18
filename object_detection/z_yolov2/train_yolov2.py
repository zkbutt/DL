import os
import sys

from torchvision import models

'''用户命令行启动'''
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from f_tools.fits.fitting.f_fit_class_base import Train_1gpu
from f_pytorch.tools_model.f_layer_get import ModelOuts4DarkNet19, ModelOuts4Resnet
from object_detection.z_yolov2.nets.net_yolov2 import Yolo_v2
from object_detection.z_yolov2.CONFIG_YOLOV2 import CFG
from f_pytorch.tools_model.backbones.darknet import darknet19
from torch import optim
from f_tools.datas.data_loader import cfg_type2, DataLoader2

from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_gpu.f_gpu_api import model_device_init, mgpu_process0_init

from f_tools.GLOBAL_LOG import flog

'''

linux用这个   python /AI/temp/tmp_pycharm/DL/object_detection/z_yolov2/train_yolov2.py
tensorboard --host=192.168.0.199 --logdir=/AI/temp/tmp_pycharm/DL/object_detection/z_yolov2/runs_type3

100轮 ap70_45
与yolo1一致

'''


def train_eval_set(cfg):
    # 基本不动
    cfg.TB_WRITER = True
    cfg.LOSS_EPOCH = False
    cfg.USE_MGPU_EVAL = True  # 一个有一个没得会卡死
    cfg.IS_MULTI_SCALE = False  # 关多尺度训练
    cfg.FILE_NAME_WEIGHT = '123' + '.pth'  # 重新开始

    batch = 16  # type
    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        batch *= 2
        cfg.IS_COCO_EVAL = False

    size = (416, 416)  # type
    cfg_type2(cfg, batch=batch, image_size=size)  # 加载数据基础参数
    # cfg_raccoon(cfg, batch=batch, image_size=size)  # 加载数据基础参数

    # anc重写
    cfg.ANCS_SCALE = [[0.074, 0.074], [0.162, 0.146], [0.314, 0.3], [0.452, 0.506], [0.729, 0.635]]
    cfg.NUM_ANC = len(cfg.ANCS_SCALE)

    cfg.NUMS_EVAL = {10: 5, 100: 3, 160: 2}
    # cfg.NUM_SAVE_INTERVAL = 100

    cfg.arg_focalloss_alpha = 0.75
    cfg.IS_MIXTURE_FIX = True

    # type3 dark19
    # cfg.FILE_NAME_WEIGHT = 'zz/t_yolo2_type3_dark19c0.01-137_3.94_p73.5_r49.8' + '.pth'  # conf-0.01 nms-0.5
    # cfg.FILE_NAME_WEIGHT = 't_yolo2_type3_res18-20_6.725' + '.pth'  # conf-0.01 nms-0.5
    cfg.MAPS_VAL = [0.71, 0.46]  # 最高

    cfg.LR0 = 1e-3 / 2
    cfg.TB_WRITER = True
    cfg.DEL_TB = True
    cfg.IS_FORCE_SAVE = False  # 强制记录
    cfg.KEEP_SIZE = False


def init_model(cfg, device, id_gpu=None):
    # model = darknet19(pretrained=True)
    # model = ModelOuts4DarkNet19(model)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_dark19'

    # model = models.resnet18(pretrained=True)
    # dims_out = (128, 256, 512)
    # model = ModelOuts4Resnet(model, dims_out)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_res18'

    model = models.resnet50(pretrained=True)
    dims_out = (512, 1024, 2048)
    model = ModelOuts4Resnet(model, dims_out)
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_res50'

    model = Yolo_v2(backbone=model, cfg=cfg)
    # f_look_model(model, input=(1, 3, *cfg.IMAGE_SIZE))

    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        for name, param in model.backbone.named_parameters():
            param.requires_grad_(False)

    model, is_mgpu = model_device_init(model, device, id_gpu, cfg)

    # ------------------------自定义backbone完成-------------------------------
    pg = model.parameters()
    optimizer = optim.Adam(pg, cfg.LR0)
    # 两次不上升，降低一半
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 80, 120], 0.1)
    start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device, is_mgpu=is_mgpu)

    model.cfg = cfg
    return model, optimizer, lr_scheduler, start_epoch


if __name__ == '__main__':
    cfg = CFG
    path_project_root = '/AI/temp/tmp_pycharm/DL/object_detection/z_yolov2'
    # cfg.LR0 = 1e-3

    train = Train_1gpu(cfg, train_eval_set, init_model, path_project_root)
    train.f_run()

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
