'''用户命令行启动'''
import os
import sys

from object_detection.z_yolov3.nets.Layer_yolo3 import CSPDarknet_slim

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from torchvision import models
from object_detection.z_yolov3.nets.net_yolov3 import Yolo_v3
from f_pytorch.tools_model.backbones.darknet import darknet53, darknet19
from f_tools.datas.data_loader import cfg_type3, cfg_type4
from f_tools.fits.fitting.f_fit_class_base import Train_1gpu
from torch import optim
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_gpu.f_gpu_api import model_device_init
from object_detection.z_yolov3.CONFIG_YOLO3 import CFG

from f_tools.GLOBAL_LOG import flog
from f_pytorch.tools_model.f_layer_get import ModelOuts4DarkNet53, ModelOuts4DarkNet19, ModelOuts4Resnet

'''
python /AI/temp/tmp_pycharm/DL/object_detection/z_yolov3/train_yolo3.py
tensorboard --host=192.168.0.199 --logdir=/AI/temp/tmp_pycharm/DL/object_detection/z_yolov3/runs_type3

'''


def train_eval_set(cfg):
    # 基本不动
    cfg.TB_WRITER = True
    cfg.LOSS_EPOCH_TB = False
    cfg.USE_MGPU_EVAL = True  # 一个有一个没得会卡死
    cfg.IS_MULTI_SCALE = False  # 关多尺度训练
    cfg.FILE_NAME_WEIGHT = '123' + '.pth'  # 重新开始

    batch = 64  # type
    # batch = 3
    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        batch *= 2
        cfg.IS_COCO_EVAL = False

    size = (416, 416)  # type
    cfg_type4(cfg, batch=batch, image_size=size)  # 加载数据基础参数
    # cfg_type3(cfg, batch=batch, image_size=size)  # 加载数据基础参数
    # cfg.NUM_ANC = len(cfg.ANCS_SCALE) # cfg_type2 已有

    cfg.NUMS_EVAL = {10: 10, 100: 3, 160: 2}
    # cfg.NUM_SAVE_INTERVAL = 100

    '''特有参数'''
    cfg.MODE_TRAIN = 1  # base
    # cfg.MODE_TRAIN = 5  # yolo5
    cfg.NUM_REG = 1  # 这个是必须
    cfg.KEEP_SIZE = False  # 有anc建议用这个

    # type3 dark19
    # cfg.FILE_NAME_WEIGHT = 't_yolo3_type3_dark19-200_4.841' + '.pth'  # conf-0.01 nms-0.5
    # cfg.FILE_NAME_WEIGHT = 't_yolo3_type3_dark19-300_4.744' + '.pth'  # conf-0.01 nms-0.5
    # cfg.FILE_NAME_WEIGHT = 't_yolo3_type3_res18-90_5.154' + '.pth'  # conf-0.01 nms-0.5
    cfg.MAPS_VAL = [0.70, 0.47]  # 最高

    cfg.LR0 = 1e-3
    # cfg.DEL_TB = True 已弃用
    cfg.TB_WRITER = True
    cfg.IS_FORCE_SAVE = False  # 强制记录


def init_model(cfg, device, id_gpu=None):
    # model = darknet53(pretrained=True)
    # model = ModelOuts4DarkNet53(model)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_dark53'

    # model = darknet19(pretrained=True)
    # model = ModelOuts4DarkNet19(model)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_dark19'

    model = models.resnet18(pretrained=True)
    dims_out = (128, 256, 512)
    model = ModelOuts4Resnet(model, dims_out)
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_res18'

    # model = CSPDarknet_slim()
    # model.dims_out = (128, 256, 512)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_csspds'

    model = Yolo_v3(backbone=model, cfg=cfg)
    # f_look_model(model, input=(1, 3, *cfg.IMAGE_SIZE))

    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        for name, param in model.net.backbone.named_parameters():
            param.requires_grad_(False)

    model, is_mgpu = model_device_init(model, device, id_gpu, cfg)
    # ------------------------模型完成-------------------------------

    pg = model.parameters()
    optimizer = optim.Adam(pg, cfg.LR0)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 80, 100], 0.1)
    start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device, is_mgpu=is_mgpu)

    model.cfg = cfg
    return model, optimizer, lr_scheduler, start_epoch


if __name__ == '__main__':
    cfg = CFG
    path_project_root = '/AI/temp/tmp_pycharm/DL/object_detection/z_yolov3'
    # cfg.LR0 = 1e-3

    train = Train_1gpu(cfg, train_eval_set, init_model, path_project_root)
    train.f_run()

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
