import os
import sys

from f_pytorch.tools_model.backbones.darknet import darknet19

'''用户命令行启动'''
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from f_tools.fits.fitting.f_fit_class_base import Train_1gpu

from torch import optim
from f_tools.datas.data_loader import cfg_raccoon, cfg_type2, DataLoader2

from f_pytorch.tools_model.f_layer_get import ModelOut4Mobilenet_v2, ModelOut4Resnet18, ModelOut4Mobilenet_v3, \
    ModelOut4Resnet50, ModelOuts4DarkNet19, ModelOut4DarkNet19
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_gpu.f_gpu_api import model_device_init, mgpu_process0_init
from object_detection.z_yolov1.nets.net_yolov1 import Yolo_v1_1

from object_detection.z_yolov1.CONFIG_YOLOV1 import CFG

from f_tools.GLOBAL_LOG import flog
from torchvision import models

'''

linux用这个   python /AI/temp/tmp_pycharm/DL/object_detection/z_yolov1/train_yolov1.py
tensorboard --host=192.168.0.199 --logdir=/AI/temp/tmp_pycharm/DL/object_detection/z_yolov1/runs_type3

focloss 训练 75轮开始收敛 AP67_43
126轮AP72_47.5

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.362
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.720
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.331
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.009
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.105
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.440
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.397
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.475
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.478
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.016
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.207
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.555
'''


def train_eval_set(cfg):
    # 基本不动
    cfg.TB_WRITER = True
    cfg.LOSS_EPOCH = False
    cfg.USE_MGPU_EVAL = True  # 一个有一个没得会卡死
    cfg.IS_MULTI_SCALE = False  # 关多尺度训练
    cfg.FILE_NAME_WEIGHT = '123' + '.pth'  # 重新开始

    batch = 32  # type
    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        batch *= 2
        cfg.IS_COCO_EVAL = False

    size = (416, 416)  # 多尺寸时这个用于预测
    cfg_type2(cfg, batch=batch, image_size=size)  # 加载数据基础参数
    # cfg_raccoon(cfg, batch=batch, image_size=size)  # 加载数据基础参数

    cfg.NUMS_EVAL = {10: 5, 100: 3, 160: 2}
    # cfg.NUMS_EVAL = {100: 3, 160: 2}
    # cfg.NUM_SAVE_INTERVAL = 100

    # cfg.loss_args['s_conf'] = 'ohem'
    cfg.loss_args['s_conf'] = 'mse'
    # cfg.KEEP_SIZE = True  # ap-54 loss-2.6 lxy-2.3
    cfg.KEEP_SIZE = False

    # 稳定训练用这个
    # cfg.START_EVAL = 50  # 这个要 + EVAL_INTERVAL
    # cfg.EVAL_INTERVAL = 3  #

    cfg.arg_focalloss_alpha = 0.65
    cfg.IS_MIXTURE_FIX = True
    cfg.NUM_ANC = 1

    # type3 resnet18
    # cfg.FILE_NAME_WEIGHT = 'zz/t_yolo1_type3_res18c0.01-110_4.47_p72.4_r46.2' + '.pth'  # conf-0.01 nms-0.5
    # cfg.FILE_NAME_WEIGHT = 't_yolo1_type3_res18-50_5.369' + '.pth'  # conf-0.01 nms-0.5
    cfg.MAPS_VAL = [0.724, 0.47]

    cfg.LR0 = 1e-3 / 1.5
    cfg.TB_WRITER = True
    cfg.DEL_TB = True
    cfg.IS_FORCE_SAVE = False  # 强制记录


def init_model(cfg, device, id_gpu=None):
    # model = models.mobilenet_v2(pretrained=True)
    # model = ModelOut4Mobilenet_v2(model)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_mv3'

    # model = f_get_mv3(cfg.PATH_HOST, device)
    # model = ModelOut4Mobilenet_v3(model)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_mv3'

    model = models.resnet18(pretrained=True)
    model = ModelOut4Resnet18(model)
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_res18'

    # model = darknet19(pretrained=True)
    # model = ModelOut4DarkNet19(model)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_d19'

    model = Yolo_v1_1(backbone=model, cfg=cfg)
    # f_look_model(model, input=(1, 3, *cfg.IMAGE_SIZE))

    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        for name, param in model.backbone.named_parameters():
            param.requires_grad_(False)

    model, is_mgpu = model_device_init(model, device, id_gpu, cfg)

    # ------------------------自定义backbone完成-------------------------------
    pg = model.parameters()
    optimizer = optim.Adam(pg, cfg.LR0)
    # optimizer = optim.Adam(pg, cfg.LR0, weight_decay=5e-4)
    # optimizer = optim.SGD(pg, lr=cfg.LR0, momentum=0.937, weight_decay=5e-4, nesterov=True)
    # optimizer = optim.SGD(pg, lr=cfg.LR0, momentum=0.9, weight_decay=5e-4)
    # 两次不上升，降低一半
    # lr_scheduler = None
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 90, 120], 0.1)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=1, verbose=True)
    # start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, None, lr_scheduler, device, is_mgpu=is_mgpu)
    start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device, is_mgpu=is_mgpu)

    model.cfg = cfg
    return model, optimizer, lr_scheduler, start_epoch


if __name__ == '__main__':
    cfg = CFG
    path_project_root = '/AI/temp/tmp_pycharm/DL/object_detection/z_yolov1'
    # cfg.LR0 = 1e-3

    train = Train_1gpu(cfg, train_eval_set, init_model, path_project_root)
    train.f_run()

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
