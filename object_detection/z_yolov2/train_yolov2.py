import os
import sys

'''用户命令行启动'''
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from torchvision import models
from f_tools.fits.fitting.f_fit_class_base import Train_1gpu
from f_pytorch.tools_model.f_layer_get import ModelOuts4DarkNet19, ModelOuts4Resnet
from object_detection.z_yolov2.nets.net_yolov2 import Yolo_v2
from object_detection.z_yolov2.CONFIG_YOLOV2 import CFG
from f_pytorch.tools_model.backbones.darknet import darknet19
from torch import optim
from f_tools.datas.data_loader import cfg_type3, DataLoader2, cfg_type4, cfg_voc

from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_gpu.f_gpu_api import model_device_init, mgpu_process0_init

from f_tools.GLOBAL_LOG import flog

'''

linux用这个   python /AI/temp/tmp_pycharm/DL/object_detection/z_yolov2/train_yolov2.py
tensorboard --host=192.168.0.199 --logdir=/AI/temp/tmp_pycharm/DL/object_detection/z_yolov2/runs_type4
与yolo1的区别在添加anc的先验, 预测的是wh的比例 yolo为直接预测wh实际值
正反例 845：1


'''


def train_eval_set(cfg):
    # 基本不动
    cfg.TB_WRITER = True
    cfg.LOSS_EPOCH = False
    cfg.USE_MGPU_EVAL = True  # 一个有一个没得会卡死
    cfg.IS_MULTI_SCALE = False  # 关多尺度训练
    cfg.FILE_NAME_WEIGHT = '123' + '.pth'  # 重新开始

    # batch = 16  # type
    batch = 64
    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        batch *= 2
        cfg.IS_COCO_EVAL = False

    size = (416, 416)  # type
    cfg_voc(cfg, batch=batch, image_size=size)  # 加载数据基础参数
    # cfg_type3(cfg, batch=batch, image_size=size)  # 加载数据基础参数
    # cfg_type4(cfg, batch=batch, image_size=size)  # 加载数据基础参数
    # cfg_raccoon(cfg, batch=batch, image_size=size)  # 加载数据基础参数

    # anc重写
    # cfg.ANCS_SCALE = [[0.034, 0.064],
    #                   [0.07, 0.146],
    #                   [0.166, 0.218],
    #                   [0.406, 0.332],
    #                   [0.898, 0.49]]
    cfg.NUM_ANC = len(cfg.ANCS_SCALE)

    cfg.MODE_TRAIN = 1  # base
    # cfg.MODE_TRAIN = 4  # IOU 损失及预测

    cfg.NUMS_EVAL = {10: 5, 100: 3, 160: 2}
    # cfg.NUM_SAVE_INTERVAL = 100

    # type3 dark19
    # cfg.FILE_NAME_WEIGHT = 'zz/t_yolo2_type3_dark19c0.01-137_3.94_p73.5_r49.8' + '.pth'  # conf-0.01 nms-0.5
    cfg.FILE_NAME_WEIGHT = 't_yolo2__voc_res18-50_5.492' + '.pth'  # conf-0.01 nms-0.5
    cfg.MAPS_VAL = [0.71, 0.46]  # 最高

    cfg.LR0 = 1e-3 / 2
    cfg.TB_WRITER = True
    cfg.DEL_TB = True
    cfg.IS_FORCE_SAVE = False  # 强制记录


def init_model(cfg, device, id_gpu=None):
    # model = darknet19(pretrained=True)
    # model = ModelOuts4DarkNet19(model)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_dark19'

    model = models.resnet18(pretrained=True)
    dims_out = (128, 256, 512)
    model = ModelOuts4Resnet(model, dims_out)
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_res18'

    # model = models.resnet50(pretrained=True)
    # dims_out = (512, 1024, 2048)
    # model = ModelOuts4Resnet(model, dims_out)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_res50'

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
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 80, 100], 0.1)
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
