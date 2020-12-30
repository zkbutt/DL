import os
import sys

'''用户命令行启动'''
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from torch import optim

from f_pytorch.tools_model.f_layer_get import ModelOut4Mobilenet_v2, ModelOut4Resnet18
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_gpu.f_gpu_api import model_device_init
from object_detection.z_yolov1.nets.net_yolov1 import Yolo_v1

from object_detection.z_yolov1.CONFIG_YOLOV1 import CFG

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_fit_fun import init_od, base_set, train_eval4od, fdatas_l2

'''

linux用这个   python /AI/temp/tmp_pycharm/DL/object_detection/z_yolov1/train_yolov1.py
'''
from torchvision import models


def init_model(cfg, device, id_gpu=None):
    # model = models.mobilenet_v2(pretrained=True)
    # model = ModelOut4Mobilenet_v2(model)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'mobilenet_v2'

    model = models.resnet18(pretrained=True)
    model = ModelOut4Resnet18(model)
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'resnet18'

    model = Yolo_v1(backbone=model, dim_in=model.dim_out, grid=cfg.NUM_GRID,
                    num_classes=cfg.NUM_CLASSES, num_bbox=cfg.NUM_BBOX, cfg=cfg)
    # f_look_model(model, input=(1, 3, *cfg.IMAGE_SIZE))

    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        for name, param in model.backbone.named_parameters():
            param.requires_grad_(False)
        # 除最后的全连接层外，其他权重全部冻结
        # if "fc" not in name:
        #     param.requires_grad_(False)

    model, is_mgpu = model_device_init(model, device, id_gpu, cfg)

    # ------------------------自定义backbone完成-------------------------------
    pg = model.parameters()
    # 最初学习率
    lr0 = 1e-3
    lrf = lr0 / 100
    # 权重衰减(如L2惩罚)(默认: 0)
    optimizer = optim.Adam(pg, lr0)
    # optimizer = optim.Adam(pg, lr0, weight_decay=5e-4)
    # 两次不上升，降低一半
    lr_scheduler = None
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=1, verbose=True)
    start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, None, lr_scheduler, device, is_mgpu=is_mgpu)
    # start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device, is_mgpu=is_mgpu)

    model.cfg = cfg
    return model, optimizer, lr_scheduler, start_epoch


def train_set(cfg):
    # cfg.FILE_NAME_WEIGHT = 'train_yolo1_raccoon200mobilenet_v2-1_47.912' + '.pth'
    cfg.FILE_NAME_WEIGHT = 'train_yolo1_raccoon200resnet18-1_42.395' + '.pth'
    # cfg.FILE_NAME_WEIGHT = '123' + '.pth'
    pass


if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    # -----------通用系统配置----------------
    init_od()
    device, cfg = base_set(CFG)
    train_set(cfg)

    '''---------------数据加载及处理--------------'''
    loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler = cfg.FUN_LOADER_DATA(cfg,
                                                                                                      is_mgpu=False, )
    '''------------------模型定义---------------------'''
    model, optimizer, lr_scheduler, start_epoch = init_model(cfg, device, id_gpu=None)

    train_eval4od(start_epoch=start_epoch, model=model, optimizer=optimizer,
                  fdatas_l2=fdatas_l2, lr_scheduler=lr_scheduler,
                  loader_train=loader_train, loader_val_fmap=loader_val_fmap, loader_val_coco=loader_val_coco,
                  device=device, train_sampler=None, eval_sampler=None,
                  tb_writer=None,
                  )
    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
