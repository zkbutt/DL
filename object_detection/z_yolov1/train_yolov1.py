import os
import sys

'''用户命令行启动'''
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from torch import optim
from f_tools.datas.data_loader import cfg_raccoon, DataLoader, cfg_type

from f_pytorch.tools_model.f_layer_get import ModelOut4Mobilenet_v2, ModelOut4Resnet18, ModelOut4Mobilenet_v3, \
    ModelOut4Resnet50
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_gpu.f_gpu_api import model_device_init, mgpu_process0_init
from object_detection.z_yolov1.nets.net_yolov1 import Yolo_v1_1

from object_detection.z_yolov1.CONFIG_YOLOV1 import CFG

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_fit_fun import init_od, base_set, train_eval4od, fdatas_l2, show_train_info

'''

linux用这个   python /AI/temp/tmp_pycharm/DL/object_detection/z_yolov1/train_yolov1.py

[ obj 2.33 || cls 0.33 || bbox 3.70 || total 6.36 ]

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.316
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.676
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.263
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.003
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.105
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.401
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.345
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.427
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.431
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.004
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.204
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.516
'''
from torchvision import models


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

    '''这个训不动'''
    # model = models.resnet50(pretrained=True)
    # model = ModelOut4Resnet50(model)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_res50'

    model = Yolo_v1_1(backbone=model, cfg=cfg)

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
    optimizer = optim.Adam(pg, cfg.LR0)
    # optimizer = optim.Adam(pg, cfg.LR0, weight_decay=5e-4)
    # optimizer = optim.SGD(pg, lr=cfg.LR0, momentum=0.937, weight_decay=5e-4, nesterov=True)
    # optimizer = optim.SGD(pg, lr=cfg.LR0, momentum=0.9, weight_decay=5e-4)
    # 两次不上升，降低一半
    # lr_scheduler = None
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 90], 0.1)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=1, verbose=True)
    start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, None, lr_scheduler, device, is_mgpu=is_mgpu)
    start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device, is_mgpu=is_mgpu)

    model.cfg = cfg
    return model, optimizer, lr_scheduler, start_epoch


def train_eval_set(cfg):
    cfg.FILE_NAME_WEIGHT = '123' + '.pth'  # 重新开始

    # batch = 20  # raccoon
    batch = 32  # type
    # batch = 5  # type
    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        batch *= 3
        cfg.IS_COCO_EVAL = False

    # batch = 10  # type
    size = (416, 416)  # type
    cfg_type(cfg, batch=batch, image_size=size)  # 加载数据基础参数
    # cfg_raccoon(cfg, batch=batch, image_size=size)  # 加载数据基础参数
    cfg.START_EVAL = 10  # cfg.START_EVAL=10 and EVAL_INTERVAL=3 实际是12
    cfg.END_EVAL = 60  # cfg.START_EVAL=10 and EVAL_INTERVAL=3 实际是12
    cfg.EVAL_INTERVAL = 3
    # cfg.NUM_SAVE_INTERVAL = 100
    cfg.match_str = 'log'  # 'log' 'whoned'
    cfg.loss_conf_str = 'mse'  # 'mse' 'foc' 迭代400次开始收敛
    cfg.arg_focalloss_alpha = 0.75
    cfg.IS_MIXTURE_FIX = False

    # raccoon mobilenet_v2
    # cfg.FILE_NAME_WEIGHT = 'zz/train_yolo1_raccoon200_mv2-base' + '.pth'  # 初始resnet
    # cfg.FILE_NAME_WEIGHT = 'zz/train_yolo1_raccoon200_mv2-39_0.52_p52.8_r31.7' + '.pth'
    # cfg.MAPS_VAL = [0.529, 0.318]

    # raccoon resnet18
    # cfg.FILE_NAME_WEIGHT = 'zz/train_yolo1_raccoon200_res18-base' + '.pth'  # 初始resnet
    # cfg.FILE_NAME_WEIGHT = 'train_yolo1_raccoon200_res18-30_80.742' + '.pth'  # 最好
    # cfg.MAPS_VAL = [0.614, 0.351]

    # raccoon mobilenet_v3
    # cfg.FILE_NAME_WEIGHT = 'zz/train_yolo1_raccoon200_mv3-base' + '.pth'  # 初始resnet
    # cfg.FILE_NAME_WEIGHT = 'zz/train_yolo1_raccoon200_mv3-5_0.94_p25.5_r11.1' + '.pth'
    # cfg.MAPS_VAL = [0.60, 0.24]

    # type3 resnet18 0:00:29
    cfg.FILE_NAME_WEIGHT = 't_yolo1_type3_res18c0.01-47_3.1_p50.2_r31.7' + '.pth'
    # cfg.FILE_NAME_WEIGHT = 't_yolo1_type3_res18c0.01-47_3.1_p50.2_r31.7' + '.pth'
    '''cls455收  wh550收'''
    cfg.MAPS_VAL = [0.544, 0.316]

    cfg.LR0 = 1e-3
    cfg.TB_WRITER = True
    cfg.DEL_TB = True
    cfg.LOSS_EPOCH = False
    cfg.USE_MGPU_EVAL = True  # 一个有一个没得会卡死


if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    init_od()
    device, cfg = base_set(CFG, id_gpu=1)
    train_eval_set(cfg)

    cfg.PATH_PROJECT_ROOT = cfg.PATH_HOST + '/AI/temp/tmp_pycharm/DL/object_detection/z_yolov1'  # 这个要改

    '''---------------数据加载及处理--------------'''
    data_loader = DataLoader(cfg)
    _ret = data_loader.get_train_eval_datas(is_mgpu=False)
    loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler = _ret

    show_train_info(cfg, loader_train, loader_val_coco)

    '''------------------模型定义---------------------'''
    model, optimizer, lr_scheduler, start_epoch = init_model(cfg, device, id_gpu=None)
    tb_writer = None
    if cfg.TB_WRITER:
        tb_writer = mgpu_process0_init(None, cfg, loader_train, loader_val_coco, model, device)
        show_train_info(cfg, loader_train, loader_val_coco)

    train_eval4od(start_epoch=start_epoch, model=model, optimizer=optimizer,
                  fdatas_l2=fdatas_l2, lr_scheduler=lr_scheduler,
                  loader_train=loader_train, loader_val_fmap=loader_val_fmap, loader_val_coco=loader_val_coco,
                  device=device, train_sampler=None, eval_sampler=None,
                  tb_writer=tb_writer, maps_def=cfg.MAPS_VAL
                  )
    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
