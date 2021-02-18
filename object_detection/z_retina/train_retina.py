import os
import sys

'''用户命令行启动'''
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from object_detection.z_retina.nets.net_retina import Retina, Retina2
from f_tools.datas.data_loader import cfg_type2
from f_tools.fits.fitting.f_fit_class_base import Train_1gpu
from f_tools.GLOBAL_LOG import flog

from torch import optim
from torchvision import models

from f_pytorch.tools_model.f_layer_get import ModelOuts4Mobilenet_v2, ModelOuts4Resnet
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_gpu.f_gpu_api import model_device_init

from object_detection.z_retina.CONFIG_RETINAFACE import CFG

'''
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.253
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.573
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.192
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.005
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.133
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.295
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.283
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.373
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.374
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.004
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.234
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.422
'''


def train_eval_set(cfg):
    # 基本不动
    cfg.TB_WRITER = True
    cfg.LOSS_EPOCH = False
    cfg.USE_MGPU_EVAL = True  # 一个有一个没得会卡死
    cfg.IS_MULTI_SCALE = False  # 关多尺度训练
    cfg.FILE_NAME_WEIGHT = '123' + '.pth'  # 重新开始

    batch = 32  # type
    # batch = 16  # type
    # batch = 2  # type
    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        batch *= 2
        cfg.IS_COCO_EVAL = False

    size = (416, 416)  # type
    cfg_type2(cfg, batch=batch, image_size=size)  # 加载数据基础参数

    cfg.NUMS_EVAL = {10: 5, 100: 3, 150: 1}
    # cfg.NUM_SAVE_INTERVAL = 100

    cfg.loss_args['s_conf'] = 'ohem'
    # cfg.loss_args['s_conf'] = 'foc'
    cfg.variances = (0.1, 0.2)

    # type3 dark19
    # cfg.FILE_NAME_WEIGHT = 'zz/t_yolo2_type3_dark19c0.01-137_3.94_p73.5_r49.8' + '.pth'  # conf-0.01 nms-0.5
    # cfg.FILE_NAME_WEIGHT = 't_retina_type3-10_366.704' + '.pth'  # conf-0.01 nms-0.5
    # cfg.FILE_NAME_WEIGHT = 't_retina_type3-50_220.132' + '.pth'  # conf-0.01 nms-0.5
    cfg.FILE_NAME_WEIGHT = 't_retina_type3-40_3.37' + '.pth'  # conf-0.01 nms-0.5
    cfg.MAPS_VAL = [0.58, 0.38]  # 最高

    cfg.LR0 = 1e-5
    cfg.DEL_TB = True
    cfg.TB_WRITER = True
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

    cfg.NUMS_ANC = [9, 9, 9, 9, 9]
    # cfg.NUMS_ANC = [1, 1, 1, 1, 1]
    cfg.FEATURE_MAP_STEPS = [8, 16, 32, 64, 128]
    # cfg.ANCS_SCALE = [[0.078, 0.07775],
    #                   [0.174, 0.164],
    #                   [0.324, 0.336],
    #                   [0.578, 0.466],
    #                   [0.698, 0.674]]
    cfg.ANCS_SCALE = [[0.054, 0.042],
                      [0.046, 0.072],
                      [0.106, 0.058],
                      [0.078, 0.088],
                      [0.136, 0.094],
                      [0.092, 0.142],
                      [0.18575, 0.106],
                      [0.144, 0.166],
                      [0.212, 0.14],
                      [0.134, 0.264],
                      [0.188, 0.196],
                      [0.252, 0.212],
                      [0.184, 0.296],
                      [0.354, 0.166],
                      [0.198, 0.408],
                      [0.25, 0.338],
                      [0.313, 0.281],
                      [0.478, 0.202],
                      [0.385, 0.254],
                      [0.324, 0.367],
                      [0.279, 0.435],
                      [0.388, 0.333],
                      [0.356, 0.446],
                      [0.573, 0.286],
                      [0.496, 0.334],
                      [0.438, 0.402],
                      [0.312, 0.57],
                      [0.596, 0.352],
                      [0.472, 0.511],
                      [0.576, 0.458],
                      [0.416, 0.656],
                      [0.82, 0.362],
                      [0.71, 0.437],
                      [0.566, 0.564],
                      [0.65, 0.528],
                      [0.6, 0.682],
                      [0.79, 0.521],
                      [0.702, 0.61],
                      [0.5, 0.86],
                      [0.928, 0.532],
                      [0.858, 0.602],
                      [0.776, 0.714],
                      [0.656, 0.922],
                      [0.958, 0.738],
                      [0.896, 0.95]]
    # cfg.LOSS_WEIGHT = [1, 1, 1, 1, 1]  # conf_pos conf_neg cls loss_txty  loss_twth
    model = Retina2(model, cfg, device)

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
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 80, 150], 0.1)
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
