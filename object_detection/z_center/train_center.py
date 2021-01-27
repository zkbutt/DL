import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from f_tools.datas.data_loader import cfg_type, DataLoader
from torch import optim
from torchvision import models

from f_pytorch.tools_model.f_layer_get import ModelOut4Mobilenet_v2
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_gpu.f_gpu_api import model_device_init
from object_detection.z_center.nets.net_center import CenterNet

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.fitting.f_fit_fun import init_od_e, base_set_1gpu, train_eval4od, show_train_info, fdatas_l2
from object_detection.z_center.CONFIG_CENTER import CFG


def init_model(cfg, device, id_gpu=None):
    model = models.mobilenet_v2(pretrained=True)
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'm2'
    model = ModelOut4Mobilenet_v2(model)

    # model = models.resnet50(pretrained=True)
    # model = ModelOut4Resnet50(model)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'r50'

    model = CenterNet(cfg=cfg, backbone=model, num_classes=cfg.NUM_CLASSES, dim_in_backbone=model.dim_out)
    # f_look_model(model, input=(1, 3, *cfg.IMAGE_SIZE))

    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        for name, param in model.backbone.named_parameters():
            param.requires_grad_(False)
        # 除最后的全连接层外，其他权重全部冻结
        # if "fc" not in name:
        #     param.requires_grad_(False)

    model, is_mgpu = model_device_init(model, device, id_gpu, cfg)
    # ------------------------模型完成-------------------------------
    pg = model.parameters()
    optimizer = optim.Adam(pg, cfg.LR0, weight_decay=5e-4)
    # 两次不上升，降低一半
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=3, verbose=True)
    lr_scheduler = None
    start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, None, lr_scheduler, device, is_mgpu=is_mgpu)

    model.cfg = cfg
    return model, optimizer, lr_scheduler, start_epoch


def train_eval_set(cfg):
    cfg.FILE_NAME_WEIGHT = '123' + '.pth'  # 重新开始

    # batch = 20  # raccoon
    batch = 32  # type
    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        batch *= 3
        cfg.IS_COCO_EVAL = False

    # batch = 10  # type
    size = (512, 512)  # type
    cfg_type(cfg, batch=batch, image_size=size)  # 加载数据基础参数
    # size = (416, 416)  # -> 104*104
    # size = (320, 320)  # -> 80*80
    # cfg_raccoon(cfg, batch=batch, image_size=size)  # 加载数据基础参数

    # raccoon mobilenet_v2
    # cfg.FILE_NAME_WEIGHT = 'zz/t_center_raccoon200m2-base' + '.pth'  # 初始resnet
    # cfg.FILE_NAME_WEIGHT = 'zz/t_center_raccoon200m2c0.7-16_4.25_p65.4_r31.9' + '.pth'  # [1, 5., 7., 1.]
    # cfg.FILE_NAME_WEIGHT = 't_center_raccoon200m2c0.4-32_9593.52_p100.0_r20.0' + '.pth'  # [1, 4, 6., 1.]
    # cfg.MAPS_VAL = [0.654, 0.319]

    # raccoon- res50
    # cfg.FILE_NAME_WEIGHT = 'zz/t_center_raccoon200r50-base' + '.pth'  # 初始resnet
    # cfg.FILE_NAME_WEIGHT = 'zz/train_retina_raccoon200_m2c0.5-15_1.11_p61.3_r27.9' + '.pth'
    # cfg.MAPS_VAL = [0.444, 0.29]
    # cfg.BATCH_SIZE = 20  # m2
    # cfg_raccoon(cfg)  # 加载数据基础参数

    # type3 mobilenet_v2
    # cfg.FILE_NAME_WEIGHT = 'zz/t_center_type3m2-base' + '.pth'  # 初始resnet
    # cfg.FILE_NAME_WEIGHT = 't_center_type3m2-1_2.119' + '.pth'
    cfg.FILE_NAME_WEIGHT = 't_center_type3m2-20_1.645' + '.pth'
    cfg.FILE_NAME_WEIGHT = 't_center_type3m2-40_2.249' + '.pth'
    cfg.FILE_NAME_WEIGHT = 't_center_type3m2-50_2.126' + '.pth'
    cfg.FILE_NAME_WEIGHT = 't_center_type3m2-90_1.998' + '.pth'

    # cfg.FILE_NAME_WEIGHT = 't_center_type3m2-20_23248.586' + '.pth'
    # cfg.FILE_NAME_WEIGHT = 't_center_type3m2-20_1.062' + '.pth'
    # cfg.MAPS_VAL = [0.595, 0.431]
    # cfg.BATCH_SIZE = 40
    # cfg_type(cfg)  # 加载数据基础参数
    # cfg.IMAGE_SIZE = (512, 512)

    # type3 res50
    # cfg.FILE_NAME_WEIGHT = 'zz/train_retina_type3_res50-base' + '.pth'  # conf>0.7
    # cfg.FILE_NAME_WEIGHT = 'zz/train_retina_type3_res50-15_5.28_p16.6_r13.3' + '.pth'  # conf>0.7
    # cfg.BATCH_SIZE = 24
    # cfg.MAPS_VAL = [0.1667, 0.134]

    # type3 resnet50
    # cfg.FILE_NAME_WEIGHT = 'zz/train_retina_type3densenet121-1_base' + '.pth'  # 初始resnet
    # cfg.FILE_NAME_WEIGHT = 'train_retina_type3_m2-9_5.64_p33.3_r20.2' + '.pth'  # conf>0.5
    # cfg.MAPS_VAL = [0.334, 0.203]

    cfg.LR0 = 1e-4
    cfg.DEL_TB = False
    cfg.USE_MGPU_EVAL = True  # 一个有一个没得会卡死


'''
python /AI/temp/tmp_pycharm/DL/object_detection/z_center/train_center.py
'''

if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    init_od_e()
    device, cfg = base_set_1gpu(CFG)
    train_eval_set(cfg)

    '''---------------数据加载及处理--------------'''
    data_loader = DataLoader(cfg)
    _ret = data_loader.get_train_eval_datas(is_mgpu=False)
    loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler = _ret

    show_train_info(cfg, loader_train, loader_val_coco)

    '''------------------模型定义---------------------'''
    model, optimizer, lr_scheduler, start_epoch = init_model(cfg, device, id_gpu=None)

    train_eval4od(start_epoch=start_epoch, model=model, optimizer=optimizer,
                  fdatas_l2=fdatas_l2, lr_scheduler=lr_scheduler,
                  loader_train=loader_train, loader_val_fmap=loader_val_fmap, loader_val_coco=loader_val_coco,
                  device=device, train_sampler=None, eval_sampler=None,
                  tb_writer=None, maps_def=cfg.MAPS_VAL
                  )
    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
