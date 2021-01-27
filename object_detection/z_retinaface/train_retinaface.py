import os
import sys

'''用户命令行启动'''
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from f_tools.datas.data_loader import cfg_type, DataLoader

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.fitting.f_fit_fun import init_od_e, base_set_1gpu, train_eval4od, fdatas_l2, show_train_info

from torch import optim
from torchvision import models

from f_pytorch.tools_model.f_layer_get import ModelOuts4Mobilenet_v2
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_gpu.f_gpu_api import model_device_init

from object_detection.z_retinaface.nets.net_retinaface import RetinaFace
from object_detection.z_retinaface.CONFIG_RETINAFACE import CFG


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

    model = models.mobilenet_v2(pretrained=True)
    model = ModelOuts4Mobilenet_v2(model)
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_m2'
    cfg.FEATURE_MAP_STEPS = [8, 16, 32]  # 特图的步距 下采倍数

    # from f_pytorch.tools_model.model_look import f_look_model
    # f_look_model(model, input=(1, 3, 416, 416))
    # # conv 可以取 in_channels 不支持数组层

    model = RetinaFace(backbone=model, num_classes=cfg.NUM_CLASSES, anchor_num=cfg.NUMS_ANC[0],
                       in_channels_fpn=model.dims_out, cfg=cfg, device=device)
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
    lrf = cfg.LR0 / 100
    # 权重衰减(如L2惩罚)(默认: 0)
    optimizer = optim.Adam(pg, cfg.LR0, weight_decay=5e-4)
    # 两次不上升，降低一半
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.85, patience=1, verbose=True)
    # lr_scheduler = f_lr_cos(optimizer, 0, cfg.END_EPOCH, lrf)
    lr_scheduler = None
    start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, None, lr_scheduler, device, is_mgpu=is_mgpu)

    model.cfg = cfg
    return model, optimizer, lr_scheduler, start_epoch


def train_eval_set(cfg):
    cfg.FILE_NAME_WEIGHT = '123' + '.pth'  # 重新开始

    # raccoon mobilenet_v2
    # cfg.FILE_NAME_WEIGHT = 'zz/train_retina_raccoon200_m2-base' + '.pth'  # 初始resnet
    # cfg.FILE_NAME_WEIGHT = 'zz/train_retina_raccoon200_m2c0.5-15_1.11_p61.3_r27.9' + '.pth'
    # cfg.MAPS_VAL = [0.613, 0.279]
    # cfg.BATCH_SIZE = 40  # m2
    # cfg_raccoon(cfg)  # 加载数据基础参数

    # raccoon- res50
    # cfg.FILE_NAME_WEIGHT = 'zz/train_retina_raccoon200_res50-base' + '.pth'  # 初始resnet
    # cfg.FILE_NAME_WEIGHT = 'zz/train_retina_raccoon200_m2c0.5-15_1.11_p61.3_r27.9' + '.pth'
    # cfg.MAPS_VAL = [0.618, 0.316]
    # cfg.BATCH_SIZE = 20  # m2
    # cfg_raccoon(cfg)  # 加载数据基础参数

    # type3 mobilenet_v2
    # cfg.FILE_NAME_WEIGHT = 'zz/train_retina_type3_m2-1_base' + '.pth'  # 初始resnet
    # cfg.FILE_NAME_WEIGHT = 'zz/train_retina_type3_m2c0.5-10_3.92_p60.2_r43.6' + '.pth'  # conf>0.5
    cfg.FILE_NAME_WEIGHT = 'train_retina_type3_m2c0.5-2_8.36_p13.7_r8.3' + '.pth'  # conf>0.5
    cfg.BATCH_SIZE = 40
    # cfg.MAPS_VAL = [0.595, 0.431]
    cfg_type(cfg)  # 加载数据基础参数

    # type3 res50
    # cfg.FILE_NAME_WEIGHT = 'zz/train_retina_type3_res50-base' + '.pth'  # conf>0.7
    # cfg.FILE_NAME_WEIGHT = 'zz/train_retina_type3_res50-15_5.28_p16.6_r13.3' + '.pth'  # conf>0.7
    # cfg.BATCH_SIZE = 24
    # cfg.MAPS_VAL = [0.1667, 0.134]

    # type3 resnet50
    # cfg.FILE_NAME_WEIGHT = 'zz/train_retina_type3densenet121-1_base' + '.pth'  # 初始resnet
    # cfg.FILE_NAME_WEIGHT = 'train_retina_type3_m2-9_5.64_p33.3_r20.2' + '.pth'  # conf>0.5
    # cfg.MAPS_VAL = [0.334, 0.203]

    cfg.LR0 = 1e-3
    cfg.DEL_TB = True
    cfg.USE_MGPU_EVAL = True  # 一个有一个没得会卡死


'''
IOU大于0.5的为正  小于0.3的为负 负例倒序取3倍
'''

if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    # -----------通用系统配置----------------
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
