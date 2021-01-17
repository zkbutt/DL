import os
import sys

'''用户命令行启动'''
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from torch import optim
from f_tools.datas.data_loader import cfg_type, DataLoader
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_gpu.f_gpu_api import model_device_init
from object_detection.z_yolov3.nets.net_yolov3 import YoloV3SPP
import math
from object_detection.z_yolov3.CONFIG_YOLO3 import CFG
from f_tools.fits.f_fit_fun import init_od, base_set, show_train_info, train_eval4od, fdatas_l2

from f_tools.GLOBAL_LOG import flog
from torchvision import models
from f_pytorch.tools_model.f_layer_get import ModelOut4Resnet18, ModelOut4Mobilenet_v2, ModelOut4Resnet50, \
    ModelOuts4Mobilenet_v2, ModelOuts4Resnet

'''
python /AI/temp/tmp_pycharm/DL/object_detection/z_yolov3/train_yolo3.py
'''


def init_model(cfg, device, id_gpu=None):
    # model = models.mobilenet_v2(pretrained=True)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'm2'
    # model = ModelOuts4Mobilenet_v2(model)

    model = models.resnet18(pretrained=True)
    dims_out = (128, 256, 512)
    model = ModelOuts4Resnet(model, dims_out=dims_out)
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'r18'
    cfg.FEATURE_MAP_STEPS = [8, 16, 32]

    model = YoloV3SPP(backbone=model, nums_anc=cfg.NUMS_ANC, num_classes=cfg.NUM_CLASSES,
                      dims_rpn_in=model.dims_out, device=device, cfg=cfg, is_spp=True)
    # f_look_model(model, input=(1, 3, *cfg.IMAGE_SIZE))

    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        for name, param in model.backbone.named_parameters():
            param.requires_grad_(False)
        # 除最后的全连接层外，其他权重全部冻结
        # if "fc" not in name:
        #     param.requires_grad_(False)

    # 通用
    model, is_mgpu = model_device_init(model, device, id_gpu, cfg)
    # ------------------------模型完成-------------------------------

    pg = model.parameters()
    # 最初学习率
    lr0 = cfg.LR0
    lrf = lr0 / 100
    # 权重衰减(如L2惩罚)(默认: 0)
    optimizer = optim.Adam(pg, lr0)
    # optimizer = optim.Adam(pg, lr0, weight_decay=5e-4)
    # 两次不上升，降低一半
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=3, verbose=True)
    lr_scheduler = None
    start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, None, lr_scheduler, device, is_mgpu=is_mgpu)
    # start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device, is_mgpu=is_mgpu)

    model.cfg = cfg
    return model, optimizer, lr_scheduler, start_epoch


def train_eval_set(cfg):
    cfg.FILE_NAME_WEIGHT = '123' + '.pth'  # 重新开始

    # batch = 20  # raccoon
    # cfg_raccoon(cfg, batch=batch, image_size=(512, 512))  # 加载数据基础参数
    batch = 40  # type
    batch = 5  # test
    size = (416, 416)
    cfg_type(cfg, batch=batch, image_size=size)  # 加载数据基础参数

    # raccoon mobilenet_v2
    # cfg.FILE_NAME_WEIGHT = 'zz/t_center_raccoon200m2-base' + '.pth'  # 初始resnet
    # cfg.FILE_NAME_WEIGHT = 'zz/t_center_raccoon200m2c0.7-16_4.25_p65.4_r31.9' + '.pth'  # [1, 5., 7., 1.]
    # cfg.FILE_NAME_WEIGHT = 'zz/t_center_raccoon200m2c0.7-16_3.78_p69.9_r32.8' + '.pth'  # [1, 4, 6., 1.]
    # cfg.MAPS_VAL = [0.654, 0.319]

    # raccoon- res50
    # cfg.FILE_NAME_WEIGHT = 'zz/t_center_raccoon200r50-base' + '.pth'  # 初始resnet
    # cfg.FILE_NAME_WEIGHT = 'zz/train_retina_raccoon200_m2c0.5-15_1.11_p61.3_r27.9' + '.pth'
    # cfg.MAPS_VAL = [0.618, 0.316]
    # cfg.BATCH_SIZE = 20  # m2
    # cfg_raccoon(cfg)  # 加载数据基础参数

    # type3 mobilenet_v2
    # cfg.FILE_NAME_WEIGHT = 'zz/t_center_type3m2-base' + '.pth'  # 初始resnet
    # cfg.FILE_NAME_WEIGHT = 't_center_type3m2-10_4.456' + '.pth'
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

    cfg.LR0 = 1e-3
    cfg.DEL_TB = True
    cfg.USE_MGPU_EVAL = True  # 一个有一个没得会卡死


if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    init_od()
    device, cfg = base_set(CFG)
    train_eval_set(cfg)

    if hasattr(cfg, 'FEATURE_MAP_STEPS'):
        # 预设尺寸必须是下采样倍数的整数倍 输入一般是正方形
        down_sample = cfg.FEATURE_MAP_STEPS[-1]  # 模型下采样倍数
        assert math.fmod(cfg.IMAGE_SIZE[0], down_sample) == 0, "尺寸 %s must be a %s 的倍数" % (cfg.IMAGE_SIZE, down_sample)
        # assert math.fmod(cfg.IMAGE_SIZE[1], down_sample) == 0, "尺寸 %s must be a %s 的倍数" % (cfg.IMAGE_SIZE, small_conf)

        '''-----多尺度训练-----'''
        if cfg.IS_MULTI_SCALE:
            # 动态输入尺寸选定 根据预设尺寸  0.667~1.5 之间 满足32的倍数
            imgsz_min = cfg.IMAGE_SIZE[0] // cfg.MULTI_SCALE_VAL[1]
            imgsz_max = cfg.IMAGE_SIZE[0] // cfg.MULTI_SCALE_VAL[0]
            # 将给定的最大，最小输入尺寸向下调整到32的整数倍
            grid_min, grid_max = imgsz_min // down_sample, imgsz_max // down_sample
            imgsz_min, imgsz_max = int(grid_min * down_sample), int(grid_max * down_sample)
            sizes_in = []
            for i in range(imgsz_min, imgsz_max + 1, down_sample):
                sizes_in.append(i)
            # imgsz_train = imgsz_max  # initialize with max size
            # img_size = random.randrange(grid_min, grid_max + 1) * gs
            flog.info("输入画像的尺寸范围为[{}, {}] 可选尺寸为{}".format(imgsz_min, imgsz_max, sizes_in))

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
