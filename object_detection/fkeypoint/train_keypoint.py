import os
import sys

from object_detection.fkeypoint.CONFIG_FKEYPOINT import CFG
from object_detection.fkeypoint.net.net_keypoint import KFcos

'''用户命令行启动'''

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from f_tools.datas.data_loader import cfg_type3, cfg_type4, cfg_widerface, cfg_face98, cfg_face5
from f_tools.fits.fitting.f_fit_class_base import Train_1gpu
from f_tools.GLOBAL_LOG import flog

from torch import optim
from torchvision import models

from f_pytorch.tools_model.f_layer_get import ModelOuts4Mobilenet_v2, ModelOuts4Resnet
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_gpu.f_gpu_api import model_device_init

'''
tensorboard --host=192.168.0.199 --logdir=/AI/temp/tmp_pycharm/DL/object_detection/fcos/log/runs_type3/2021-06-18_19_12_04

'''


def train_eval_set(cfg):
    # 基本不动
    cfg.TB_WRITER = True
    cfg.LOSS_EPOCH_TB = False  # 这个用于控制训练时 TB_WRITER采用 显示iter次数 或还轮
    cfg.IS_MULTI_SCALE = False  # 关多尺度训练

    # batch = 16  # type
    batch = 2  # type
    if cfg.IS_LOCK_BACKBONE_WEIGHT:  # 锁定权重时 增大batch 验证关闭
        batch *= 2
        cfg.IS_COCO_EVAL = False

    cfg.MODE_TRAIN = 1  # base

    # cfg.USE_BASE4NP = True  # 最简图像预处理 这个用于测试
    # cfg_face98(cfg, batch=batch, image_size=size)  # 加载数据基础参数

    size = (128, 128)  # cfg_face5
    cfg_face5(cfg, batch=batch, image_size=size, mode='bbox')  # 加载数据基础参数

    # cfg_widerface(cfg, batch=batch, image_size=size, mode='keypoints')  # bbox segm keypoints caption

    cfg.MODE_TRAIN = 1  # 单人脸
    cfg.NUMS_EVAL = {2: 1, 100: 1, 160: 1}
    cfg.NUM_SAVE_INTERVAL = 1
    cfg.KEEP_SIZE = True  # 开启避免人脸太小 数据增强不支持

    # 原装是416 输出为5层 (0,64,128,256,512,inf)

    # type3 dark19
    # cfg.FILE_NAME_WEIGHT = 'zz/t_yolo2_type3_dark19c0.01-137_3.94_p73.5_r49.8' + '.pth'  # conf-0.01 nms-0.5
    # cfg.FILE_NAME_WEIGHT = 't_fcos__widerface_res18-1_7.19' + '.pth'  # conf-0.01 nms-0.5
    # cfg.MAPS_VAL = [0.8726537249998292, 0.6458333333333334]

    cfg.LR0 = 1e-3
    cfg.IS_FORCE_SAVE = False  # 强制记录 用于多GPU时 初始化权重保存


def init_model(cfg, device, id_gpu=None):
    model = models.resnet18(pretrained=True)
    dims_out = (128, 256, 512)
    model = ModelOuts4Resnet(model, dims_out)
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_res18'

    # model = models.resnet50(pretrained=True)
    # dims_out = (512, 1024, 2048)
    # model = ModelOuts4Resnet(model, dims_out)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_res50'

    cfg.STRIDES = [8, 16, 32, 64]  # 特有参数下采样步距
    assert cfg.IMAGE_SIZE[0] % cfg.STRIDES[-1] == 0, 'cfg.IMAGE_SIZE=%s 与 cfg.STRIDES=%s 不能匹配(整除)' % (cfg.IMAGE_SIZE, cfg.STRIDES)
    cfg.SCALE_THRESHOLDS = [0, 49, 98, 196, 10000000000.0]  # 用于确保每一个特图预测相应大小的框,且一个GT只在一个层进行匹配

    model = KFcos(model, cfg)

    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        for name, param in model.backbone.named_parameters():
            param.requires_grad_(False)

    model, is_mgpu = model_device_init(model, device, id_gpu, cfg)

    # ------------------------自定义backbone完成-------------------------------
    pg = model.parameters()
    # optimizer = optim.Adam(pg, lr=cfg.LR0)
    optimizer = optim.SGD(pg, lr=cfg.LR0, momentum=0.9, weight_decay=5e-6)
    # 两次不上升，降低一半
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, verbose=True)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 80, 100], 0.1)
    start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device, is_mgpu=is_mgpu)

    model.cfg = cfg
    return model, optimizer, lr_scheduler, start_epoch


if __name__ == '__main__':
    cfg = CFG
    # 这里要改
    path_project_root = '/AI/temp/tmp_pycharm/DL/object_detection/fkeypoint'

    train = Train_1gpu(cfg, train_eval_set, init_model, path_project_root)
    train.f_run()

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
