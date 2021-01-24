import os
import sys

'''用户命令行启动'''
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from f_pytorch.tools_model.f_layer_get import ModelOuts4DarkNet
from object_detection.z_yolov2.nets.net_yolov2 import Yolo_v2
from object_detection.z_yolov2.CONFIG_YOLOV2 import CFG
from f_pytorch.tools_model.backbones.darknet import darknet19
from torch import optim
from f_tools.datas.data_loader import cfg_raccoon, cfg_type, cfg_type2, DataLoader2

from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_gpu.f_gpu_api import model_device_init, mgpu_process0_init

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_fit_fun import init_od, base_set, train_eval4od, fdatas_l2, show_train_info

'''

linux用这个   python /AI/temp/tmp_pycharm/DL/object_detection/z_yolov2/train_yolov2.py
tensorboard --host=192.168.0.199 --logdir=/AI/temp/tmp_pycharm/DL/object_detection/z_yolov2/runs_type3

'''


def init_model(cfg, device, id_gpu=None):
    model = darknet19(pretrained=True)
    model = ModelOuts4DarkNet(model)
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_dark19'

    model = Yolo_v2(backbone=model, cfg=cfg)

    # f_look_model(model, input=(1, 3, *cfg.IMAGE_SIZE))

    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        for name, param in model.backbone.named_parameters():
            param.requires_grad_(False)
        # 除最后的全连接层外，其他权重全部冻结
        # if "fc" not in name:
        #     param.requires_grad_(False)
    # if id_gpu is not None:
    #     cfg.LR0 = cfg.LR0 / 2

    model, is_mgpu = model_device_init(model, device, id_gpu, cfg)

    # ------------------------自定义backbone完成-------------------------------
    pg = model.parameters()
    optimizer = optim.Adam(pg, cfg.LR0)
    # 两次不上升，降低一半
    # lr_scheduler = None
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 90], 0.1)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=1, verbose=True)
    # start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, None, lr_scheduler, device, is_mgpu=is_mgpu)
    start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device, is_mgpu=is_mgpu)

    model.cfg = cfg
    return model, optimizer, lr_scheduler, start_epoch


def train_eval_set(cfg):
    cfg.FILE_NAME_WEIGHT = '123' + '.pth'  # 重新开始

    # batch = 20  # raccoon
    batch = 32  # type
    # batch = 3  # type
    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        batch *= 3
        cfg.IS_COCO_EVAL = False

    # batch = 10  # type
    size = (416, 416)  # type
    cfg_type2(cfg, batch=batch, image_size=size)  # 加载数据基础参数
    # anc重写
    cfg.ANC_SIZE = [[0.074, 0.074], [0.162, 0.146], [0.314, 0.3], [0.452, 0.506], [0.729, 0.635]]
    cfg.NUM_ANC = len(cfg.ANC_SIZE)

    # cfg_raccoon(cfg, batch=batch, image_size=size)  # 加载数据基础参数
    cfg.START_EVAL = 10  # cfg.START_EVAL=10 and EVAL_INTERVAL=3 实际是12
    cfg.END_EVAL = 100  # 结束间隙验证
    cfg.EVAL_INTERVAL = 3  #
    # cfg.NUM_SAVE_INTERVAL = 100
    cfg.loss_args = {
        's_match': 'log_g',  # 'log' 'whoned' 'log_g'
        's_conf': 'mse',  # 'mse' 'foc'
    }
    print(cfg.loss_args)

    cfg.arg_focalloss_alpha = 0.75
    cfg.IS_MIXTURE_FIX = True

    # type3 dark19
    cfg.FILE_NAME_WEIGHT = '123' + '.pth'  # conf-0.01 nms-0.5
    cfg.MAPS_VAL = [0.724, 0.463]

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

    cfg.PATH_PROJECT_ROOT = cfg.PATH_HOST + '/AI/temp/tmp_pycharm/DL/object_detection/z_yolov2'  # 这个要改

    '''---------------数据加载及处理--------------'''
    data_loader = DataLoader2(cfg)
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
