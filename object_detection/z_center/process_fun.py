import torch
from torch import optim
from torchvision import models

from f_pytorch.tools_model.f_layer_get import ModelOuts4Mobilenet_v2, ModelOut4Mobilenet_v2
from f_pytorch.tools_model.model_look import f_look_model
from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_factory import MapDataSet, load_od4voc
from f_tools.datas.f_coco.convert_data.coco_dataset import CocoDataset
from f_tools.datas.f_map.map_go import f_do_fmap
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_gpu.f_gpu_api import model_device_init
from f_tools.fits.fitting.f_fit_eval_base import f_train_one_epoch4, f_evaluate4coco2, f_evaluate4fmap
from f_tools.pic.enhance.f_data_pretreatment import Compose, ColorJitter, ToTensor, RandomHorizontalFlip4TS, \
    Normalization4TS, Resize, ResizeKeep
from object_detection.z_center.nets.net_center import CenterNet


def cre_data_transform(cfg):
    if cfg.IS_MOSAIC:
        data_transform = {
            "train": Compose([
                # ResizeKeep(cfg.IMAGE_SIZE),  # (h,w)
                # Resize(cfg.IMAGE_SIZE),
                ColorJitter(),
                ToTensor(),
                RandomHorizontalFlip4TS(0.5),
                Normalization4TS(),
            ], cfg),
        }
    else:
        data_transform = {
            "train": Compose([
                # ResizeKeep(cfg.IMAGE_SIZE),  # 这个有边界需要修正
                Resize(cfg.IMAGE_SIZE),
                ColorJitter(),
                ToTensor(),
                RandomHorizontalFlip4TS(0.5),
                Normalization4TS(),
            ], cfg),
        }
    data_transform["val"] = Compose([
        # ResizeKeep(cfg.IMAGE_SIZE),  # (h,w)
        Resize(cfg.IMAGE_SIZE),
        ToTensor(),
        Normalization4TS(),
    ], cfg)

    return data_transform


def init_model(cfg, device, id_gpu=None):
    model = models.mobilenet_v2(pretrained=True)
    model = ModelOut4Mobilenet_v2(model)
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'm2'

    model = CenterNet(cfg=cfg, backbone=model, num_classes=cfg.NUM_CLASSES, dim_in_backbone=model.dim_out)
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
    lr0 = cfg.LR
    lrf = lr0 / 100
    # 权重衰减(如L2惩罚)(默认: 0)
    optimizer = optim.Adam(pg, lr0, weight_decay=5e-4)
    # 两次不上升，降低一半
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=3, verbose=True)
    lr_scheduler = None
    start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, None, lr_scheduler, device, is_mgpu=is_mgpu)
    # start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device, is_mgpu=is_mgpu)

    model.cfg = cfg
    return model, optimizer, lr_scheduler, start_epoch


def data_loader(cfg, is_mgpu=False, ids2classes=None):
    data_transform = cre_data_transform(cfg)

    _res = load_od4voc(cfg, data_transform, is_mgpu, ids2classes)
    loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler = _res

    # 返回数据已预处理 返回np(batch,(3,640,640))  , np(batch,(x个选框,15维))
    # iter(data_loader).__next__()
    return loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler
