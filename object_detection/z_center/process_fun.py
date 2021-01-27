import torch
from torch import optim
from torchvision import models

from f_pytorch.tools_model.f_layer_get import ModelOuts4Mobilenet_v2, ModelOut4Mobilenet_v2, ModelOut4Resnet18
from f_pytorch.tools_model.model_look import f_look_tw
from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_factory import MapDataSet, load_od4voc, init_dataloader
from f_tools.datas.f_coco.convert_data.coco_dataset import CocoDataset, CustomCocoDataset
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




def data_loader4voc(cfg, is_mgpu=False, ids2classes=None):
    cfg.NUM_CLASSES = 20
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_voc'
    cfg.PATH_TENSORBOARD = 'runs_voc'

    data_transform = cre_data_transform(cfg)

    _res = load_od4voc(cfg, data_transform, is_mgpu, ids2classes)

    loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler = _res
    # 返回数据已预处理 返回np(batch,(3,640,640))  , np(batch,(x个选框,15维))
    # iter(data_loader).__next__()
    return loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler


def data_loader4widerface(cfg, is_mgpu=False):
    cfg.NUM_CLASSES = 1
    cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/widerface/'
    cfg.PATH_COCO_TARGET_TRAIN = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_TRAIN = cfg.PATH_DATA_ROOT + '/coco/images/train2017'
    cfg.PATH_COCO_TARGET_EVAL = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_EVAL = cfg.PATH_DATA_ROOT + '/coco/images/val2017'
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_widerface'
    cfg.PATH_TENSORBOARD = 'runs_widerface'
    cfg.cfg.DATA_NUM_WORKERS = 5

    data_transform = cre_data_transform(cfg)

    # 返回数据已预处理 返回np(batch,(3,640,640))  , np(batch,(x个选框,15维))
    if cfg.NUM_KEYPOINTS > 0:
        mode = 'keypoints'  # keypoints  与 IS_MOSAIC 不能一起用
        cfg.IS_MOSAIC = False
    else:
        mode = 'bbox'
    dataset_train = CocoDataset(
        path_coco_target=cfg.PATH_COCO_TARGET_TRAIN,
        path_img=cfg.PATH_IMG_TRAIN,
        mode=mode,
        data_type='train2017',
        transform=data_transform['train'],
        is_mosaic=cfg.IS_MOSAIC,
        is_mosaic_keep_wh=cfg.IS_MOSAIC_KEEP_WH,
        is_mosaic_fill=cfg.IS_MOSAIC_FILL,
        is_debug=cfg.DEBUG,
        cfg=cfg
    )
    dataset_val = CocoDataset(
        path_coco_target=cfg.PATH_COCO_TARGET_EVAL,
        path_img=cfg.PATH_IMG_EVAL,
        # mode='keypoints',
        mode='bbox',
        data_type='val2017',
        transform=data_transform['val'],
        is_debug=cfg.DEBUG,
        cfg=cfg
    )
    _res = init_dataloader(cfg, dataset_train, dataset_val, is_mgpu)
    loader_train, loader_val_coco, train_sampler, eval_sampler = _res
    loader_val_fmap = None
    return loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler


def data_loader4raccoon200(cfg, is_mgpu=False):
    cfg.NUM_CLASSES = 1
    cfg.DATA_NUM_WORKERS = 2
    # cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/widerface/'
    # cfg.PATH_COCO_TARGET_TRAIN = cfg.PATH_DATA_ROOT + '/coco/annotations'
    # cfg.PATH_IMG_TRAIN = cfg.PATH_DATA_ROOT + '/coco/images/train2017'
    # cfg.PATH_COCO_TARGET_EVAL = cfg.PATH_DATA_ROOT + '/coco/annotations'
    # cfg.PATH_IMG_EVAL = cfg.PATH_DATA_ROOT + '/coco/images/val2017'
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_raccoon200'
    cfg.PATH_TENSORBOARD = 'runs_raccoon200'

    data_transform = cre_data_transform(cfg)

    mode = 'bbox'  # bbox segm keypoints caption
    # 返回数据已预处理 返回np(batch,(3,640,640))  , np(batch,(x个选框,15维))
    file_json = cfg.PATH_HOST + r'/AI/datas/raccoon200/coco/annotations/instances_train2017.json'
    path_img = cfg.PATH_HOST + r'/AI/datas/raccoon200/VOCdevkit/JPEGImages'
    dataset_train = CustomCocoDataset(
        file_json=file_json,
        path_img=path_img,
        mode=mode,
        transform=data_transform['train'],
        is_mosaic=cfg.IS_MOSAIC,
        is_mosaic_keep_wh=cfg.IS_MOSAIC_KEEP_WH,
        is_mosaic_fill=cfg.IS_MOSAIC_FILL,
        is_debug=cfg.DEBUG,
        cfg=cfg
    )

    file_json = cfg.PATH_HOST + r'/AI/datas/raccoon200/coco/annotations/instances_val2017.json'
    dataset_val = CustomCocoDataset(
        file_json=file_json,
        path_img=path_img,
        mode=mode,
        transform=data_transform['val'],
        is_mosaic=cfg.IS_MOSAIC,
        is_mosaic_keep_wh=cfg.IS_MOSAIC_KEEP_WH,
        is_mosaic_fill=cfg.IS_MOSAIC_FILL,
        is_debug=cfg.DEBUG,
        cfg=cfg
    )
    _res = init_dataloader(cfg, dataset_train, dataset_val, is_mgpu)
    loader_train, loader_val_coco, train_sampler, eval_sampler = _res
    loader_val_fmap = None
    return loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler
