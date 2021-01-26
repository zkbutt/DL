import json
import os
import socket

import torch
import numpy as np
from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_factory import load_od4voc, init_dataloader
from f_tools.datas.f_coco.convert_data.coco_dataset import CocoDataset, CustomCocoDataset, CustomCocoDataset4cv
from f_tools.datas.f_map.convert_data.extra.intersect_gt_and_dr import f_recover_gt
from f_tools.pic.enhance.f_data_pretreatment4np import cre_transform_resize4np
from f_tools.pic.enhance.f_data_pretreatment4pil import cre_transform_resize4pil


# class DataLoader:
#
#     def __init__(self, cfg) -> None:
#         super().__init__()
#         self.cfg = cfg
#
#         self.set_init()  # 初始化方法
#         self.data_transform = cre_transform_resize4pil(cfg)  # 写死用这个
#
#     def set_init(self):
#         host_name = socket.gethostname()
#         if host_name == 'Feadre-NB':
#             self.cfg.PATH_HOST = 'M:'
#             # raise Exception('当前主机: %s 及主数据路径: %s ' % (host_name, cfg.PATH_HOST))
#         elif host_name == 'e2680v2':
#             self.cfg.PATH_HOST = ''
#
#         import platform
#
#         sysstr = platform.system()
#         print('当前系统为:', sysstr)
#         if sysstr == 'Windows':  # 'Linux'
#             torch.backends.cudnn.enabled = False
#
#     def set_tail(self):
#         self.cfg.PATH_SAVE_WEIGHT = self.cfg.PATH_HOST + '/AI/weights/feadre'
#         self.cfg.FILE_FIT_WEIGHT = self.cfg.PATH_SAVE_WEIGHT + '/' + self.cfg.FILE_NAME_WEIGHT
#
#         # json_file = open(os.path.join(self.cfg.PATH_DATA_ROOT, 'ids_classes.json'), 'r', encoding='utf-8')
#         # self.cfg.IDS_CLASSES = json.load(json_file, encoding='utf-8')  # json key是字符
#
#         if self.cfg.IS_FMAP_EVAL:
#             f_recover_gt(self.cfg.PATH_EVAL_INFO + '/gt_info')
#             # device = torch.device("cpu")
#
#         self.cfg.DATA_NUM_WORKERS = min([os.cpu_count(), self.cfg.DATA_NUM_WORKERS])
#
#         # 检查保存权重文件夹是否存在，不存在则创建
#         if not os.path.exists(self.cfg.PATH_SAVE_WEIGHT):
#             try:
#                 os.makedirs(self.cfg.PATH_SAVE_WEIGHT)
#             except Exception as e:
#                 flog.error(' %s %s', self.cfg.PATH_SAVE_WEIGHT, e)
#
#     def get_test_dataset(self):
#         dataset_val = CustomCocoDataset(
#             file_json=self.cfg.FILE_JSON_TEST,
#             path_img=self.cfg.PATH_IMG_EVAL,
#             mode=self.cfg.MODE_COCO_EVAL,
#             transform=None,
#             is_mosaic=False,
#             is_mosaic_keep_wh=self.cfg.IS_MOSAIC_KEEP_WH,
#             is_mosaic_fill=self.cfg.IS_MOSAIC_FILL,
#             is_debug=self.cfg.DEBUG,
#             cfg=self.cfg
#         )
#
#         self.set_tail()  # 尾设置
#         return dataset_val
#
#     def get_train_eval_datas(self, is_mgpu):
#         dataset_train = CustomCocoDataset(
#             file_json=self.cfg.FILE_JSON_TRAIN,
#             path_img=self.cfg.PATH_IMG_TRAIN,
#             mode=self.cfg.MODE_COCO_TRAIN,
#             transform=self.data_transform['train'],
#             is_mosaic=self.cfg.IS_MOSAIC,
#             is_mosaic_keep_wh=self.cfg.IS_MOSAIC_KEEP_WH,
#             is_mosaic_fill=self.cfg.IS_MOSAIC_FILL,
#             is_debug=self.cfg.DEBUG,
#             cfg=self.cfg
#         )
#
#         dataset_val = CustomCocoDataset(
#             file_json=self.cfg.FILE_JSON_TEST,
#             path_img=self.cfg.PATH_IMG_EVAL,
#             mode=self.cfg.MODE_COCO_EVAL,
#             transform=self.data_transform['val'],
#             is_mosaic=False,
#             is_mosaic_keep_wh=False,
#             is_mosaic_fill=False,
#             is_debug=self.cfg.DEBUG,
#             cfg=self.cfg
#         )
#         _res = init_dataloader(self.cfg, dataset_train, dataset_val, is_mgpu, use_mgpu_eval=self.cfg.USE_MGPU_EVAL)
#         loader_train, loader_val_coco, train_sampler, eval_sampler = _res
#         loader_val_fmap = None
#
#         self.set_tail()  # 尾设置
#         return loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler


class DataLoader2:

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.set_init()  # 初始化方法
        self.data_transform = cre_transform_resize4np(cfg)  # 写死用这个

    def set_init(self):
        host_name = socket.gethostname()
        if host_name == 'Feadre-NB':
            self.cfg.PATH_HOST = 'M:'
            # raise Exception('当前主机: %s 及主数据路径: %s ' % (host_name, cfg.PATH_HOST))
        elif host_name == 'e2680v2':
            self.cfg.PATH_HOST = ''

        import platform

        sysstr = platform.system()
        print('当前系统为:', sysstr)
        if sysstr == 'Windows':  # 'Linux'
            torch.backends.cudnn.enabled = False

    def set_tail(self):
        self.cfg.PATH_SAVE_WEIGHT = self.cfg.PATH_HOST + '/AI/weights/feadre'
        self.cfg.FILE_FIT_WEIGHT = self.cfg.PATH_SAVE_WEIGHT + '/' + self.cfg.FILE_NAME_WEIGHT

        # json_file = open(os.path.join(self.cfg.PATH_DATA_ROOT, 'ids_classes.json'), 'r', encoding='utf-8')
        # self.cfg.IDS_CLASSES = json.load(json_file, encoding='utf-8')  # json key是字符

        if self.cfg.IS_FMAP_EVAL:
            f_recover_gt(self.cfg.PATH_EVAL_INFO + '/gt_info')
            # device = torch.device("cpu")

        self.cfg.DATA_NUM_WORKERS = min([os.cpu_count(), self.cfg.DATA_NUM_WORKERS])

        # 检查保存权重文件夹是否存在，不存在则创建
        if not os.path.exists(self.cfg.PATH_SAVE_WEIGHT):
            try:
                os.makedirs(self.cfg.PATH_SAVE_WEIGHT)
            except Exception as e:
                flog.error(' %s %s', self.cfg.PATH_SAVE_WEIGHT, e)

    def get_test_dataset(self):
        dataset_val = CustomCocoDataset4cv(
            file_json=self.cfg.FILE_JSON_TEST,
            path_img=self.cfg.PATH_IMG_EVAL,
            mode=self.cfg.MODE_COCO_EVAL,
            transform=None,
            is_mosaic=False,
            is_mosaic_keep_wh=self.cfg.IS_MOSAIC_KEEP_WH,
            is_mosaic_fill=self.cfg.IS_MOSAIC_FILL,
            is_debug=self.cfg.DEBUG,
            cfg=self.cfg
        )

        self.set_tail()  # 尾设置
        return dataset_val

    def get_train_eval_datas(self, is_mgpu):
        dataset_train = CustomCocoDataset4cv(
            file_json=self.cfg.FILE_JSON_TRAIN,
            path_img=self.cfg.PATH_IMG_TRAIN,
            mode=self.cfg.MODE_COCO_TRAIN,
            transform=self.data_transform['train'],
            is_mosaic=self.cfg.IS_MOSAIC,
            is_mosaic_keep_wh=self.cfg.IS_MOSAIC_KEEP_WH,
            is_mosaic_fill=self.cfg.IS_MOSAIC_FILL,
            is_debug=self.cfg.DEBUG,
            cfg=self.cfg
        )

        dataset_val = CustomCocoDataset4cv(
            file_json=self.cfg.FILE_JSON_TEST,
            path_img=self.cfg.PATH_IMG_EVAL,
            mode=self.cfg.MODE_COCO_EVAL,
            transform=self.data_transform['val'],
            is_mosaic=False,
            is_mosaic_keep_wh=False,
            is_mosaic_fill=False,
            is_debug=self.cfg.DEBUG,
            cfg=self.cfg
        )
        _res = init_dataloader(self.cfg, dataset_train, dataset_val, is_mgpu, use_mgpu_eval=self.cfg.USE_MGPU_EVAL)
        loader_train, loader_val_coco, train_sampler, eval_sampler = _res
        loader_val_fmap = None

        self.set_tail()  # 尾设置
        return loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler


'''-----------------------voc---------------------------------'''


def cfg_voc(cfg):
    # cfg.BATCH_SIZE = 16
    cfg.FORWARD_COUNT = 1  # 连续前传次数 accumulate = max(round(64 / CFG.BATCH_SIZE), 1)

    cfg.PRINT_FREQ = 20  # 400张图打印

    cfg.IMAGE_SIZE = (512, 512)
    cfg.NUM_SAVE_INTERVAL = 1  # epoch+1
    cfg.START_EVAL = 3  # epoch

    cfg.IS_MOSAIC = False  # IS_MOSAIC 是主开关 直接拉伸
    cfg.IS_MOSAIC_KEEP_WH = False  # 是IS_MOSAIC_KEEP_WH 副形状
    cfg.IS_MOSAIC_FILL = True

    cfg.NUM_CLASSES = 20  # 这里要改
    cfg.NUM_KEYPOINTS = 0  # 关键点数, 0为没有 不能和 IS_MOSAIC 一起用
    cfg.MODE_COCO_TRAIN = 'bbox'  # bbox segm keypoints caption
    cfg.MODE_COCO_EVAL = 'bbox'  # bbox segm keypoints caption
    cfg.DATA_NUM_WORKERS = 4

    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_voc'
    cfg.PATH_TENSORBOARD = 'runs_voc'

    # cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/VOC2012'
    cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/VOC2007'

    # 训练
    cfg.PATH_COCO_TARGET_TRAIN = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_TRAIN = cfg.PATH_DATA_ROOT + '/VOCdevkit/JPEGImages'
    cfg.FILE_JSON_TRAIN = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_train.json'

    # 验证
    cfg.PATH_COCO_TARGET_EVAL = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_EVAL = cfg.PATH_IMG_TRAIN
    cfg.FILE_JSON_TEST = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_val.json'

    cfg.IS_KEEP_SCALE = False  # 数据处理保持长宽
    cfg.ANC_SCALE = [
        [[0.04, 0.056], [0.092, 0.104]],
        [[0.122, 0.218], [0.254, 0.234]],
        [[0.326, 0.462], [0.71, 0.572]],
    ]

    # cfg.PIC_MEAN = [0.45320560056079773, 0.43316440952455354, 0.3765994764105359]
    # cfg.PIC_STD = [0.2196906701893696, 0.21533684244241802, 0.21516573455080967]


'''-----------------------widerface---------------------------------'''


def cfg_widerface(cfg):
    # cfg.BATCH_SIZE = 16
    cfg.FORWARD_COUNT = 1  # 连续前传次数 accumulate = max(round(64 / CFG.BATCH_SIZE), 1)

    cfg.PRINT_FREQ = 40  # 400张图打印

    cfg.IMAGE_SIZE = (512, 512)
    cfg.NUM_SAVE_INTERVAL = 1  # epoch+1
    cfg.START_EVAL = 3  # epoch

    cfg.IS_MOSAIC = False  # IS_MOSAIC 是主开关 直接拉伸
    cfg.IS_MOSAIC_KEEP_WH = False  # 是IS_MOSAIC_KEEP_WH 副形状
    cfg.IS_MOSAIC_FILL = True

    cfg.NUM_CLASSES = 1
    cfg.NUM_KEYPOINTS = 0  # 关键点数, 0为没有 不能和 IS_MOSAIC 一起用
    cfg.MODE_COCO_TRAIN = 'bbox'  # bbox segm keypoints caption
    cfg.MODE_COCO_EVAL = 'bbox'  # bbox segm keypoints caption
    cfg.DATA_NUM_WORKERS = 5

    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_widerface'
    cfg.PATH_TENSORBOARD = 'runs_widerface'

    cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/widerface'

    # 训练
    cfg.PATH_COCO_TARGET_TRAIN = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_TRAIN = cfg.PATH_DATA_ROOT + '/VOCdevkit/JPEGImages'
    cfg.FILE_JSON_TRAIN = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_train2017.json'

    # 验证
    cfg.PATH_COCO_TARGET_EVAL = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_EVAL = cfg.PATH_IMG_TRAIN
    cfg.FILE_JSON_TEST = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_val2017.json'

    cfg.IS_KEEP_SCALE = False  # 数据处理保持长宽
    # cfg.ANC_SCALE = [
    #     [[0.255, 0.263], [0.354, 0.317]],
    #     [[0.389, 0.451], [0.49, 0.612]],
    #     [[0.678, 0.532], [0.804, 0.732]],
    # ]
    # cfg.PIC_MEAN = [0.45320560056079773, 0.43316440952455354, 0.3765994764105359]
    # cfg.PIC_STD = [0.2196906701893696, 0.21533684244241802, 0.21516573455080967]


'''-----------------------raccoon---------------------------------'''


def cfg_raccoon(cfg, batch=40, image_size=(448, 448)):
    cfg.BATCH_SIZE = batch
    cfg.FORWARD_COUNT = 1  # 连续前传次数 accumulate = max(round(64 / CFG.BATCH_SIZE), 1)

    cfg.PRINT_FREQ = 10  # 400张图打印

    cfg.IMAGE_SIZE = image_size
    cfg.NUM_SAVE_INTERVAL = 30  # epoch+1
    cfg.START_EVAL = 1  # 从 cfg.START_EVAL + 1 开始，实际需要+1

    cfg.IS_MOSAIC = False  # IS_MOSAIC 是主开关 直接拉伸
    cfg.IS_MOSAIC_KEEP_WH = False  # 是IS_MOSAIC_KEEP_WH 副形状
    cfg.IS_MOSAIC_FILL = True

    cfg.NUM_CLASSES = 1
    cfg.NUM_KEYPOINTS = 0  # 关键点数, 0为没有 不能和 IS_MOSAIC 一起用
    cfg.DATA_NUM_WORKERS = 2

    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'raccoon200'
    cfg.PATH_TENSORBOARD = 'runs_rac'

    cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/raccoon200'

    # 训练
    cfg.PATH_COCO_TARGET_TRAIN = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_TRAIN = cfg.PATH_DATA_ROOT + '/VOCdevkit/JPEGImages'
    cfg.FILE_JSON_TRAIN = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_train2017.json'

    # 验证
    cfg.PATH_COCO_TARGET_EVAL = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_EVAL = cfg.PATH_IMG_TRAIN
    cfg.FILE_JSON_TEST = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_val2017.json'

    cfg.IS_KEEP_SCALE = False  # 数据处理保持长宽
    cfg.ANC_SCALE = [
        [[0.255, 0.263], [0.354, 0.317]],
        [[0.389, 0.451], [0.49, 0.612]],
        [[0.678, 0.532], [0.804, 0.732]],
    ]
    cfg.PIC_MEAN = [0.45320560056079773, 0.43316440952455354, 0.3765994764105359]
    cfg.PIC_STD = [0.2196906701893696, 0.21533684244241802, 0.21516573455080967]


'''-----------------------type3---------------------------------'''


def cfg_type2(cfg, batch=40, image_size=(448, 448)):
    cfg.BATCH_SIZE = batch
    cfg.FORWARD_COUNT = 1  # 连续前传次数 accumulate = max(round(64 / CFG.BATCH_SIZE), 1)

    cfg.PRINT_FREQ = 5  # BATCH_SIZE * PRINT_FREQ 张图片

    cfg.IMAGE_SIZE = image_size
    cfg.NUM_SAVE_INTERVAL = 10  # 第一次是19
    cfg.START_EVAL = 10  # 1第一轮

    cfg.IS_MOSAIC = False  # IS_MOSAIC 是主开关 直接拉伸
    cfg.IS_MOSAIC_KEEP_WH = False  # 是IS_MOSAIC_KEEP_WH 副形状
    cfg.IS_MOSAIC_FILL = True

    cfg.NUM_CLASSES = 3
    cfg.NUM_KEYPOINTS = 0  # 关键点数, 0为没有 不能和 IS_MOSAIC 一起用
    cfg.DATA_NUM_WORKERS = 2

    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'type3'
    cfg.PATH_TENSORBOARD = 'runs_type3'

    # cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/VOC2012'
    cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/VOC2007'

    # 训练
    cfg.PATH_COCO_TARGET_TRAIN = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_TRAIN = cfg.PATH_DATA_ROOT + '/train/JPEGImages'
    cfg.FILE_JSON_TRAIN = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_type3_train_1096.json'

    # 验证
    cfg.PATH_COCO_TARGET_EVAL = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_EVAL = cfg.PATH_DATA_ROOT + '/val/JPEGImages'
    cfg.FILE_JSON_TEST = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_type3_val_416.json'

    # 测试
    cfg.PATH_COCO_TARGET_EVAL = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_EVAL = cfg.PATH_DATA_ROOT + '/val/JPEGImages'
    cfg.FILE_JSON_TEST = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_type3_test_621.json'

    cfg.IS_KEEP_SCALE = False  # 数据处理保持长宽
    # cfg.ANC_SCALE = [
    #     [[0.078, 0.076], [0.166, 0.164]],
    #     [[0.374, 0.254], [0.288, 0.424]],
    #     [[0.574, 0.46], [0.698, 0.668]],
    # ]
    cfg.ANC_SCALE = [
        [[0.05, 0.045], [0.078, 0.084], [0.142, 0.106]],
        [[0.185, 0.2], [0.384, 0.27], [0.284, 0.426]],
        [[0.642, 0.436], [0.48, 0.628], [0.77, 0.664]],
    ]
    cfg.ANCHORS_CLIP = True  # 是否剔除超边界
    cfg.NUMS_ANC = [3, 3, 3]
    cfg.NUM_ANC = np.array(cfg.NUMS_ANC).prod()

    cfg.PIC_MEAN = (0.406, 0.456, 0.485)
    cfg.PIC_STD = (0.225, 0.224, 0.229)
