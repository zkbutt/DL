import os
import socket

import torch
import numpy as np
from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_factory import  init_dataloader
from f_tools.datas.f_coco.coco_dataset import CustomCocoDataset
from f_tools.pic.enhance.f_data_pretreatment4np import cre_transform_resize4np, cre_transform_base4np


class DataLoader2:

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.set_init()  # 初始化方法
        if not cfg.USE_BASE4NP:
            self.data_transform = cre_transform_resize4np(cfg)  # 写死用这个
        else:
            self.data_transform = cre_transform_base4np(cfg)  # 写死用这个

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
        self.cfg.FILE_FIT_WEIGHT = os.path.join(self.cfg.PATH_SAVE_WEIGHT, self.cfg.FILE_NAME_WEIGHT)

        # json_file = open(os.path.join(self.cfg.PATH_DATA_ROOT, 'ids_classes.json'), 'r', encoding='utf-8')
        # self.cfg.IDS_CLASSES = json.load(json_file, encoding='utf-8')  # json key是字符

        if self.cfg.IS_FMAP_EVAL:
            f_recover_gt(self.cfg.PATH_EVAL_INFO + '/gt_info')
            # device = torch.device("cpu")

        self.cfg.DATA_NUM_WORKERS = min([os.cpu_count(), self.cfg.DATA_NUM_WORKERS])
        # self.cfg.DATA_NUM_WORKERS = 0  # 这里设置数据线程

        # 检查保存权重文件夹是否存在，不存在则创建
        if not os.path.exists(self.cfg.PATH_SAVE_WEIGHT):
            try:
                os.makedirs(self.cfg.PATH_SAVE_WEIGHT)
            except Exception as e:
                flog.error(' %s %s', self.cfg.PATH_SAVE_WEIGHT, e)

    def get_test_dataset(self):
        dataset_val = CustomCocoDataset(
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
        dataset_train = CustomCocoDataset(
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

        dataset_val = CustomCocoDataset(
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


def _cfg_base(cfg):
    pass


def cfg_voc(cfg, batch=40, image_size=(448, 448)):
    cfg.IMAGE_SIZE = image_size

    cfg.BATCH_SIZE = batch
    cfg.FORWARD_COUNT = 1  # 连续前传次数 accumulate = max(round(64 / CFG.BATCH_SIZE), 1)

    cfg.PRINT_FREQ = 20  # 400张图打印

    cfg.NUM_SAVE_INTERVAL = 10  # 第一次是19
    cfg.START_EVAL = 10  # 1第一轮

    cfg.IS_MOSAIC = False  # IS_MOSAIC 是主开关 直接拉伸
    cfg.IS_MOSAIC_KEEP_WH = False  # 是IS_MOSAIC_KEEP_WH 副形状
    cfg.IS_MOSAIC_FILL = True

    cfg.NUM_CLASSES = 20  # 这里要改
    cfg.NUM_KEYPOINTS = 0  # 关键点数, 0为没有 不能和 IS_MOSAIC 一起用
    # cfg.MODE_COCO_TRAIN = 'bbox'  # bbox segm keypoints caption
    # cfg.MODE_COCO_EVAL = 'bbox'  # bbox segm keypoints caption
    cfg.DATA_NUM_WORKERS = 4

    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_voc'
    cfg.PATH_TENSORBOARD = 'runs_voc'

    # cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/VOC2012'
    cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/VOC2007'

    # 训练
    cfg.PATH_COCO_TARGET_TRAIN = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_TRAIN = cfg.PATH_DATA_ROOT + '/train/JPEGImages'
    cfg.FILE_JSON_TRAIN = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_train_5011.json'

    # 验证
    cfg.PATH_COCO_TARGET_EVAL = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_EVAL = cfg.PATH_DATA_ROOT + '/val/JPEGImages'
    cfg.FILE_JSON_TEST = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_val_1980.json'

    # 暂无需测试集

    cfg.IS_KEEP_SCALE = False  # 数据处理保持长宽
    cfg.ANCS_SCALE = [[0.04, 0.056], [0.092, 0.104],
                      [0.122, 0.218], [0.254, 0.234],
                      [0.326, 0.462], [0.71, 0.572],
                      ]

    cfg.ANCHORS_CLIP = True  # 是否剔除超边界
    cfg.NUMS_ANC = [2, 2, 2]

    cfg.PIC_MEAN = (0.406, 0.456, 0.485)
    cfg.PIC_STD = (0.225, 0.224, 0.229)


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
    # cfg.ANCS_SCALE = [
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
    cfg.ANCS_SCALE = [
        [[0.255, 0.263], [0.354, 0.317]],
        [[0.389, 0.451], [0.49, 0.612]],
        [[0.678, 0.532], [0.804, 0.732]],
    ]
    cfg.PIC_MEAN = [0.45320560056079773, 0.43316440952455354, 0.3765994764105359]
    cfg.PIC_STD = [0.2196906701893696, 0.21533684244241802, 0.21516573455080967]


'''-----------------------type3---------------------------------'''


def cfg_type3(cfg, batch=40, image_size=(448, 448)):
    # _cfg_base(cfg)
    cfg.IMAGE_SIZE = image_size

    cfg.BATCH_SIZE = batch
    cfg.FORWARD_COUNT = 1  # 连续前传次数 accumulate = max(round(64 / CFG.BATCH_SIZE), 1)

    cfg.PRINT_FREQ = 10  # BATCH_SIZE * PRINT_FREQ 张图片

    cfg.NUM_SAVE_INTERVAL = 10  # 第一次是19
    cfg.START_EVAL = 10  # 1第一轮

    cfg.IS_MOSAIC = False  # IS_MOSAIC 是主开关 直接拉伸
    cfg.IS_MOSAIC_KEEP_WH = False  # 是IS_MOSAIC_KEEP_WH 副形状
    cfg.IS_MOSAIC_FILL = True

    cfg.NUM_CLASSES = 3
    cfg.NUM_KEYPOINTS = 0  # 关键点数, 0为没有 不能和 IS_MOSAIC 一起用
    cfg.DATA_NUM_WORKERS = 4

    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'type3'
    cfg.PATH_TENSORBOARD = 'runs_type3'

    # cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/VOC2012'
    cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/VOC2007'

    # 训练
    cfg.PATH_COCO_TARGET_TRAIN = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_TRAIN = cfg.PATH_DATA_ROOT + '/train/JPEGImages'
    cfg.FILE_JSON_TRAIN = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_type3_train_1066.json'

    # 验证
    cfg.PATH_COCO_TARGET_EVAL = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_EVAL = cfg.PATH_DATA_ROOT + '/val/JPEGImages'
    cfg.FILE_JSON_TEST = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_type3_val_413.json'

    # 测试
    cfg.PATH_COCO_TARGET_EVAL = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_EVAL = cfg.PATH_DATA_ROOT + '/val/JPEGImages'
    cfg.FILE_JSON_TEST = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_type3_test_637.json'

    cfg.IS_KEEP_SCALE = False  # 数据处理保持长宽
    # Accuracy: 73.32%  [3,3,2]
    cfg.ANCS_SCALE = [[0.136, 0.126],
                      [0.22, 0.282],
                      [0.392, 0.232],
                      [0.342, 0.432],
                      [0.548, 0.338],
                      [0.574, 0.562],
                      [0.82, 0.43],
                      [0.64696, 0.862],
                      [0.942, 0.662], ]

    cfg.ANCHORS_CLIP = True  # 是否剔除超边界
    cfg.NUMS_ANC = [3, 3, 3]

    cfg.PIC_MEAN = (0.406, 0.456, 0.485)
    cfg.PIC_STD = (0.225, 0.224, 0.229)


def cfg_type3_t(cfg, batch=3, image_size=(448, 448)):
    # _cfg_base(cfg)
    cfg.IMAGE_SIZE = image_size

    cfg.BATCH_SIZE = batch
    cfg.FORWARD_COUNT = 1  # 连续前传次数 accumulate = max(round(64 / CFG.BATCH_SIZE), 1)

    cfg.PRINT_FREQ = 10  # BATCH_SIZE * PRINT_FREQ 张图片

    cfg.NUM_SAVE_INTERVAL = 10  # 第一次是19
    cfg.START_EVAL = 10  # 1第一轮

    cfg.IS_MOSAIC = False  # IS_MOSAIC 是主开关 直接拉伸
    cfg.IS_MOSAIC_KEEP_WH = False  # 是IS_MOSAIC_KEEP_WH 副形状
    cfg.IS_MOSAIC_FILL = True

    cfg.NUM_CLASSES = 3
    cfg.NUM_KEYPOINTS = 0  # 关键点数, 0为没有 不能和 IS_MOSAIC 一起用
    cfg.DATA_NUM_WORKERS = 4

    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'type3'
    cfg.PATH_TENSORBOARD = 'runs_type3'

    # cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/VOC2012'
    cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/VOC2007'

    # 训练
    cfg.PATH_COCO_TARGET_TRAIN = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_TRAIN = cfg.PATH_DATA_ROOT + '/train/JPEGImages'
    cfg.FILE_JSON_TRAIN = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_type3_train_3.json'

    # 验证
    cfg.PATH_COCO_TARGET_EVAL = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_EVAL = cfg.PATH_DATA_ROOT + '/train/JPEGImages'
    cfg.FILE_JSON_TEST = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_type3_train_3.json'

    # 测试
    cfg.PATH_COCO_TARGET_EVAL = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_EVAL = cfg.PATH_DATA_ROOT + '/train/JPEGImages'
    cfg.FILE_JSON_TEST = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_type3_train_3.json'

    cfg.IS_KEEP_SCALE = False  # 数据处理保持长宽
    # Accuracy: 73.32%  [3,3,2]
    cfg.ANCS_SCALE = [[0.136, 0.126],
                      [0.22, 0.282],
                      [0.392, 0.232],
                      [0.342, 0.432],
                      [0.548, 0.338],
                      [0.574, 0.562],
                      [0.82, 0.43],
                      [0.64696, 0.862],
                      [0.942, 0.662], ]

    cfg.ANCHORS_CLIP = True  # 是否剔除超边界
    cfg.NUMS_ANC = [3, 3, 3]

    cfg.PIC_MEAN = (0.406, 0.456, 0.485)
    cfg.PIC_STD = (0.225, 0.224, 0.229)


def cfg_type4(cfg, batch=40, image_size=(448, 448)):
    # _cfg_base(cfg)
    cfg.IMAGE_SIZE = image_size

    cfg.BATCH_SIZE = batch
    cfg.FORWARD_COUNT = 1  # 连续前传次数 accumulate = max(round(64 / CFG.BATCH_SIZE), 1)

    cfg.PRINT_FREQ = 10  # BATCH_SIZE * PRINT_FREQ 张图片

    cfg.NUM_SAVE_INTERVAL = 10  # 第一次是19
    cfg.START_EVAL = 10  # 1第一轮

    cfg.IS_MOSAIC = False  # IS_MOSAIC 是主开关 直接拉伸
    cfg.IS_MOSAIC_KEEP_WH = False  # 是IS_MOSAIC_KEEP_WH 副形状
    cfg.IS_MOSAIC_FILL = True

    cfg.NUM_CLASSES = 4
    cfg.NUM_KEYPOINTS = 0  # 关键点数, 0为没有 不能和 IS_MOSAIC 一起用
    cfg.DATA_NUM_WORKERS = 4

    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'type4'
    cfg.PATH_TENSORBOARD = 'runs_type4'

    # cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/VOC2012'
    cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/VOC2007'

    # 训练
    cfg.PATH_COCO_TARGET_TRAIN = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_TRAIN = cfg.PATH_DATA_ROOT + '/train/JPEGImages'
    cfg.FILE_JSON_TRAIN = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_type4_train_994.json'

    # 验证
    cfg.PATH_COCO_TARGET_EVAL = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_EVAL = cfg.PATH_DATA_ROOT + '/val/JPEGImages'
    cfg.FILE_JSON_TEST = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_type4_val_369.json'

    # 测试
    cfg.PATH_COCO_TARGET_EVAL = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_EVAL = cfg.PATH_DATA_ROOT + '/val/JPEGImages'
    cfg.FILE_JSON_TEST = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_type4_test_550.json'

    cfg.IS_KEEP_SCALE = False  # 数据处理保持长宽
    # Accuracy: 73.32%  [3,3,2]
    cfg.ANCS_SCALE = [[0.028, 0.044],
                      [0.034, 0.084],
                      [0.056, 0.14],
                      [0.122, 0.092],
                      [0.1, 0.24],
                      [0.238, 0.178],
                      [0.297, 0.362],
                      [0.671, 0.336],
                      [0.836, 0.624], ]

    cfg.ANCHORS_CLIP = True  # 是否剔除超边界
    cfg.NUMS_ANC = [3, 3, 3]

    cfg.PIC_MEAN = (0.406, 0.456, 0.485)
    cfg.PIC_STD = (0.225, 0.224, 0.229)


if __name__ == '__main__':
    from object_detection.z_yolov3.CONFIG_YOLO3 import CFG

    cfg = CFG()
    cfg_type3(cfg)
    print(len(cfg.ANCS_SCALE))
    print(np.array(cfg.ANCS_SCALE).shape)  # [3,3,2]
    print(cfg.NUM_ANC)
