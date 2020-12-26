import json
import os
import socket

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_factory import load_od4voc, init_dataloader
from f_tools.datas.f_coco.convert_data.coco_dataset import CocoDataset, CustomCocoDataset
from f_tools.datas.f_map.convert_data.extra.intersect_gt_and_dr import f_recover_gt
from f_tools.pic.enhance.f_data_pretreatment import Compose, ColorJitter, ToTensor, RandomHorizontalFlip4TS, \
    Normalization4TS, Resize, ResizeKeep


def cre_transform4resize(cfg):
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


def base_set(cfg):
    cfg.PATH_HOST = 'M:'

    host_name = socket.gethostname()
    if host_name == 'Feadre-NB':
        cfg.PATH_HOST = 'M:'
        # raise Exception('当前主机: %s 及主数据路径: %s ' % (host_name, cfg.PATH_HOST))
    elif host_name == 'e2680v2':
        cfg.PATH_HOST = ''


def custom_set(cfg):
    cfg.PATH_SAVE_WEIGHT = cfg.PATH_HOST + '/AI/weights/feadre'
    cfg.FILE_FIT_WEIGHT = cfg.PATH_SAVE_WEIGHT + '/' + cfg.FILE_NAME_WEIGHT
    json_file = open(os.path.join(cfg.PATH_DATA_ROOT, 'ids_classes.json'), 'r', encoding='utf-8')
    cfg.ids2classes = json.load(json_file, encoding='utf-8')  # json key是字符

    if cfg.IS_FMAP_EVAL:
        f_recover_gt(cfg.PATH_EVAL_INFO + '/gt_info')
        # device = torch.device("cpu")

    cfg.DATA_NUM_WORKERS = min([os.cpu_count(), cfg.DATA_NUM_WORKERS])

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(cfg.PATH_SAVE_WEIGHT):
        try:
            os.makedirs(cfg.PATH_SAVE_WEIGHT)
        except Exception as e:
            flog.error(' %s %s', cfg.PATH_SAVE_WEIGHT, e)


def fload_voc(cfg, is_mgpu):
    base_set(cfg)

    '''样本及预处理'''
    cfg.BATCH_SIZE = 1  # batch过小需要设置连续前传
    cfg.FORWARD_COUNT = 1  # 连续前传次数 accumulate = max(round(64 / CFG.BATCH_SIZE), 1)
    cfg.PRINT_FREQ = 40  # 400张图打印
    cfg.IMAGE_SIZE = (512, 512)  # wh 预处理 统一尺寸
    cfg.NUM_SAVE = 1

    cfg.IS_KEEP_SCALE = True  # 数据处理保持长宽
    cfg.IS_MOSAIC = False
    cfg.IS_MOSAIC_KEEP_WH = False  # IS_MOSAIC 是主开关 直接拉伸
    cfg.IS_MOSAIC_FILL = True  # IS_MOSAIC 使用 是IS_MOSAIC_KEEP_WH 副形状

    cfg.DATA_NUM_WORKERS = 8
    cfg.NUM_KEYPOINTS = 0  # 关键点数, 0为没有 不能和 IS_MOSAIC 一起用
    # PRINT_FREQ = int(400 / BATCH_SIZE)  # 400张图打印

    # IMAGE_SIZE = (1024, 1024)  # wh 预处理 统一尺寸
    cfg.NUM_CLASSES = 20  # 模型分类数 人脸只有1 0 影响类别输出   -----这个要根据样本改----
    cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/VOC2012'
    # 训练
    cfg.PATH_COCO_TARGET_TRAIN = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_TRAIN = cfg.PATH_DATA_ROOT + '/trainval/JPEGImages'
    # 验证
    cfg.PATH_COCO_TARGET_EVAL = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_EVAL = cfg.PATH_DATA_ROOT + '/test/JPEGImages'
    # fmap
    cfg.PATH_EVAL_IMGS = cfg.PATH_HOST + r'/AI/datas/VOC2012/test/JPEGImages'
    cfg.PATH_EVAL_INFO = cfg.PATH_HOST + r'/AI/datas/VOC2012/f_map'  # dt会自动创建

    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_voc'
    cfg.PATH_TENSORBOARD = 'runs_voc'

    data_transform = cre_transform4resize(cfg)

    _res = load_od4voc(cfg, data_transform, is_mgpu, cfg.ids2classes)

    loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler = _res
    # 返回数据已预处理 返回np(batch,(3,640,640))  , np(batch,(x个选框,15维))
    # iter(data_loader).__next__()

    custom_set(cfg)
    return loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler


def fload_widerface(cfg, is_mgpu):
    base_set(cfg)

    cfg.BATCH_SIZE = 1  # batch过小需要设置连续前传
    cfg.FORWARD_COUNT = 1  # 连续前传次数 accumulate = max(round(64 / CFG.BATCH_SIZE), 1)
    cfg.PRINT_FREQ = 40  # 400张图打印
    cfg.IMAGE_SIZE = (512, 512)  # wh 预处理 统一尺寸
    cfg.NUM_SAVE = 1

    cfg.NUM_KEYPOINTS = 0  # 关键点数, 0为没有 不能和 IS_MOSAIC 一起用
    cfg.IS_KEEP_SCALE = True  # 数据处理保持长宽
    cfg.IS_MOSAIC = False
    cfg.IS_MOSAIC_KEEP_WH = False  # IS_MOSAIC 是主开关 直接拉伸
    cfg.IS_MOSAIC_FILL = True  # IS_MOSAIC 使用 是IS_MOSAIC_KEEP_WH 副形状

    cfg.NUM_CLASSES = 1
    cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/widerface/'
    cfg.PATH_COCO_TARGET_TRAIN = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_TRAIN = cfg.PATH_DATA_ROOT + '/coco/images/train2017'
    cfg.PATH_COCO_TARGET_EVAL = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_EVAL = cfg.PATH_DATA_ROOT + '/coco/images/val2017'
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + '_widerface'
    cfg.PATH_TENSORBOARD = 'runs_widerface'
    cfg.DATA_NUM_WORKERS = 5

    data_transform = cre_transform4resize(cfg)

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

    custom_set(cfg)
    return loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler


def fload_raccoon(cfg, is_mgpu):
    base_set(cfg)
    '''样本及预处理'''
    cfg.BATCH_SIZE = 32  # batch过小需要设置连续前传
    cfg.FORWARD_COUNT = 1  # 连续前传次数 accumulate = max(round(64 / CFG.BATCH_SIZE), 1)
    cfg.PRINT_FREQ = 2  # 400张图打印
    cfg.IMAGE_SIZE = (512, 512)  # wh 预处理 统一尺寸
    cfg.NUM_SAVE = 20

    cfg.IS_KEEP_SCALE = True  # 数据处理保持长宽
    cfg.IS_MOSAIC = False
    cfg.IS_MOSAIC_KEEP_WH = False  # IS_MOSAIC 是主开关 直接拉伸
    cfg.IS_MOSAIC_FILL = True  # IS_MOSAIC 使用 是IS_MOSAIC_KEEP_WH 副形状

    cfg.NUM_CLASSES = 1
    cfg.NUM_KEYPOINTS = 0  # 关键点数, 0为没有 不能和 IS_MOSAIC 一起用
    cfg.DATA_NUM_WORKERS = 2
    cfg.PATH_DATA_ROOT = cfg.PATH_HOST + '/AI/datas/raccoon200/'
    cfg.PATH_COCO_TARGET_TRAIN = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_TRAIN = cfg.PATH_DATA_ROOT + '/VOCdevkit/JPEGImages'
    cfg.PATH_COCO_TARGET_EVAL = cfg.PATH_DATA_ROOT + '/coco/annotations'
    cfg.PATH_IMG_EVAL = cfg.PATH_DATA_ROOT + '/coco/images/val2017'
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'raccoon200'
    cfg.PATH_TENSORBOARD = 'runs_raccoon200'

    data_transform = cre_transform4resize(cfg)

    mode = 'bbox'  # bbox segm keypoints caption
    # 返回数据已预处理 返回np(batch,(3,640,640))  , np(batch,(x个选框,15维))
    file_json = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_train2017.json'
    path_img = cfg.PATH_IMG_TRAIN
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

    file_json = cfg.PATH_COCO_TARGET_TRAIN + r'/instances_val2017.json'
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

    custom_set(cfg)
    return loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler