from f_tools.datas.data_loader import fload_raccoon
from object_detection.CONFIG_BASE import CfgBase


class CFG(CfgBase):
    DEBUG = False
    IS_FORCE_SAVE = False
    IS_LOCK_BACKBONE_WEIGHT = False  # 锁定 BACKBONE_WEIGHT keypoints 不能使用
    END_EPOCH = 180

    IS_TRAIN = False
    IS_COCO_EVAL = True
    IS_VISUAL = False

    FUN_LOADER_DATA = fload_raccoon  # raccoon
    # FUN_LOADER_DATA = fload_voc  # voc
    # FUN_LOADER_DATA = fload_type  # cat

    '''可视化'''
    IS_VISUAL_PRETREATMENT = False  # 图片预处理
    IS_FMAP_EVAL = False  # 只运行生成一次
    IS_KEEP_SCALE = False

    '''Loss参数'''
    LOSS_WEIGHT = [1., 1, 1, 1]  # 未用
    THRESHOLD_CONF_NEG = 0.5  # 负例权重减小
    THRESHOLD_BOX = 5  # box放大
    THRESHOLD_PREDICT_CONF = 0.5  # 用于预测的阀值
    THRESHOLD_PREDICT_NMS = 0.3  # 用于预测的阀值

    '''模型参数'''
    NUM_GRID = 7  # 输出网格
    NUM_BBOX = 2
    SAVE_FILE_NAME = 'train_yolo1_'
