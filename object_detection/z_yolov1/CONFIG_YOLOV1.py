from f_tools.datas.data_loader import fload_raccoon_train, fload_type_train, fload_type_eval, fload_raccoon_eval
from object_detection.CONFIG_BASE import CfgBase


class CFG(CfgBase):
    DEBUG = False
    IS_FORCE_SAVE = False
    IS_LOCK_BACKBONE_WEIGHT = False  # 锁定 BACKBONE_WEIGHT keypoints 不能使用
    END_EPOCH = 180

    IS_TRAIN = True
    IS_COCO_EVAL = True
    IS_VISUAL = False

    FUN_LOADER_DATA = fload_raccoon_train  # raccoon
    FUN_EVAL_DATA = fload_raccoon_eval  # raccoon

    # FUN_LOADER_DATA = fload_voc  # voc

    # FUN_LOADER_DATA = fload_type_train  # type
    # FUN_EVAL_DATA = fload_type_eval  # type

    '''可视化'''
    IS_VISUAL_PRETREATMENT = False  # 图片预处理
    IS_FMAP_EVAL = False  # 只运行生成一次
    IS_KEEP_SCALE = False

    '''
    Loss参数 
    [30., 1, 0.05, 0.9]  负例降得慢  正例与cls波动
    
    '''
    # box conf_pos conf_neg cls
    LOSS_WEIGHT = [30., 3, 0.05, 0.9]  # conf0.7
    # LOSS_WEIGHT = [50., 30, 1, 0.9]  # conf0.7
    LOSS_WEIGHT = [50., 1, 0.04, 1]  # conf0.7

    THRESHOLD_PREDICT_CONF = 0.7  # 用于预测的阀值
    THRESHOLD_PREDICT_NMS = 0.3  # 用于预测的阀值

    '''模型参数'''
    NUM_GRID = 7  # 输出网格
    NUM_BBOX = 2
    SAVE_FILE_NAME = 'train_yolo1_'
