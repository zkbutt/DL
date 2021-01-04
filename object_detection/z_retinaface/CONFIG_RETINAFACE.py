from f_tools.datas.data_loader import fload_raccoon_train, fload_voc, fload_type_train, fload_raccoon_eval, \
    fload_type_eval
from object_detection.CONFIG_BASE import CfgBase


class CFG(CfgBase):
    DEBUG = False
    IS_FORCE_SAVE = False
    IS_LOCK_BACKBONE_WEIGHT = False  # 锁定 BACKBONE_WEIGHT keypoints 不能使用
    END_EPOCH = 180

    IS_TRAIN = True
    IS_COCO_EVAL = True
    IS_VISUAL = False

    # FUN_LOADER_DATA = fload_raccoon_train  # raccoon
    # FUN_EVAL_DATA = fload_raccoon_eval  # raccoon

    FUN_LOADER_DATA = fload_type_train  # type
    FUN_EVAL_DATA = fload_type_eval  # type
    # BATCH_SIZE = 40  # m2
    BATCH_SIZE = 20  # d121

    # FUN_LOADER_DATA = fload_voc  # voc
    # FUN_LOADER_DATA = fload_type  # cat

    '''可视化'''
    IS_VISUAL_PRETREATMENT = False  # 图片预处理可视化
    IS_FMAP_EVAL = False  # 只运行生成一次
    IS_KEEP_SCALE = False

    '''LOSS参数'''
    THRESHOLD_CONF_NEG = 0.5
    THRESHOLD_CONF_POS = 0.7
    THRESHOLD_PREDICT_CONF = 0.7  # 用于预测的阀值
    THRESHOLD_PREDICT_NMS = 0.3  # 用于预测的阀值
    LOSS_WEIGHT = [1., 1, 1, 1]  # 损失系数 用于  loss_bboxs l_conf loss_labels  loss_keypoints
    NEG_RATIO = 3  # 负样本倍数
    LR0 = 1e-3
    FOCALLOSS_ALPHA = 0.25
    FOCALLOSS_GAMMA = 2

    '''模型参数'''
    SAVE_FILE_NAME = 'train_retina_'  # 预置一个 实际无用 根据文件名确定
    ANCHORS_CLIP = True  # 是否剔除超边界
    NUMS_ANC = [2, 2, 2]
