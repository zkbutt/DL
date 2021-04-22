from object_detection.CONFIG_BASE import CfgBase


class CFG(CfgBase):
    DEBUG = False
    IS_FORCE_SAVE = False
    IS_LOCK_BACKBONE_WEIGHT = False  # 锁定 BACKBONE_WEIGHT keypoints 不能使用
    END_EPOCH = 120

    IS_TRAIN = True
    IS_COCO_EVAL = True
    IS_VISUAL = False
    IS_TRAIN_DEBUG = True

    '''可视化'''
    IS_VISUAL_PRETREATMENT = False  # 图片预处理
    IS_FMAP_EVAL = False  # 只运行生成一次
    IS_KEEP_SCALE = False

    #  conf_pos conf_neg cls loss_txty  loss_twth
    LOSS_WEIGHT = [5., 1, 1, 1, 1]

    # THRESHOLD_PREDICT_CONF = 0.01  # 用于预测的阀值
    THRESHOLD_PREDICT_CONF = 0.1  # 用于预测的阀值
    THRESHOLD_PREDICT_NMS = 0.5  # 提高 conf 提高召回, 越小框越少

    '''模型参数'''
    SAVE_FILE_NAME = 't_yolo1_'
