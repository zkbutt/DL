from object_detection.CONFIG_BASE import CfgBase


class CFG(CfgBase):
    DEBUG = False
    IS_FORCE_SAVE = False
    IS_LOCK_BACKBONE_WEIGHT = False  # 锁定 BACKBONE_WEIGHT keypoints 不能使用
    END_EPOCH = 120

    IS_TRAIN = False
    IS_COCO_EVAL = True
    IS_VISUAL = False
    IS_TRAIN_DEBUG = True

    THRESHOLD_PREDICT_CONF = 0.05  # 用于预测的阀值
    THRESHOLD_PREDICT_NMS = 0.5  # 用于预测的阀值

    SAVE_FILE_NAME = 't_fkey_'  # 预置一个 实际无用 根据文件名确定

    ''' 特有参数 '''
    # 这两个在data中定义
    # MODE_COCO_TRAIN = 'keypoints'  # bbox segm keypoints caption
    # MODE_COCO_EVAL = 'keypoints'  # bbox segm keypoints caption
    MATCH_RADIUS = 1.5  # fcos 匹配半径
