from object_detection.CONFIG_BASE import CfgBase


class CFG(CfgBase):
    DEBUG = False
    IS_FORCE_SAVE = False
    IS_LOCK_BACKBONE_WEIGHT = False  # 锁定 BACKBONE_WEIGHT keypoints 不能使用
    END_EPOCH = 180

    IS_TRAIN = False
    IS_COCO_EVAL = True,
    IS_VISUAL = True

    '''可视化'''
    IS_VISUAL_PRETREATMENT = False  # 图片预处理可视化
    IS_FMAP_EVAL = False  # 只运行生成一次
    IS_KEEP_SCALE = False

    # 调参区
    LOSS_WEIGHT = [1, 1, 1., 1.]  # loss_conf,loss_xy,loss_wh

    THRESHOLD_PREDICT_CONF = 0.01  # 用于预测的阀值
    THRESHOLD_PREDICT_NMS = 0.5  # 用于预测的阀值
    SAVE_FILE_NAME = 't_yolo3_'  # 预置一个 实际无用 根据文件名确定
