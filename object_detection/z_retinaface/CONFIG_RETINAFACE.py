from object_detection.CONFIG_BASE import CfgBase


class CFG(CfgBase):
    DEBUG = False
    IS_FORCE_SAVE = False
    IS_LOCK_BACKBONE_WEIGHT = False  # 锁定 BACKBONE_WEIGHT keypoints 不能使用
    END_EPOCH = 180

    IS_TRAIN = True
    IS_COCO_EVAL = True,
    IS_VISUAL = False

    '''可视化'''
    IS_VISUAL_PRETREATMENT = False  # 图片预处理可视化
    IS_FMAP_EVAL = False  # 只运行生成一次
    IS_KEEP_SCALE = False

    '''LOSS参数'''
    THRESHOLD_CONF_POS = 0.5
    THRESHOLD_CONF_NEG = 0.3

    # THRESHOLD_PREDICT_CONF = 0.5  # 用于预测的阀值
    THRESHOLD_PREDICT_CONF = 0.5  # type3 0.7
    THRESHOLD_PREDICT_NMS = 0.3  # 用于预测的阀值
    LOSS_WEIGHT = [1, 1, 1, 1]  # 损失系数 用于  loss_bboxs l_conf loss_labels  loss_keypoints
    NEG_RATIO = 3  # 负样本倍数
    LR0 = 1e-3
    FOCALLOSS_ALPHA = 0.25
    FOCALLOSS_GAMMA = 2

    SAVE_FILE_NAME = 'train_retina_'  # 预置一个 实际无用 根据文件名确定

