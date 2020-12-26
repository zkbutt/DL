from f_tools.datas.data_loader import fload_raccoon
from object_detection.CONFIG_BASE import CfgBase


class CFG(CfgBase):
    DEBUG = False
    IS_FORCE_SAVE = False
    IS_LOCK_BACKBONE_WEIGHT = False  # 锁定 BACKBONE_WEIGHT keypoints 不能使用
    END_EPOCH = 91

    IS_TRAIN = True
    IS_COCO_EVAL = True
    IS_FMAP_EVAL = False  # 只运行生成一次
    IS_KEEP_SCALE = False

    FUN_LOADER_DATA = fload_raccoon

    '''LOSS参数'''
    # NUM_NEG = 2000  # 负样本最大数量
    NUM_NEG = 99999  # 负样本最大数量
    THRESHOLD_NEG_IOU = 0.3  # 小于时作用负例 0.3~0.5 忽略
    THRESHOLD_PREDICT_CONF = 0.7  # 用于预测的阀值
    THRESHOLD_PREDICT_NMS = 0.3  # 用于预测的阀值
    LOSS_WEIGHT = [1, 1, 1]  # 损失系数 用于  loss_bboxs loss_labels  loss_keypoints
    NEG_RATIO = 3  # 负样本倍数
    LR0 = 1e-3
    FOCALLOSS_ALPHA = 0.75
    FOCALLOSS_GAMMA = 2

    '''训练参数'''
    SYSNC_BN = False  # 不冻结时可使用多设备同步BN,速度减慢
    IS_MULTI_SCALE = True  # 多尺度训练
    MULTI_SCALE_VAL = [0.667, 1.5]  # 多尺寸的比例0.667~1.5 之间 满足32的倍数

    '''可视化'''
    IS_VISUAL = False
    IS_VISUAL_PRETREATMENT = False  # 图片预处理

    SAVE_FILE_NAME = 'train_retinaface_'  # 预置一个 实际无用 根据文件名确定
    FILE_NAME_WEIGHT = 'train_retinaface_raccoon200_m2-1_9.056' + '.pth'
    # FILE_NAME_WEIGHT = '123'

    '''ANCHORS相关'''  # 每层anc数需统一，各层数据才能进行融合处理
    ANC_SCALE = [
        [[0.025, 0.025], [0.05, 0.05]],
        [[0.1, 0.1], [0.2, 0.2], ],
        [[0.4, 0.4], [0.8, 0.8], ],
    ]
    FEATURE_MAP_STEPS = [8, 16, 32]  # 特图的步距 下采倍数
    ANCHORS_CLIP = True  # 是否剔除超边界
    NUMS_ANC = [2, 2, 2]
