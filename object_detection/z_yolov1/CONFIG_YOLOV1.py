from object_detection.CONFIG_BASE import CfgBase


class CFG(CfgBase):
    DEBUG = False
    IS_FORCE_SAVE = False
    IS_LOCK_BACKBONE_WEIGHT = False  # 锁定 BACKBONE_WEIGHT keypoints 不能使用
    END_EPOCH = 300

    IS_TRAIN = False
    IS_COCO_EVAL = True
    IS_VISUAL = False
    IS_TRAIN_DEBUG = True

    '''可视化'''
    IS_VISUAL_PRETREATMENT = False  # 图片预处理
    IS_FMAP_EVAL = False  # 只运行生成一次
    IS_KEEP_SCALE = False

    '''
    Loss参数 
    [30., 1, 0.05, 0.9]  负例降得慢  正例与cls波动
    
    '''
    #  conf_pos conf_neg cls loss_txty  loss_twth
    LOSS_WEIGHT = [1., 1, 1, 1, 1]

    THRESHOLD_PREDICT_CONF = 0.01  # 用于预测的阀值
    THRESHOLD_PREDICT_NMS = 0.5  # 用于预测的阀值

    '''模型参数'''
    NUM_GRID = 13  # 输出网格
    NUM_BBOX = 2
    SAVE_FILE_NAME = 't_yolo1_'
