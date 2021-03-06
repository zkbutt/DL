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
    IS_VISUAL_PRETREATMENT = False  # 图片预处理可视化 用于调试
    IS_FMAP_EVAL = False  # 只运行生成一次 这个是另一种 map 几乎没用
    IS_KEEP_SCALE = False  # 计算map时恢复尺寸 几乎没用

    THRESHOLD_PREDICT_CONF = 0.05  # 用于预测的阀值
    THRESHOLD_PREDICT_NMS = 0.5  # 用于预测的阀值

    SAVE_FILE_NAME = 't_fcos_'  # 预置一个 实际无用 根据文件名确定

    ''' 特有参数 '''
    MATCH_RADIUS = 1.5  # fcos 匹配半径
