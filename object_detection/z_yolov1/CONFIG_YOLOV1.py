class CFG:
    DEBUG = False
    IS_FORCE_SAVE = False

    IS_LOCK_BACKBONE_WEIGHT = False  # 锁定 BACKBONE_WEIGHT keypoints 不能使用
    BATCH_SIZE = 52  # batch过小需要设置连续前传
    # BATCH_SIZE = 120  # batch过小需要设置连续前传
    FORWARD_COUNT = 2  # 连续前传次数 accumulate = max(round(64 / CFG.BATCH_SIZE), 1)
    PRINT_FREQ = int(400 / BATCH_SIZE)  # 400张图打印
    # PRINT_FREQ = 1  # 400张图打印
    END_EPOCH = 110

    IS_TRAIN = False
    IS_MOSAIC = False
    IS_COCO_EVAL = True
    IS_FMAP_EVAL = False  # FMAP只运行生成一次

    # PATH_HOST = 'M:'  # 416
    PATH_HOST = '/home/bak3t/bak299g'  # 需要尾部加/
    '''训练参数'''
    SYSNC_BN = False  # 不冻结时可使用多设备同步BN,速度减慢

    '''可视化 用于CPU调试'''
    IS_VISUAL = False
    IS_VISUAL_PRETREATMENT = False  # 图片预处理

    '''样本及预处理'''
    DATA_NUM_WORKERS = 8
    # PATH_DATA_ROOT = PATH_ROOT + '/AI/datas/widerface/coco'
    PATH_DATA_ROOT = PATH_HOST + '/AI/datas/VOC2012'
    PATH_COCO_TARGET_TRAIN = PATH_DATA_ROOT + '/coco/annotations'
    PATH_IMG_TRAIN = PATH_DATA_ROOT + '/trainval/JPEGImages'

    PATH_COCO_TARGET_EVAL = PATH_DATA_ROOT + '/coco/annotations'
    PATH_IMG_EVAL = PATH_DATA_ROOT + '/test/JPEGImages'

    IMAGE_SIZE = (416, 416)  # wh 预处理 统一尺寸
    NUM_GRID = 7  # 输出网格
    NUM_CLASSES = 20  # 模型分类数 人脸只有1 0 影响类别输出   -----这个要根据样本改----
    NUM_BBOX = 2

    '''模型权重'''
    PATH_SAVE_WEIGHT = PATH_HOST + '/AI/weights/feadre'
    SAVE_FILE_NAME = 'train_yolov1_'  # 预置一个 实际无用 根据文件名确定
    FILE_FIT_WEIGHT = PATH_HOST + '/AI/weights/feadre/train_yolov1mobilenet_v2-110_1.7359310388565063.pth'
    # [0.060798525304197784, 0.16004431729031623, 0.032818981070488314, 0.0, 0.02438206740414694, 0.08068660961275709, 0.08239716312056737, 0.10574468085106382, 0.10574468085106382, 0.0, 0.06552526595744682, 0.1241599707815924]
    # FILE_FIT_WEIGHT = PATH_HOST + '/AI/weights/feadre/train_yolov1mobilenet_v2-102_2.189227819442749.pth'  # 最好一次
    # [0.061976429380835085, 0.16622805719398262, 0.03247406096586994, 0.0002994048486785443, 0.02573890122808864, 0.08035893212461312, 0.08181733108740409, 0.10648979973797493, 0.10648979973797493, 0.008108108108108109, 0.06542025148908008, 0.12504663898417284]
    # FILE_FIT_WEIGHT = PATH_ROOT + '/AI/weights/feadre/train_yolov1mobilenet_v2-57_5.949509143829346.pth'

    '''Loss参数'''
    THRESHOLD_CONF_NEG = 0.5  # 负例权重减小
    THRESHOLD_BOX = 5  # box放大
    THRESHOLD_PREDICT_CONF = 0.5  # 用于预测的阀值
    THRESHOLD_PREDICT_NMS = 0.3  # 用于预测的阀值

    IS_MIXTURE_FIX = True  # 这个不要动 半精度训练

    '''EVAL---MAP运算'''
    PATH_EVAL_IMGS = PATH_HOST + r'/AI/datas/widerface/coco/images/val2017'
    PATH_EVAL_INFO = PATH_HOST + r'/AI/datas/widerface/f_map'  # dt会自动创建
