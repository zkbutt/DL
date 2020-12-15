class CFG:
    DEBUG = True
    IS_FORCE_SAVE = False

    IS_LOCK_BACKBONE_WEIGHT = False  # 锁定 BACKBONE_WEIGHT keypoints 不能使用
    BATCH_SIZE = 52  # batch过小需要设置连续前传
    # BATCH_SIZE = 120  # batch过小需要设置连续前传
    FORWARD_COUNT = 2  # 连续前传次数 accumulate = max(round(64 / CFG.BATCH_SIZE), 1)
    PRINT_FREQ = int(400 / BATCH_SIZE)  # 400张图打印
    # PRINT_FREQ = 1  # 400张图打印
    END_EPOCH = 120

    IS_TRAIN = False
    IS_MOSAIC = False
    IS_MOSAIC_KEEP_WH = False  # IS_MOSAIC 使用
    IS_MOSAIC_FILL = False  # IS_MOSAIC 使用
    IS_COCO_EVAL = True
    IS_FMAP_EVAL = False  # FMAP只运行生成一次

    PATH_HOST = 'M:'  # 416
    # PATH_HOST = ''  # 需要尾部加/
    '''训练参数'''
    SYSNC_BN = False  # 不冻结时可使用多设备同步BN,速度减慢

    '''可视化 用于CPU调试'''
    IS_VISUAL = False
    IS_VISUAL_PRETREATMENT = False  # 图片预处理

    '''样本及预处理'''
    DATA_NUM_WORKERS = 8
    # PATH_DATA_ROOT = PATH_ROOT + '/AI/datas/widerface/coco'
    PATH_DATA_ROOT = PATH_HOST + '/AI/datas/VOC2012'

    # 训练
    PATH_COCO_TARGET_TRAIN = PATH_DATA_ROOT + '/coco/annotations'
    PATH_IMG_TRAIN = PATH_DATA_ROOT + '/trainval/JPEGImages'
    # 验证
    PATH_COCO_TARGET_EVAL = PATH_DATA_ROOT + '/coco/annotations'
    PATH_IMG_EVAL = PATH_DATA_ROOT + '/test/JPEGImages'

    IMAGE_SIZE = (416, 416)  # wh 预处理 统一尺寸
    NUM_GRID = 7  # 输出网格
    NUM_CLASSES = 20  # 模型分类数 人脸只有1 0 影响类别输出   -----这个要根据样本改----
    NUM_BBOX = 2

    '''模型权重'''
    PATH_SAVE_WEIGHT = PATH_HOST + '/AI/weights/feadre'
    SAVE_FILE_NAME = 'train_yolov1_'  # 预置一个 实际无用 根据文件名确定
    FILE_FIT_WEIGHT = PATH_SAVE_WEIGHT + '/train_yolov1_mobilenet_v2-111_2.4843227863311768.pth'
    # [0.060798525304197784, 0.16004431729031623, 0.032818981070488314, 0.0, 0.02438206740414694, 0.08068660961275709, 0.08239716312056737, 0.10574468085106382, 0.10574468085106382, 0.0, 0.06552526595744682, 0.1241599707815924]

    '''Loss参数'''
    THRESHOLD_CONF_NEG = 0.5  # 负例权重减小
    THRESHOLD_BOX = 5  # box放大
    THRESHOLD_PREDICT_CONF = 0.5  # 用于预测的阀值
    THRESHOLD_PREDICT_NMS = 0.3  # 用于预测的阀值

    IS_MIXTURE_FIX = True  # 这个不要动 半精度训练

    '''EVAL---MAP运算'''
    PATH_EVAL_IMGS = PATH_HOST + r'/AI/datas/widerface/coco/images/val2017'
    PATH_EVAL_INFO = PATH_HOST + r'/AI/datas/widerface/f_map'  # dt会自动创建
