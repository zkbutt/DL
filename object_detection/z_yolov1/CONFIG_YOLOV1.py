class CFG:
    DEBUG = True
    IS_FORCE_SAVE = False

    IS_LOCK_BACKBONE_WEIGHT = False  # 锁定 BACKBONE_WEIGHT keypoints 不能使用
    IS_MOSAIC = False
    BATCH_SIZE = 22  # batch过小需要设置连续前传
    FORWARD_COUNT = 1  # 连续前传次数 accumulate = max(round(64 / CFG.BATCH_SIZE), 1)
    PRINT_FREQ = int(400 / BATCH_SIZE)  # 400张图打印
    # PRINT_FREQ = 1  # 400张图打印
    END_EPOCH = 65

    IS_TRAIN = True
    IS_FMAP_EVAL = False  # 只运行生成一次
    IS_COCO_EVAL = False

    IS_MIXTURE_FIX = True  # 半精度训练
    PATH_ROOT = 'M:/'
    # PATH_ROOT = '/home/bak3t/bak299g/'  # 需要尾部加/
    '''训练参数'''
    SYSNC_BN = False  # 不冻结时可使用多设备同步BN,速度减慢

    '''EVAL---MAP运算'''
    PATH_EVAL_IMGS = PATH_ROOT + r'AI/datas/widerface/coco/images/val2017'
    PATH_EVAL_INFO = PATH_ROOT + r'AI/datas/widerface/f_map'  # dt会自动创建

    '''可视化'''
    IS_VISUAL = False
    IS_VISUAL_PRETREATMENT = False  # 图片预处理

    '''样本及预处理'''
    DATA_NUM_WORKERS = 10
    PATH_DATA_ROOT = PATH_ROOT + 'AI/datas/widerface/coco'
    IMAGE_SIZE = (416, 416)  # wh 预处理 统一尺寸
    NUM_GRID = 7  # 输出网格
    NUM_CLASSES = 1  # 模型分类数 人脸只有1 0 影响类别输出   -----这个要根据样本改----
    NUM_BBOX = 2

    '''模型权重'''
    PATH_SAVE_WEIGHT = PATH_ROOT + 'AI/weights/feadre'
    SAVE_FILE_NAME = 'train_yolov1'  # 预置一个 实际无用 根据文件名确定
    # FILE_FIT_WEIGHT = PATH_ROOT + 'AI/weights/feadre/train_retinaface_mobilenet_v2-35_6.583288669586182.pth'
    # FILE_FIT_WEIGHT = PATH_ROOT + 'AI/weights/feadre/train_retinaface_densenet121-15_7.896054744720459.pth'
    FILE_FIT_WEIGHT = None


    '''Loss参数'''
    L_NOOBJ = 0.5
    L_COORD = 5
