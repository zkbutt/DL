import socket


class CFG:
    DEBUG = False
    IS_FORCE_SAVE = False

    IS_USE_KEYPOINT = True

    IS_LOCK_BACKBONE_WEIGHT = False  # 锁定 BACKBONE_WEIGHT keypoints 不能使用
    BATCH_SIZE = 22  # batch过小需要设置连续前传
    FORWARD_COUNT = 5  # 连续前传次数 accumulate = max(round(64 / CFG.BATCH_SIZE), 1)
    PRINT_FREQ = int(400 / BATCH_SIZE)  # 400张图打印
    # PRINT_FREQ = 1  # 400张图打印
    END_EPOCH = 91

    IS_TRAIN = False
    IS_MOSAIC = False
    IS_MOSAIC_KEEP_WH = False  # IS_MOSAIC 使用
    IS_MOSAIC_FILL = False  # IS_MOSAIC 使用
    IS_COCO_EVAL = True
    IS_FMAP_EVAL = False  # 只运行生成一次

    '''LOSS参数'''
    # NUM_NEG = 2000  # 负样本最大数量
    NUM_NEG = 99999  # 负样本最大数量
    THRESHOLD_NEG_IOU = 0.3  # 小于时作用负例 0.3~0.5 忽略
    THRESHOLD_PREDICT_CONF = 0.7  # 用于预测的阀值
    THRESHOLD_PREDICT_NMS = 0.3  # 用于预测的阀值
    LOSS_WEIGHT = [1, 1, 1]  # 损失系数 用于  loss_bboxs loss_labels  loss_keypoints
    NEG_RATIO = 3  # 负样本倍数
    LR0 = 1e-3 / 10 / 2
    FOCALLOSS_ALPHA = 0.75
    FOCALLOSS_GAMMA = 2

    IS_MIXTURE_FIX = True  # 半精度训练
    PATH_HOST = 'M:'
    # PATH_HOST = '/home/bak3t/bak299g'  # 需要尾部加/

    host_name = socket.gethostname()
    if host_name == 'Feadre-NB':
        PATH_HOST = 'M:'
        # raise Exception('当前主机: %s 及主数据路径: %s ' % (host_name, cfg.PATH_HOST))
    elif host_name == 'e2680v2':
        PATH_HOST = ''
    # PATH_HOST = 'M:'  # 416
    # PATH_HOST = '/home/bak3t/bak299g'  # 需要尾部加/

    '''训练参数'''
    SYSNC_BN = False  # 不冻结时可使用多设备同步BN,速度减慢
    IS_MULTI_SCALE = True  # 多尺度训练
    MULTI_SCALE_VAL = [0.667, 1.5]  # 多尺寸的比例0.667~1.5 之间 满足32的倍数

    '''可视化'''
    IS_VISUAL = False
    IS_VISUAL_PRETREATMENT = False  # 图片预处理

    '''样本及预处理'''
    DATA_NUM_WORKERS = 10
    PATH_DATA_ROOT = PATH_HOST + '/AI/datas/widerface/'
    # PATH_DATA_ROOT = PATH_ROOT + '/AI/datas/VOC2012/'

    # 训练
    PATH_COCO_TARGET_TRAIN = PATH_DATA_ROOT + '/coco/annotations'
    PATH_IMG_TRAIN = PATH_DATA_ROOT + '/coco/images/train2017'
    # 验证
    PATH_COCO_TARGET_EVAL = PATH_DATA_ROOT + '/coco/annotations'
    PATH_IMG_EVAL = PATH_DATA_ROOT + '/coco/images/val2017'

    IMAGE_SIZE = [640, 640]  # wh 预处理 统一尺寸
    # GRID = 7  # 输出网格
    NUM_CLASSES = 1  # 模型分类数 人脸只有1 0 影响类别输出   -----这个要根据样本改----
    # NUM_BBOX = 1

    '''模型权重'''
    PATH_SAVE_WEIGHT = PATH_HOST + '/AI/weights/feadre'
    SAVE_FILE_NAME = 'train_retinaface_'  # 预置一个 实际无用 根据文件名确定
    FILE_FIT_WEIGHT = PATH_SAVE_WEIGHT + '/train_retinaface_mobilenet_v2-88_2.6454436779022217.pth'
    # FILE_FIT_WEIGHT = None

    '''ANCHORS相关'''  # 每层anc数需统一，各层数据才能进行融合处理
    ANC_SCALE = [
        [[0.025, 0.025], [0.05, 0.05]],
        [[0.1, 0.1], [0.2, 0.2], ],
        [[0.4, 0.4], [0.8, 0.8], ],
    ]
    FEATURE_MAP_STEPS = [8, 16, 32]  # 特图的步距 下采倍数
    ANCHORS_CLIP = True  # 是否剔除超边界
    NUMS_ANC = [2, 2, 2]

    '''EVAL---MAP运算'''
    PATH_EVAL_IMGS = PATH_HOST + r'AI/datas/widerface/coco/images/val2017'
    PATH_EVAL_INFO = PATH_HOST + r'AI/datas/widerface/f_map'  # dt会自动创建


if __name__ == '__main__':
    import numpy as np

    array = np.array([[16, 32], [64, 128], [256, 512]])
    print(array / CFG.IMAGE_SIZE)
