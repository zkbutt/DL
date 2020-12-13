class CFG:
    DEBUG = False
    IS_FORCE_SAVE = False

    IS_LOCK_BACKBONE_WEIGHT = False  # 锁定 BACKBONE_WEIGHT keypoints 不能使用
    # BATCH_SIZE = 128  # batch过小需要设置连续前传
    BATCH_SIZE = 40  # batch过小需要设置连续前传
    FORWARD_COUNT = 4  # 连续前传次数 accumulate = max(round(64 / CFG.BATCH_SIZE), 1)
    PRINT_FREQ = int(400 / BATCH_SIZE)  # 400张图打印
    END_EPOCH = 100

    IS_TRAIN = True
    IS_MOSAIC = False
    IS_MOSAIC_KEEP_WH = False  # IS_MOSAIC 使用
    IS_MOSAIC_FILL = False  # IS_MOSAIC 使用
    IS_COCO_EVAL = False
    IS_FMAP_EVAL = False  # FMAP只运行生成一次

    # NUM_NEG = 2000  # 负样本最大数量
    NUM_NEG = 99999  # 负样本最大数量
    THRESHOLD_CONF_NEG = 0.3  # 负例权重减小
    THRESHOLD_CONF_POS = 0.7  # 负例权重减小
    # THRESHOLD_PREDICT_CONF = 0.5  # 用于预测的阀值
    THRESHOLD_PREDICT_CONF = 0.1  # 用于预测的阀值
    THRESHOLD_PREDICT_NMS = 0.3  # 用于预测的阀值

    # 调参区
    LOSS_WEIGHT = [2., 2., 1., 1.]  # l_box_p,l_conf_p,l_cls_p,l_conf_n
    FOCALLOSS_ALPHA = 0.25
    FOCALLOSS_GAMMA = 2.
    LR = 1e-3

    # 暂时未处理
    IS_MULTI_SCALE = True  # 多尺度训练
    MULTI_SCALE_VAL = [0.667, 1.5]  # 多尺寸的比例0.667~1.5 之间 满足32的倍数

    # PATH_HOST = 'M:'  # 416
    PATH_HOST = '/home/bak3t/bak299g'  # 需要尾部加/
    '''训练参数'''
    SYSNC_BN = False  # 不冻结时可使用多设备同步BN,速度减慢

    '''可视化'''
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
    NUM_CLASSES = 20  # 模型分类数 人脸只有1 0 影响类别输出   -----这个要根据样本改----

    '''模型权重'''
    PATH_SAVE_WEIGHT = PATH_HOST + '/AI/weights/feadre'
    SAVE_FILE_NAME = 'train_yolov3_'  # 预置一个 实际无用 根据文件名确定
    FILE_FIT_WEIGHT = PATH_SAVE_WEIGHT + '/train_yolov3_mobilenet_v2-50_7.309931755065918.pth'

    '''Loss参数'''
    # IGNORE_THRESH = 0.225

    '''ANCHORS相关'''  # 每层anc数需统一，各层数据才能进行融合处理
    # ANCHORS_SIZE = [
    #     [[10, 13], [16, 30], [33, 23]],  # 大特图小目标 52, 52
    #     [[30, 61], [62, 45], [59, 119]],  # 26, 26
    #     [[116, 90], [156, 198], [373, 326]],  # 小特图大目标 13x13
    # ]
    ANC_SCALE = [
        [[0.13, 0.10666667], [0.06, 0.15733333], [0.036, 0.06006006], ],
        [[0.196, 0.51466667], [0.29, 0.28], [0.12, 0.28], ],
        [[0.81786211, 0.872], [0.374, 0.72266667], [0.612, 0.452], ],
    ]
    FEATURE_MAP_STEPS = [8, 16, 32]  # 特图的步距 下采倍数
    ANCHORS_CLIP = True  # 是否剔除超边界
    NUMS_ANC = [3, 3, 3]

    IS_MIXTURE_FIX = True  # 这个不要动 半精度训练


if __name__ == '__main__':
    import numpy as np

    array = np.array(CFG.ANC_SCALE)
    print(array * 416)
