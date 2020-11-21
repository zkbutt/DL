class CFG:
    DEBUG = False

    IS_TRAIN = True
    IS_MIXTURE_FIX = False
    # PATH_ROOT = 'M:/'
    PATH_ROOT = '/home/bak3t/bak299g/'  # 需要尾部加/
    '''训练参数'''
    SYSNC_BN = False  # 不冻结时可使用多设备同步BN,速度减慢
    BATCH_SIZE = 15  # batch过小需要设置连续前传
    FORWARD_COUNT = 2  # 连续前传次数 accumulate = max(round(64 / CFG.BATCH_SIZE), 1)
    PRINT_FREQ = 50  # 每50批打印一次
    END_EPOCH = 50
    IS_MULTI_SCALE = True
    MULTI_SCALE_VAL = [0.667, 1.5]

    '''可视化'''
    IS_VISUAL = False
    IS_VISUAL_PRETREATMENT = False  # 图片预处理

    '''EVAL---MAP运算'''
    IS_EVAL = False  # 判断个文件夹 防误操作
    IS_RUN_ONE = True
    # PATH_DT_ROOT = r'M:\AI\datas\widerface\val' # 不能在CPU上弄
    PATH_DT_ROOT = '/home/win10_sys/AI/datas/widerface/val'
    # PATH_DT_RES = r'M:\temp\11'
    PATH_DT_RES = '/home/win10_sys/AI/datas/widerface/val'

    '''样本及预处理'''
    DATA_NUM_WORKERS = 10
    PATH_DATA_ROOT = PATH_ROOT + 'AI/datas/VOC2012/trainval'
    # PATH_DATA_ROOT = '/home/win10_sys/' + 'AI/datas/widerface/coco'
    IMAGE_SIZE = (416, 416)  # wh 预处理 统一尺寸
    GRID = 7  # 输出网格
    NUM_CLASSES = 20  # 模型分类数 人脸只有1 0 影响类别输出   -----这个要根据样本改----
    NUM_BBOX = 1

    '''模型权重'''
    PATH_SAVE_WEIGHT = PATH_ROOT + 'AI/weights/feadre'
    SAVE_FILE_NAME = 'def'  # 预置一个 实际无用 根据文件名确定
    FILE_FIT_WEIGHT = PATH_ROOT + 'AI/weights/feadre/train_yolo3_DDP.pydarknet53-2_0.24243876338005066.pth'
    # FILE_FIT_WEIGHT = None

    '''Loss参数'''
    PREDICT_IOU_THRESHOLD = 0.3  # 用于预测的阀值
    NEGATIVE_RATIO = 3  # 负样本倍数
    NEG_IOU_THRESHOLD = 0.35  # 小于时作用负例,用于 MultiBoxLoss

    '''ANCHORS相关'''
    # 每层anc数需统一，各层数据才能进行融合处理
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


if __name__ == '__main__':
    import numpy as np

    array = np.array(CFG.ANC_SCALE)
    print(array * 416)
