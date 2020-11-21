class CFG:
    DEBUG = False
    IS_VISUAL = False  # 可视化总开关
    IS_VISUAL_PRETREATMENT = False  # 图片预处理阶段

    IS_TRAIN = True
    IS_MIXTURE_FIX = True  # 是否采用混合训练
    # PATH_ROOT = 'M:/'
    PATH_ROOT = '/home/bak3t/bak299g/'  # 需要尾部加/
    '''训练参数'''
    SYSNC_BN = False  # 不冻结时可使用多设备同步BN,速度减慢
    BATCH_SIZE = 4  # batch过小需要设置连续前传
    FORWARD_COUNT = 2  # 连续前传次数
    PRINT_FREQ = 100  # 每50批打印一次
    END_EPOCH = 6

    '''EVAL---MAP运算'''
    IS_EVAL = False  # 判断个文件夹 防误操作
    IS_RUN_ONE = True
    PATH_DT_ROOT = r'M:\AI\datas\VOC2012\test'  # 不能在CPU上弄
    # PATH_DT_ROOT = '/home/win10_sys/AI/datas/widerface/val'
    PATH_DT_RES = r'M:\temp\11'
    # PATH_DT_RES = '/home/win10_sys/AI/datas/widerface/val'

    '''样本及预处理'''
    DATA_NUM_WORKERS = 20
    PATH_DATA_ROOT = PATH_ROOT + 'AI/datas/VOC2012/trainval'
    IMAGE_SIZE = (416, 416)  # wh 预处理 统一尺寸
    GRID = 7  # 输出网格
    NUM_CLASSES = 20  # 模型分类数 人脸只有1 0 影响类别输出   -----这个要根据样本改----
    NUM_BBOX = 1

    '''模型权重'''
    PATH_SAVE_WEIGHT = PATH_ROOT + 'AI/weights/feadre'
    SAVE_FILE_NAME = 'def'  # 预置一个 实际无用 根据文件名确定
    FILE_FIT_WEIGHT = PATH_ROOT + 'AI/weights/feadre/train_yolo1_DDP.pydensenet121-4_3.0535318851470947.pth'
    # FILE_FIT_WEIGHT = None

    '''Loss参数'''
    L_NOOBJ = 0.5
    L_COORD = 5
