class CFG:
    DEBUG = True
    IS_VISUAL = False  # 可视化总开关
    IS_VISUAL_PRETREATMENT = False  # 图片预处理阶段

    IS_TRAIN = True
    IS_MIXTURE_FIX = False  # 是否采用混合训练
    PATH_ROOT = 'M:/'
    # PATH_ROOT = '/home/win10_sys/'
    # PATH_ROOT = '/home/bak3t/bakls299g/'

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
    IMAGE_SIZE = (224, 224)  # wh 预处理 统一尺寸
    BATCH_SIZE = 7  # b32_i2_d1  b16_i0.98_d0.5  b24_i0.98_d0.5
    NUM_CLASSES = 20  # 模型分类数 人脸只有1 0 影响类别输出   -----这个要根据样本改----

    '''模型权重'''
    PATH_SAVE_WEIGHT = PATH_ROOT + 'AI/weights/feadre'
    SAVE_FILE_NAME = 'yolo3'
    # FILE_FIT_WEIGHT = PATH_ROOT + 'AI/weights/feadre/train_retinaface.py-10_7.423844337463379.pth'
    # FILE_FIT_WEIGHT = PATH_ROOT + 'AI/weights/feadre/train_retinaface.py-40_6.929875373840332.pth'
    # FILE_FIT_WEIGHT = PATH_ROOT + 'AI/weights/feadre/train_retinaface_DDP.py-4_7.723141670227051.pth'  # resnet50
    # FILE_FIT_WEIGHT = PATH_ROOT + 'AI/weights/feadre/train_retinaface_DDP.py_resnext50-2_7.655235290527344.pth'
    # FILE_FIT_WEIGHT = PATH_ROOT + 'AI/weights/feadre/train_retinaface.py-3_6.152596473693848.pth'
    # FILE_FIT_WEIGHT = '/home/win10_sys/tmp/pycharm_project_243/object_detection/retinaface/file/Retinaface_mobilenet0.25.pth'
    # FILE_FIT_WEIGHT = r'D:\tb\tb\ai_code\DL\object_detection\retinaface\file\Retinaface_mobilenet0.25.pth'

    '''训练'''
    PRINT_FREQ = 50  # 每50批打印一次
    END_EPOCH = 50

    '''Loss参数'''
    L_NOOBJ = 0.5
    L_COORD = 5
    GRID = 7

