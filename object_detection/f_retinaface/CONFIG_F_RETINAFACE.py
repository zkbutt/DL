class CFG():
    DEBUG = False
    IS_VISUAL = False
    IS_TRAIN = True
    IS_EVAL = False
    IS_MIXTURE_FIX = True
    # PATH_ROOT = 'M:/'
    PATH_ROOT = '/home/bak3t/bak299g/'

    '''样本及预处理'''
    DATA_NUM_WORKERS = 10
    PATH_DATA_ROOT = PATH_ROOT + 'AI/datas/widerface/coco'
    IMAGE_SIZE = (640, 640)  # wh 预处理 统一尺寸
    BATCH_SIZE = 48  # b32_i2_d1  b16_i0.98_d0.5  b24_i0.98_d0.5
    NUM_CLASSES = 2  # 模型分类数 人脸只有1 0 影响类别输出   -----这个要根据样本改----
    # rgb_mean = (104, 117, 123)  # 图片的RGB偏差

    '''模型权重'''
    PATH_SAVE_WEIGHT = PATH_ROOT + 'AI/weights/feadre'
    SAVE_FILE_NAME = 'f_retinaface'
    # loss_total: 13.1264 (16.0534)  loss_bboxs: 0.9631 (1.2294)  loss_labels: 2.2601 (2.2723)  loss_keypoints: 8.3139 (11.3223)
    # loss_total: 12.3518 (12.9931)  loss_bboxs: 0.8875 (1.0694)  loss_labels: 2.2473 (2.2477)  loss_keypoints: 8.2336 (8.6065)
    FILE_FIT_WEIGHT = PATH_ROOT + 'AI/weights/feadre/train_retinaface.py-1_24.023202896118164.pth'
    # FILE_FIT_WEIGHT = '/home/win10_sys/tmp/pycharm_project_243/object_detection/retinaface/file/Retinaface_mobilenet0.25.pth'
    # FILE_FIT_WEIGHT = r'D:\tb\tb\ai_code\DL\object_detection\retinaface\file\Retinaface_mobilenet0.25.pth'

    '''训练'''
    PRINT_FREQ = 50  # 每50批打印一次
    END_EPOCH = 50
    VARIANCE = [0.1, 0.2]  # 框修正限制
    LOSS_WEIGHT = [2, 1, 1]  # 损失系数 用于  loss_bboxs loss_labels  loss_keypoints

    '''Loss参数'''
    PREDICT_IOU_THRESHOLD = 0.3  # 用于预测的阀值
    NEGATIVE_RATIO = 3  # 负样本倍数
    NEG_IOU_THRESHOLD = 0.35  # 小于时作用负例,用于 MultiBoxLoss

    '''ANCHORS相关'''
    MODEL_NAME = 'mobilenet0.25'
    ANCHORS_SIZE = [[16, 32], [64, 128], [256, 512]]
    FEATURE_MAP_STEPS = [8, 16, 32]  # 特图的步距
    ANCHORS_VARIANCE = [0.1, 0.2]  # 修复系数 中心0.1 长宽0.2
    ANCHORS_CLIP = True  # 是否剔除超边界
    ANCHOR_NUM = 2

    '''模型参数'''
    FILE_WEIGHT = PATH_ROOT + 'AI/weights/retinaface/mobilenetV1X0.25_pretrain.tar'

    IN_CHANNELS = 32  # in_channels 用于设置 FPN的输入
    OUT_CHANNEL = 64  # 定义FPN的输出  通常为统一尺寸
    RETURN_LAYERS = {'stage1': 1, 'stage2': 2, 'stage3': 3}