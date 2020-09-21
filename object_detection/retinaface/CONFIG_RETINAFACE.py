DEBUG = True

'''样本及预处理'''
PATH_DATA_ROOT = r'D:\down\AI\datas\widerface'
# PATH_DATA_ROOT = r'M:\AI\datas\widerface'
IMAGE_SIZE = (640, 640)  # 预处理 统一尺寸
# IMAGE_SIZE = (840, 840)  # 预处理 统一尺寸
BATCH_SIZE = 2  # b32_i2_d1  b16_i0.98_d0.5  b24_i0.98_d0.5
NUM_CLASSES = 2  # 模型分类数 人脸只有1 0
rgb_mean = (104, 117, 123)  # 图片的RGB偏差

'''模型权重'''
# PATH_SAVE_WEIGHT = r'm:\AI\weights\feadre'
PATH_SAVE_WEIGHT = r'D:\down\AI\weights\feadre'
PATH_MODEL_WEIGHT = r'M:\AI\weights\pytorch\fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
# PATH_MODEL_WEIGHT = r'M:\AI\weights\pytorch\fasterrcnn_mobilenet_v2-b0353104.pth'
# PATH_MODEL_WEIGHT = None
# PATH_FIT_WEIGHT = r'm:\AI\weights\feadre\ssd300-0.pth'
# PATH_FIT_WEIGHT = r'M:\AI\weights\feadre\train_mobilenet.py-2.pth'
PATH_FIT_WEIGHT = None

'''训练'''
END_EPOCHS = 10
PRINT_FREQ = 50  # 每50批打印一次
OUT_CHANNEL = 64  # FPN的输出 与SSH输出一致

'''Loss参数'''
PREDICT_IOU_THRESHOLD = 0.3  # 用于预测的阀值
NEGATIVE_RATIO = 7  # 负样本比例
NEG_IOU_THRESHOLD = 0.35  # 小于时作用负例,用于 MultiBoxLoss


class MOBILENET025:
    '''ANCHORS相关'''
    MODEL_NAME = 'mobilenet0.25'
    ANCHORS_SIZE = [[16, 32], [64, 128], [256, 512]]
    FEATURE_MAP_STEPS = [8, 16, 32]  # 特图的步距
    ANCHORS_VARIANCE = [0.1, 0.2]  # 修复系数 中心0.1 长宽0.2
    ANCHORS_CLIP = True  # 是否剔除超边界
    ANCHOR_NUM = 2

    '''模型参数'''
    PATH_MODEL_WEIGHT = r'D:\down\AI\weights\retinaface\mobilenetV1X0.25_pretrain.tar'

    IN_CHANNELS_FPN = 32
    OUT_CHANNEL = 64
    RETURN_LAYERS = {'stage1': 1, 'stage2': 2, 'stage3': 3}
    loc_weight = 2.0
    EPOCH_START = 0
    EPOCH_END = 25


class RESNET50:
    '''ANCHORS相关'''
    MODEL_NAME = 'Resnet50'
    ANCHORS_SIZE = [[16, 32], [64, 128], [256, 512]]
    FEATURE_MAP_STEPS = [8, 16, 32]  # 特图的步距
    ANCHORS_VARIANCE = [0.1, 0.2]  # 修复系数 中心0.1 长宽0.2
    ANCHORS_CLIP = True  # 是否剔除超边界
    ANCHOR_NUM = 2

    '''模型参数'''
    IN_CHANNELS_FPN = [64, 128, 256]  # 对应RETURN_LAYERS
    IN_CHANNEL = 256
    OUT_CHANNEL = 256
    RETURN_LAYERS = {'layer2': 1, 'layer3': 2, 'layer4': 3}
    loc_weight = 2.0
