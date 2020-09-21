'''样本'''
# PATH_DATA_ROOT = r'D:\down\AI\datas\widerface'
# PATH_DATA_ROOT = r'D:\down\AI\datas\m_VOC2007'
PATH_DATA_ROOT = r'm:\AI\datas\VOC2012'
# PATH_DATA_ROOT = r'I:\VOC2012'
BATCH_SIZE = 2  # b32_i2_d1  b16_i0.98_d0.5  b24_i0.98_d0.5
PRINT_FREQ = 50  # 每50批打印一次
NUM_CLASSES = 21

'''模型权重'''
PATH_MODEL_WEIGHT = r'M:\AI\weights\pytorch\fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
# PATH_MODEL_WEIGHT = r'M:\AI\weights\pytorch\fasterrcnn_mobilenet_v2-b0353104.pth'
# PATH_MODEL_WEIGHT = None
# PATH_FIT_WEIGHT = r'm:\AI\weights\feadre\ssd300-0.pth'
# PATH_FIT_WEIGHT = r'M:\AI\weights\feadre\train_mobilenet.py-2.pth'
PATH_FIT_WEIGHT = None
PATH_SAVE_WEIGHT = r'm:\AI\weights\feadre'
END_EPOCHS = 10

'''模型参数'''
# 详见 FasterRCNN 这个类
# OUT_CHANNELS = [1024, 512, 512, 256, 256, 256]  # 特图输出维度
# NUM_DEFAULTS = [4, 6, 6, 6, 4, 4]  # 特图对应的 anc数
# MIDDLE_CHANNELS = [256, 256, 128, 128, 128]  # 特图改装中间维度
# NEG_RATIO = 3  # POS


# 这是image_net的均值和方差
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]
