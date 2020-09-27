'''样本'''
PATH_DATA_ROOT = r'm:\AI\datas\VOC2012'
BATCH_SIZE = 3  # b32_i2_d1  b16_i0.98_d0.5  b24_i0.98_d0.5
PRINT_FREQ = 50  # 每50批打印一次
NUM_CLASSES = 21
NEG_RATIO = 3  # POS正例比例

'''模型权重'''
# PATH_MODEL_WEIGHT = r'm:\AI\weights\pytorch\resnet50-19c8e357.pth'
PATH_MODEL_WEIGHT = None
# PATH_SSD_WEIGHT = r'M:\AI\weights\nvidia\ssd_nvidia_ssdpyt_amp_200703.pt'
PATH_SSD_WEIGHT = None
# PATH_FIT_WEIGHT = r'm:\AI\weights\feadre\ssd300-0.pth'
PATH_SAVE_WEIGHT = r'm:\AI\weights\feadre'
PATH_FIT_WEIGHT = r'M:\AI\weights\feadre\ssd300-9.pth'
END_EPOCHS = 12

'''模型参数'''
OUT_CHANNELS = [1024, 512, 512, 256, 256, 256]  # 特图输出维度
NUM_DEFAULTS = [4, 6, 6, 6, 4, 4]  # 特图对应的 anc数
MIDDLE_CHANNELS = [256, 256, 128, 128, 128]  # 特图改装中间维度
