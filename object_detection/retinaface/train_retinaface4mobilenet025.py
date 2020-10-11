import os

from f_tools.GLOBAL_LOG import flog

from object_detection.f_fit_tools import sysconfig, load_weight, save_weight
from object_detection.retinaface.CONFIG_RETINAFACE import PATH_SAVE_WEIGHT, PATH_DATA_ROOT, DEBUG, IMAGE_SIZE, \
    MOBILENET025, PATH_FIT_WEIGHT, NEGATIVE_RATIO, NEG_IOU_THRESHOLD, END_EPOCH, \
    PRINT_FREQ, BATCH_SIZE, VARIANCE, LOSS_COEFFICIENT, DATA_NUM_WORKERS, IS_EVAL, IS_TRAIN
from object_detection.retinaface.train_retinaface import data_loader, model_init, trainning

if __name__ == "__main__":
    '''
    BATCH_SIZE = 48
    cpu 10 进程 41% 11g  data: 0.0002
    GPU1: 5445MiB
    time: 0.5639  data: 0.0002
    Total time: 0:03:16 第二轮0:02:37
    
    GPU2: 4805
    0.5534  data: 0.0002 
    
    双gpu 5327
    time: 0.9763  data: 0.0006
    0:04:43
    
    
    '''
    '''------------------系统配置---------------------'''
    device = sysconfig(PATH_SAVE_WEIGHT)

    # claxx = RESNET50  # 这里根据实际情况改
    claxx = MOBILENET025  # 这里根据实际情况改

    '''---------------数据加载及处理--------------'''
    loader_train, loader_val = data_loader(device)

    '''------------------模型定义---------------------'''
    model, anchors, losser, optimizer, lr_scheduler, start_epoch = model_init(claxx, device)

    '''------------------模型训练---------------------'''
    trainning(start_epoch, model, device, anchors, losser, optimizer, lr_scheduler, loader_train, loader_val)

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
