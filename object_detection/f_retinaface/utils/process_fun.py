from object_detection.f_fit_tools import load_weight
from object_detection.f_retinaface.nets.mobilenet025 import MobileNetV1
from object_detection.f_retinaface.nets.retinaface import RetinaFace
from object_detection.f_retinaface.CONFIG_F_RETINAFACE import *


def init_model():
    backbone = MobileNetV1()
    model = RetinaFace(backbone, IN_CHANNELS, OUT_CHANNEL, RETURN_LAYERS, ANCHOR_NUM, NUM_CLASSES)
    # losser = KeypointsLoss(anchors, NEGATIVE_RATIO, VARIANCE, LOSS_COEFFICIENT)
    # # 权重衰减(如L2惩罚)(默认: 0)
    # optimizer = optim.Adam(model.parameters(), 1e-3, weight_decay=5e-4)
    # # 在发现loss不再降低或者acc不再提高之后，降低学习率
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    # start_epoch = load_weight(PATH_FIT_WEIGHT, model, optimizer, lr_scheduler, device)


    return model
