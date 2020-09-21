import os

import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_show import plot_loss_and_lr, plot_map
from object_detection.faster_rcnn import p_transforms4faster
from object_detection.f_fit_tools import sysconfig, load_data4voc, load_weight, save_weight
from object_detection.faster_rcnn.CONFIG_FASTER import PATH_SAVE_WEIGHT, BATCH_SIZE, NUM_CLASSES, PATH_MODEL_WEIGHT, \
    PATH_FIT_WEIGHT, PRINT_FREQ, PATH_DATA_ROOT, END_EPOCHS, DEBUG
from object_detection.faster_rcnn.network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from object_detection.faster_rcnn.backbone.resnet50_fpn_model import resnet50_fpn_backbone
from object_detection.faster_rcnn.train_utils import train_eval_utils as utils


def create_model(num_classes, path_weights):
    backbone = resnet50_fpn_backbone()
    model = FasterRCNN(backbone=backbone, num_classes=91)  # 用的coco权重是91
    # summary(model, (3, 400, 500)) # 无法使用
    # 载入预训练模型权重
    if path_weights and os.path.exists(path_weights):
        # 这个权重是整个网络的权重,不需要分步 底层会自动冻结权重
        weights_dict = torch.load(path_weights)
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            flog.debug("missing_keys: %s", missing_keys)
            flog.warning("unexpected_keys:%s ", unexpected_keys)  # 意外的
        flog.debug('weight_ssd 权重加载成功 %s', path_weights)
    else:
        flog.warning('weight_ssd 未加载权重 %s', path_weights)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


if __name__ == "__main__":
    '''------------------系统配置---------------------'''
    device = sysconfig(PATH_SAVE_WEIGHT)
    if DEBUG:
        device = torch.device("cpu")

    '''---------------数据加载及处理--------------'''
    # 主要是需要同时处理 target 采用自定义的 transforms
    data_transform = {
        "train": p_transforms4faster.Compose([p_transforms4faster.ToTensor(),
                                              p_transforms4faster.RandomHorizontalFlip(0.5)]),
        "val": p_transforms4faster.Compose([p_transforms4faster.ToTensor()])
    }

    train_data_loader, val_data_set_loader = load_data4voc(data_transform,
                                                           PATH_DATA_ROOT,
                                                           BATCH_SIZE,
                                                           bbox2one=False,
                                                           isdebug=DEBUG,
                                                           )

    '''------------------模型定义---------------------'''
    model = create_model(NUM_CLASSES, PATH_MODEL_WEIGHT)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.33)

    '''------------------模型训练---------------------'''
    # feadre权重加载
    start_epoch = load_weight(PATH_FIT_WEIGHT, model, optimizer, lr_scheduler)
    flog.debug('---训练开始---start_epoch %s', start_epoch)

    train_loss = []
    learning_rate = []
    val_map = []
    for epoch in range(start_epoch, END_EPOCHS):
        utils.train_one_epoch(model=model, optimizer=optimizer,
                              data_loader=train_data_loader,
                              device=device, epoch=epoch,
                              print_freq=PRINT_FREQ,
                              train_loss=train_loss,
                              train_lr=learning_rate)

        lr_scheduler.step()  # 更新学习率

        utils.evaluate(model=model, data_loader=val_data_set_loader,
                       device=device, data_set=None,
                       mAP_list=val_map)

        # 每个epoch保存
        save_weight(PATH_SAVE_WEIGHT,
                    model,
                    'faster_rcnn4resnet',
                    optimizer,
                    lr_scheduler,
                    epoch)

    '''-------------结果可视化-----------------'''
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        plot_map(val_map)
    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
