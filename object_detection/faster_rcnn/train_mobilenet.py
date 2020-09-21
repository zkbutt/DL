import os

import torch
import torchvision

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_show import plot_loss_and_lr, plot_map
from object_detection.faster_rcnn import p_transforms4faster
from object_detection.faster_rcnn.CONFIG_FASTER import PATH_SAVE_WEIGHT, BATCH_SIZE, NUM_CLASSES, PATH_MODEL_WEIGHT, \
    PATH_FIT_WEIGHT, END_EPOCHS, PRINT_FREQ, PATH_DATA_ROOT
from object_detection.faster_rcnn.backbone.mobilenetv2_model import MobileNetV2
from object_detection.f_fit_tools import sysconfig, load_data4voc, load_weight, save_weight
from object_detection.faster_rcnn.network_files.faster_rcnn_framework import FasterRCNN
from object_detection.faster_rcnn.network_files.rpn_function import AnchorsGenerator
from object_detection.faster_rcnn.train_utils import train_eval_utils as utils


def create_model(num_classes, model_weights):
    backbone = MobileNetV2(weights_path=model_weights).features
    backbone.out_channels = 1280  # 这个参数是必须的 模型创建会检查

    if model_weights and os.path.exists(model_weights):
        # 这个权重是整个网络的权重,不需要分步 底层会自动冻结权重
        weights_dict = torch.load(model_weights)
        missing_keys, unexpected_keys = backbone.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            flog.debug("missing_keys: %s", missing_keys)
            flog.warning("unexpected_keys:%s ", unexpected_keys)  # 意外的
        flog.debug('backbone 权重加载成功 %s', model_weights)
    else:
        flog.warning('backbone 未加载权重 %s', model_weights)

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    # 创建ROI层
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                    output_size=[7, 7],  # roi_pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


if __name__ == "__main__":
    '''------------------系统配置---------------------'''
    device = sysconfig(PATH_SAVE_WEIGHT)

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
                                                           )

    '''------------------模型定义---------------------'''
    model = create_model(NUM_CLASSES, PATH_MODEL_WEIGHT)
    model.to(device)

    start_epoch = load_weight(PATH_FIT_WEIGHT, model)

    train_loss = []
    learning_rate = []
    val_map = []
    '''------------------模型训练---------------------'''
    if start_epoch < 10:
        flog.debug('---第一阶段---start_epoch %s', start_epoch)
        # -------------第一阶段-----------------
        # 首先冻结前置特征提取网络权重（backbone），训练rpn以及最终预测网络部分
        for param in model.backbone.parameters():
            param.requires_grad = False

        # define optimizer 遍历没有锁定的参数
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9,
                                    weight_decay=0.0005)

        for epoch in range(start_epoch, END_EPOCHS):
            # train for one epoch, printing every 10 iterations
            utils.train_one_epoch(model=model, optimizer=optimizer,
                                  data_loader=train_data_loader,
                                  device=device, epoch=epoch,
                                  print_freq=PRINT_FREQ,
                                  train_loss=train_loss,
                                  train_lr=learning_rate)
            # evaluate on the test dataset
            utils.evaluate(model=model, data_loader=val_data_set_loader,
                           device=device, data_set=None,
                           mAP_list=val_map)

            save_weight(PATH_SAVE_WEIGHT,
                        model,
                        os.path.basename(__file__),
                        epoch=epoch)

    else:
        flog.debug('---第二阶段---start_epoch %s', start_epoch)
        # -------------第二阶段-----------------
        # 解冻前置特征提取网络的部分底层权重（backbone），接着训练整个网络权重
        for name, parameter in model.backbone.named_parameters():
            split_name = name.split(".")[0]
            if split_name in ["0", "1", "2", "3"]:
                parameter.requires_grad = False
            else:
                parameter.requires_grad = True

        # define optimizer 遍历没有锁定的参数
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9,
                                    weight_decay=0.0005)
        # learning rate scheduler 每5步降低学习率
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=5,
                                                       gamma=0.33)

        for epoch in range(start_epoch, END_EPOCHS):
            # train for one epoch, printing every 50 iterations
            utils.train_one_epoch(model=model, optimizer=optimizer,
                                  data_loader=train_data_loader,
                                  device=device, epoch=epoch,
                                  print_freq=PRINT_FREQ,
                                  train_loss=train_loss,
                                  train_lr=learning_rate)
            utils.train_one_epoch(model, optimizer, train_data_loader,
                                  device, epoch, print_freq=50, warmup=True)
            # update the learning rate
            lr_scheduler.step()

            # evaluate on the test dataset
            utils.evaluate(model=model, data_loader=val_data_set_loader,
                           device=device, data_set=None,
                           mAP_list=val_map)

            # save weights
            if epoch > 10:
                save_weight(PATH_SAVE_WEIGHT,
                            model,
                            os.path.basename(__file__),
                            optimizer,
                            lr_scheduler,
                            epoch=epoch)

    '''-------------结果可视化-----------------'''
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    flog.debug('val_map %s', val_map)
    if len(val_map) != 0:
        plot_map(val_map)
    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
