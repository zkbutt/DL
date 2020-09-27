from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_show import plot_loss_and_lr, plot_map
from object_detection.f_fit_tools import load_data4voc, sysconfig, save_weight, load_weight
from object_detection.ssd.CONFIG_SSD import NUM_CLASSES, PATH_FIT_WEIGHT, PATH_DATA_ROOT, BATCH_SIZE, PATH_SAVE_WEIGHT, \
    PATH_MODEL_WEIGHT, END_EPOCHS, PRINT_FREQ, PATH_SSD_WEIGHT
from object_detection.ssd.src.ssd_model import SSD300, Backbone
import torch
from object_detection.ssd import p_transform4ssd
import os
import object_detection.ssd.train_utils.train_eval_utils as utils


def create_model(num_classes, device, weight_backbone=None, weight_ssd=None):
    '''

    :param num_classes: 数据集类别 voc 20+1
    :param weight_backbone:  这是 backbone 数据集
    :return:
    '''
    backbone = Backbone(weight_backbone)
    model = SSD300(backbone=backbone, num_classes=num_classes)

    if weight_ssd and os.path.exists(weight_ssd):
        pre_model_dict = torch.load(weight_ssd, map_location=device)
        pre_weights_dict = pre_model_dict["model"]

        del_conf_loc_dict = {}
        for k, v in pre_weights_dict.items():
            split_key = k.split(".")
            if "conf" in split_key:  # 类别的剔除
                continue
            del_conf_loc_dict.update({k: v})

        missing_keys, unexpected_keys = model.load_state_dict(del_conf_loc_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            flog.debug("missing_keys: %s", missing_keys)
            flog.warning("unexpected_keys:%s ", unexpected_keys)  # 意外的
        flog.debug('weight_ssd 权重加载成功 %s', weight_ssd)
    return model


def main():
    '''------------------系统配置---------------------'''
    device = sysconfig(PATH_SAVE_WEIGHT)

    '''---------------数据加载及处理--------------'''
    data_transform = {
        "train": p_transform4ssd.Compose([p_transform4ssd.SSDCropping(),
                                          p_transform4ssd.Resize(),
                                          p_transform4ssd.ColorJitter(),
                                          p_transform4ssd.ToTensor(),
                                          p_transform4ssd.RandomHorizontalFlip(),
                                          p_transform4ssd.Normalization(),
                                          p_transform4ssd.AssignGTtoDefaultBox()]),
        "val": p_transform4ssd.Compose([p_transform4ssd.Resize(),
                                        p_transform4ssd.ToTensor(),
                                        p_transform4ssd.Normalization()])
    }

    train_data_loader, val_data_set_loader = load_data4voc(data_transform,
                                                           PATH_DATA_ROOT,
                                                           BATCH_SIZE,
                                                           bbox2one=True,
                                                           )

    # 预处理通过GPU
    # prefetcher = DATA_PREFETCHER(train_data_loader)
    # data = prefetcher.next()
    # i = 0
    # while data is not None:
    #     print(i, len(data))
    #     i += 1
    #     data = prefetcher.next()

    '''------------------模型定义---------------------'''
    # 该模型自带 loss 输出
    model = create_model(num_classes=NUM_CLASSES,
                         device=device,
                         weight_backbone=PATH_MODEL_WEIGHT,
                         weight_ssd=PATH_SSD_WEIGHT,
                         )
    model.to(device)

    # 对需要更新的参数进行
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
    # 每隔5步学习率降低一次
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    # feadre权重加载
    start_epoch = load_weight(PATH_FIT_WEIGHT, model, optimizer, lr_scheduler)

    train_loss = []
    learning_rate = []
    val_map = []

    flog.debug('---训练开始---start_epoch %s', start_epoch)
    # 如果电脑内存充裕，可提前加载验证集数据，以免每次验证时都要重新加载一次数据，节省时间
    # val_data = get_coco_api_from_dataset(train_data_loader.dataset)
    for epoch in range(start_epoch, END_EPOCHS):
        utils.train_one_epoch(model=model, optimizer=optimizer,
                              data_loader=train_data_loader,
                              device=device, epoch=epoch,
                              print_freq=PRINT_FREQ,
                              train_loss=train_loss,
                              train_lr=learning_rate)

        lr_scheduler.step()  # 更新学习

        utils.evaluate(model=model, data_loader=val_data_set_loader,
                       device=device, data_set=None,
                       mAP_list=val_map)

        # 每个epoch保存
        save_weight(PATH_SAVE_WEIGHT,
                    model,
                    'ssd300',
                    optimizer,
                    lr_scheduler,
                    epoch)

    '''-------------结果可视化-----------------'''
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    flog.debug('val_map %s', val_map)
    if len(val_map) != 0:
        plot_map(val_map)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录
    parser.add_argument('--data-path', default='./', help='dataset')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')

    args = parser.parse_args()
    flog.debug('args %s', args)
    main()
    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
