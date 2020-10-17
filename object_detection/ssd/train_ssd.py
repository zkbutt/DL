from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_factory import load_data4voc
from f_tools.f_torch_tools import load_weight, save_weight
from f_tools.fits.f_show_fit_res import plot_loss_and_lr, plot_map
from object_detection.ssd.CONFIG_SSD import NUM_CLASSES, PATH_FIT_WEIGHT, PATH_DATA_ROOT, BATCH_SIZE, \
    END_EPOCHS, PRINT_FREQ, PATH_SSD_WEIGHT, DEBUG, PATH_SAVE_WEIGHT, PATH_MODEL_WEIGHT, DATA_NUM_WORKERS, IS_TRAIN, \
    IS_EVAL
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
                                                           isdebug=DEBUG,
                                                           data_num_workers=DATA_NUM_WORKERS

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

    # 如果电脑内存充裕，可提前加载验证集数据，以免每次验证时都要重新加载一次数据，节省时间
    # val_data = get_coco_api_from_dataset(train_data_loader.dataset)
    for epoch in range(start_epoch, END_EPOCHS):
        if IS_TRAIN:
            flog.debug('---训练开始---epoch %s', epoch + 1)
            utils.train_one_epoch(model=model, optimizer=optimizer,
                                  data_loader=train_data_loader,
                                  device=device, epoch=epoch,
                                  print_freq=PRINT_FREQ,
                                  train_loss=train_loss,
                                  train_lr=learning_rate)

            lr_scheduler.step()  # 更新学习

            # 每个epoch保存
            save_weight(
                path_save=PATH_SAVE_WEIGHT,
                model=model,
                name=os.path.basename(__file__),
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch)

        if IS_EVAL:
            flog.debug('---验证开始---epoch %s', epoch + 1)
            utils.evaluate(model=model, data_loader=val_data_set_loader,
                           device=device, data_set=None,
                           mAP_list=val_map)

    '''-------------结果可视化-----------------'''
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    flog.debug('val_map %s', val_map)
    if len(val_map) != 0:
        plot_map(val_map)


if __name__ == '__main__':
    '''
    total_losses: 2.4886 (2.6816)  time: 0.7638  data: 0.0004 cpu 33-4.8G  GPU 6211-71C
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.507
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.742
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.561
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.079
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.387
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.573
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.502
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.600
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.607
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.205
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.518
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.659
    0:07:01
    '''
    '''------------------系统配置---------------------'''
    torch.multiprocessing.set_sharing_strategy('file_system')  # 多进程开文件
    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(PATH_SAVE_WEIGHT):
        try:
            os.makedirs(PATH_SAVE_WEIGHT)
        except Exception as e:
            flog.error(' %s %s', PATH_SAVE_WEIGHT, e)

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    flog.info('模型当前设备 %s', device)
    if DEBUG:
        PRINT_FREQ = 1
        PATH_SAVE_WEIGHT = None
        # PATH_FIT_WEIGHT = None
        BATCH_SIZE = 10
        # device = torch.device("cpu") # 强制CPU

        # WIN10
        DATA_NUM_WORKERS = 0
        pass

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
