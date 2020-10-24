import os

import torch
from torch import optim

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_pretreatment import Compose, ToTensor, Resize, Normalization4TS, \
    ColorJitter, RandomHorizontalFlip4TS, ResizeKeep
from f_tools.datas.f_coco.convert_data.coco_dataset import CocoDataset
from f_tools.f_torch_tools import save_weight
from f_tools.fits.f_lossfun import KeypointsLoss
from f_tools.fits.f_show_fit_res import plot_loss_and_lr
from f_tools.fits.fitting.f_fit_eval_base import f_train_one_epoch, f_evaluate
from f_tools.fun_od.f_anc import AnchorsFound
from object_detection.f_retinaface.utils.train_eval_fun import PredictHandler
from object_detection.retinaface.nets.retinaface import RetinaFace
from object_detection.retinaface.CONFIG_RETINAFACE import PATH_SAVE_WEIGHT, PATH_DATA_ROOT, DEBUG, IMAGE_SIZE, \
    MOBILENET025, PATH_FIT_WEIGHT, NEGATIVE_RATIO, NEG_IOU_THRESHOLD, END_EPOCH, \
    PRINT_FREQ, BATCH_SIZE, VARIANCE, LOSS_COEFFICIENT, DATA_NUM_WORKERS, IS_EVAL, IS_TRAIN, NUM_CLASSES
from object_detection.retinaface.utils.retinaface_process import LossProcess, ForecastProcess

if DEBUG:
    # device = torch.device("cpu")
    PRINT_FREQ = 1
    PATH_SAVE_WEIGHT = None
    BATCH_SIZE = 2
    DATA_NUM_WORKERS = 1
    pass
else:
    torch.multiprocessing.set_sharing_strategy('file_system')  # 多进程开文件


def collate_fn(batch_datas):
    '''
    loader输出数据组装
    :param batch_datas:
        list(
            batch(
                img: (h,w,c),
                target: dict{
                    image_id: int,
                    bboxs: tensor(num_anns, 4),
                    labels: tensor(num_anns,类别数),
                    keypoints: tensor(num_anns,10),
                }
            )
        )
    :return:
        images:tensor(batch,c,h,w)
        list( # batch个
            target: dict{
                    image_id: int,
                    bboxs: tensor(num_anns, 4),
                    labels: tensor(num_anns,类别数),
                    keypoints: tensor(num_anns,10),
                }
            )
    '''
    _t = batch_datas[0][0]
    # images = torch.empty((len(batch_datas), *_t.shape), device=_t.device)
    images = torch.empty((len(batch_datas), *_t.shape)).to(_t)
    targets = []
    for i, (img, taget) in enumerate(batch_datas):
        images[i] = img
        targets.append(taget)
    return images, targets


def data_loader(device):
    loader_train, dataset_val, loader_val = None, None, None
    # 返回数据已预处理 返回np(batch,(3,640,640))  , np(batch,(x个选框,15维))
    data_transform = {
        "train": Compose([
            ResizeKeep(IMAGE_SIZE),  # (h,w)
            ColorJitter(),
            ToTensor(),
            RandomHorizontalFlip4TS(),
            Normalization4TS(),
        ]),
        "val": Compose([
            ResizeKeep(IMAGE_SIZE),  # (h,w)
            ToTensor(),
            Normalization4TS(),
        ])
    }
    '''
      归一化后 toTensor
          image_mean = [0.485, 0.456, 0.406]
          image_std = [0.229, 0.224, 0.225]
    '''

    if IS_TRAIN:
        dataset_train = CocoDataset(PATH_DATA_ROOT, 'keypoints', 'train2017',
                                    device, data_transform['val'],
                                    is_debug=DEBUG)

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=BATCH_SIZE,
            num_workers=DATA_NUM_WORKERS,
            shuffle=True,
            pin_memory=True,  # 不使用虚拟内存 GPU要报错
            drop_last=True,  # 除于batch_size余下的数据
            collate_fn=collate_fn,
        )

    # loader_train = Data_Prefetcher(loader_train)
    if IS_EVAL:
        dataset_val = CocoDataset(PATH_DATA_ROOT, 'bboxs', 'val2017',
                                  device, data_transform['train'],
                                  is_debug=DEBUG)

        loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=BATCH_SIZE,
            num_workers=DATA_NUM_WORKERS,
            # shuffle=True,
            pin_memory=True,  # 不使用虚拟内存 GPU要报错
            drop_last=True,  # 除于batch_size余下的数据
            collate_fn=collate_fn,
        )

    # iter(data_loader).__next__()
    return loader_train, loader_val


def model_init(claxx, device):
    # num_classes = len(dataset_train.coco.getCatIds())  # 根据数据集取类别数
    model = RetinaFace(claxx.MODEL_NAME,
                       claxx.FILE_WEIGHT,
                       claxx.IN_CHANNELS, claxx.OUT_CHANNEL,
                       claxx.RETURN_LAYERS, claxx.ANCHOR_NUM,
                       NUM_CLASSES)
    # if torchvision._is_tracing() 判断训练模式
    # self.training 判断训练模式
    model.train()  # 启用 BatchNormalization 和 Dropout
    model.to(device)  # 模型装入显存
    # 生成正方形anc
    anchors = AnchorsFound(IMAGE_SIZE, claxx.ANCHORS_SIZE, claxx.FEATURE_MAP_STEPS, claxx.ANCHORS_CLIP).get_anchors()
    anchors = anchors.to(device)
    losser = KeypointsLoss(anchors, NEGATIVE_RATIO, VARIANCE, LOSS_COEFFICIENT)
    # 权重衰减(如L2惩罚)(默认: 0)
    optimizer = optim.Adam(model.parameters(), 1e-3, weight_decay=5e-4)
    # 在发现loss不再降低或者acc不再提高之后，降低学习率
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    # --------feadre权重加载-----------
    # 重新训练代码
    checkpoint = torch.load(PATH_FIT_WEIGHT, map_location=device)
    del checkpoint['ClassHead.0.conv1x1.weight']
    del checkpoint['ClassHead.0.conv1x1.bias']
    del checkpoint['ClassHead.1.conv1x1.weight']
    del checkpoint['ClassHead.1.conv1x1.bias']
    del checkpoint['ClassHead.2.conv1x1.weight']
    del checkpoint['ClassHead.2.conv1x1.bias']
    model.load_state_dict(checkpoint, strict=False)  # 定制一个后面改
    start_epoch = 0

    # start_epoch = load_weight(PATH_FIT_WEIGHT, model, optimizer, lr_scheduler, device)
    # 单层修改学习率
    # optimizer.param_groups[0]['lr'] = 1e-5
    # 多层
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = 1e-1
    return model, anchors, losser, optimizer, lr_scheduler, start_epoch


def trainning(start_epoch, model, device, anchors, losser, optimizer, lr_scheduler, loader_train, loader_val):
    # 返回特图 h*w,4 anchors 是每个特图的长宽个 4维整框 这里是 x,y,w,h 调整系数
    # 特图的步距
    train_loss = []
    learning_rate = []
    for epoch in range(start_epoch, END_EPOCH):
        # if epoch < 5:
        #     # 主干网一般要冻结
        #     for param in model.body.parameters():
        #         param.requires_grad = False
        # else:
        #     # 解冻后训练
        #     for param in model.body.parameters():
        #         param.requires_grad = True
        if IS_TRAIN:
            process = LossProcess(model, device, anchors, losser, NEG_IOU_THRESHOLD)

            flog.info('训练开始 %s', epoch + 1)
            loss = f_train_one_epoch(data_loader=loader_train, loss_process=process, optimizer=optimizer,
                                     epoch=epoch, end_epoch=END_EPOCH, print_freq=PRINT_FREQ,
                                     ret_train_loss=train_loss, ret_train_lr=learning_rate,
                                     )

            lr_scheduler.step(loss)  # 更新学习

            # 每个epoch保存
            save_weight(
                path_save=PATH_SAVE_WEIGHT,
                model=model,
                name=os.path.basename(__file__),
                loss=loss,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch)

        '''------------------模型验证---------------------'''
        if IS_EVAL:
            flog.info('验证开始 %s', epoch + 1)
            predict_handler = PredictHandler(model, device, anchors,
                                             threshold_conf=0.5, threshold_nms=0.3)

            coco_res_bboxs = []
            f_evaluate(
                data_loader=loader_val,
                predict_handler=predict_handler,
                epoch=epoch,
                coco_res_bboxs=coco_res_bboxs)
            # del coco_res_bboxs
    if IS_TRAIN:
        '''-------------结果可视化-----------------'''
        if len(train_loss) != 0 and len(learning_rate) != 0:
            plot_loss_and_lr(train_loss, learning_rate)
