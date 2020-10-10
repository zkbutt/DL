import os

import torch
import torch.optim as optim

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_pretreatment import Compose, RandomHorizontalFlip, ToTensor, Resize
from f_tools.fits.f_lossfun import KeypointsLoss
from f_tools.fits.f_show import plot_loss_and_lr
from f_tools.fits.fitting.f_fit_retinaface import f_train_one_epoch, f_evaluate
from f_tools.fun_od.f_anc import AnchorsFound
from f_tools.datas.f_coco.convert_data.coco_dataset import CocoDataset

from object_detection.f_fit_tools import sysconfig, load_weight, save_weight
from object_detection.retinaface.CONFIG_RETINAFACE import PATH_SAVE_WEIGHT, PATH_DATA_ROOT, DEBUG, IMAGE_SIZE, \
    MOBILENET025, PATH_FIT_WEIGHT, NEGATIVE_RATIO, NEG_IOU_THRESHOLD, END_EPOCH, \
    PRINT_FREQ, BATCH_SIZE, VARIANCE, LOSS_COEFFICIENT, DATA_NUM_WORKERS, IS_EVAL, IS_TRAIN
from object_detection.retinaface.nets.retinaface import RetinaFace
from object_detection.retinaface.utils.retinaface_fit import LossProcess, ForecastProcess

if __name__ == "__main__":
    '''
    BATCH_SIZE = 48
    time: 2.1451    data: 1.5087
    cpu 9% 3.1G-1.4G
    GPU: 4500
    
    6进程
    cpu 19-7.2
    time: 0.8387  data: 0.2450
    
    10 进程
    cpu 20-10.4
    time: 0.6072  data: 0.0168 锁页内存0.001秒
    gpu 4435 60
    '''

    '''------------------系统配置---------------------'''
    torch.multiprocessing.set_sharing_strategy('file_system')  # 多进程开文件
    device = sysconfig(PATH_SAVE_WEIGHT)

    if DEBUG:
        # device = torch.device("cpu")
        PRINT_FREQ = 1
        PATH_SAVE_WEIGHT = None
        BATCH_SIZE = 10
        pass

    # claxx = RESNET50  # 这里根据实际情况改
    claxx = MOBILENET025  # 这里根据实际情况改

    '''---------------数据加载及处理--------------'''
    # 返回数据已预处理 返回np(batch,(3,640,640))  , np(batch,(x个选框,15维))
    data_transform = {
        "train": Compose([
            Resize(IMAGE_SIZE),  # (h,w)
            RandomHorizontalFlip(1),
            ToTensor(),
        ]),
        "val": Compose([ToTensor()])
    }
    '''
      归一化后 toTensor
          image_mean = [0.485, 0.456, 0.406]
          image_std = [0.229, 0.224, 0.225]
    '''


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


    if IS_TRAIN:
        dataset_train = CocoDataset(PATH_DATA_ROOT, 'keypoints', 'train2017',
                                    device, data_transform['train'],
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

    '''------------------模型定义---------------------'''
    num_classes = len(dataset_train.coco.getCatIds())  # 根据数据集取类别数
    model = RetinaFace(claxx.MODEL_NAME,
                       claxx.FILE_WEIGHT,
                       claxx.IN_CHANNELS, claxx.OUT_CHANNEL,
                       claxx.RETURN_LAYERS, claxx.ANCHOR_NUM,
                       num_classes)
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

    # feadre权重加载
    start_epoch = load_weight(PATH_FIT_WEIGHT, model, optimizer, lr_scheduler)

    '''------------------模型训练---------------------'''

    # 返回特图 h*w,4 anchors 是每个特图的长宽个 4维整框 这里是 x,y,w,h 调整系数
    # 特图的步距
    train_loss = []
    learning_rate = []
    val_map = []

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
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch)

        '''------------------模型验证---------------------'''
        if IS_EVAL:
            flog.info('验证开始 %s', epoch + 1)
            forecast_process = ForecastProcess(
                model=model, device=device, ancs=anchors,
                img_size=IMAGE_SIZE, coco=dataset_val.coco, eval_mode='bboxs')

            coco_res_bboxs = []
            f_evaluate(
                data_loader=loader_val,
                forecast_process=forecast_process,
                epoch=epoch,
                coco_res_bboxs=coco_res_bboxs)
            # del coco_res_bboxs

    if IS_TRAIN:
        '''-------------结果可视化-----------------'''
        if len(train_loss) != 0 and len(learning_rate) != 0:
            plot_loss_and_lr(train_loss, learning_rate)

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
