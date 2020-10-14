import os

import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_pretreatment import Compose, ResizeKeep, ColorJitter, ToTensor, RandomHorizontalFlip4TS, \
    Normalization, Resize
from f_tools.datas.f_coco.convert_data.coco_dataset import CocoDataset

from f_tools.fits.f_show_fit_res import plot_loss_and_lr
from f_tools.fits.fitting.f_fit_eval_base import f_train_one_epoch
from f_tools.fun_od.f_boxes import nms
from object_detection.f_fit_tools import save_weight
from object_detection.f_retinaface.nets.mobilenet025 import MobileNetV1
from object_detection.f_retinaface.nets.retinaface import RetinaFace
from object_detection.f_retinaface.utils.train_eval_fun import LossHandler


def output_res(p_boxes, p_keypoints, p_scores, threshold_conf=0.5, threshold_nms=0.3):
    '''
    已修复的框 点 和对应的分数
        1. 经分数过滤
        2. 经NMS 出最终结果
    :param p_boxes:
    :param p_keypoints:
    :param p_scores:
    :return:
    '''
    mask = p_scores >= threshold_conf
    p_boxes = p_boxes[mask]
    p_scores = p_scores[mask]
    p_keypoints = p_keypoints[mask]

    if p_scores.shape[0] == 0:
        flog.error('threshold_conf 过滤后 没有目标 %s', threshold_conf)
        return None, None, None

    flog.debug('threshold_conf 过滤后有 %s 个', p_scores.shape[0])
    # 2 . 根据得分对框进行从大到小排序。
    keep = nms(p_boxes, p_scores, threshold_nms)
    flog.debug('threshold_nms 过滤后有 %s 个', len(keep))
    p_boxes = p_boxes[keep]
    p_scores = p_scores[keep]
    p_keypoints = p_keypoints[keep]
    return p_boxes, p_keypoints, p_scores


def init_model(cfg):
    backbone = MobileNetV1()
    model = RetinaFace(backbone, cfg.IN_CHANNELS, cfg.OUT_CHANNEL, cfg.RETURN_LAYERS, cfg.ANCHOR_NUM, cfg.NUM_CLASSES)

    return model


def _collate_fn(batch_datas):
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


def data_loader(cfg, device):
    loader_train, dataset_val, loader_val = None, None, None
    # 返回数据已预处理 返回np(batch,(3,640,640))  , np(batch,(x个选框,15维))
    data_transform = {
        "train": Compose([
            # ResizeKeep(cfg.IMAGE_SIZE),  # (h,w)
            Resize(cfg.IMAGE_SIZE),
            ColorJitter(),
            ToTensor(),
            RandomHorizontalFlip4TS(),
            Normalization(),
        ]),
        "val": Compose([
            # ResizeKeep(cfg.IMAGE_SIZE),  # (h,w)
            Resize(cfg.IMAGE_SIZE),
            ToTensor(),
            Normalization(),
        ])
    }
    '''
      归一化后 toTensor
          image_mean = [0.485, 0.456, 0.406]
          image_std = [0.229, 0.224, 0.225]
    '''

    if cfg.IS_TRAIN:
        dataset_train = CocoDataset(cfg.PATH_DATA_ROOT, 'keypoints', 'train2017',
                                    device, data_transform['val'],
                                    is_debug=cfg.DEBUG)

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.DATA_NUM_WORKERS,
            shuffle=True,
            pin_memory=True,  # 不使用虚拟内存 GPU要报错
            drop_last=True,  # 除于batch_size余下的数据
            collate_fn=_collate_fn,
        )

    # loader_train = Data_Prefetcher(loader_train)
    if cfg.IS_EVAL:
        dataset_val = CocoDataset(cfg.PATH_DATA_ROOT, 'bboxs', 'val2017',
                                  device, data_transform['train'],
                                  is_debug=cfg.DEBUG)

        loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.DATA_NUM_WORKERS,
            # shuffle=True,
            pin_memory=True,  # 不使用虚拟内存 GPU要报错
            drop_last=True,  # 除于batch_size余下的数据
            collate_fn=_collate_fn,
        )

    # iter(data_loader).__next__()
    return loader_train, loader_val


def train_eval(cfg, start_epoch, model, anchors, losser, optimizer, lr_scheduler=None,
               loader_train=None, loader_val=None, device=torch.device('cpu'), ):
    # 返回特图 h*w,4 anchors 是每个特图的长宽个 4维整框 这里是 x,y,w,h 调整系数l
    # 特图的步距
    train_loss = []
    learning_rate = []

    for epoch in range(start_epoch, cfg.END_EPOCH):
        # if epoch < 5:
        #     # 主干网一般要冻结
        #     for param in model.body.parameters():
        #         param.requires_grad = False
        # else:
        #     # 解冻后训练
        #     for param in model.body.parameters():
        #         param.requires_grad = True
        if cfg.IS_TRAIN:
            process = LossHandler(model, device, anchors, losser, cfg.NEG_IOU_THRESHOLD)

            flog.info('训练开始 %s', epoch + 1)
            loss = f_train_one_epoch(data_loader=loader_train, loss_process=process, optimizer=optimizer,
                                     epoch=epoch, end_epoch=cfg.END_EPOCH, print_freq=cfg.PRINT_FREQ,
                                     ret_train_loss=train_loss, ret_train_lr=learning_rate,
                                     is_mixture_fix=cfg.IS_MIXTURE_FIX,
                                     )

            if lr_scheduler is not None:
                lr_scheduler.step(loss)  # 更新学习

            # 每个epoch保存
            save_weight(
                path_save=cfg.PATH_SAVE_WEIGHT,
                model=model,
                name=cfg.SAVE_FILE_NAME,
                loss=loss,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch)

        '''------------------模型验证---------------------'''
        # if cfg.IS_EVAL:
        #     flog.info('验证开始 %s', epoch + 1)
        #     forecast_process = ForecastProcess(
        #         model=model, device=device, ancs=anchors,
        #         img_size=IMAGE_SIZE, coco=loader_val.dataset.coco, eval_mode='bboxs')
        #
        #     coco_res_bboxs = []
        #     f_evaluate(
        #         data_loader=loader_val,
        #         forecast_process=forecast_process,
        #         epoch=epoch,
        #         coco_res_bboxs=coco_res_bboxs)
        #     # del coco_res_bboxs
    if cfg.IS_TRAIN:
        '''-------------结果可视化-----------------'''
        if len(train_loss) != 0 and len(learning_rate) != 0:
            plot_loss_and_lr(train_loss, learning_rate)
