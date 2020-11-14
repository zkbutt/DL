import torch
from torchvision import models

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_factory import MapDataSet, VOCDataSet
from f_tools.datas.data_pretreatment import Compose, ResizeKeep, ColorJitter, ToTensor, RandomHorizontalFlip4TS, \
    Normalization4TS, Resize, SSDCroppingPIL
from f_tools.f_torch_tools import save_weight

from f_tools.fits.f_show_fit_res import plot_loss_and_lr
from f_tools.fits.fitting.f_fit_eval_base import f_train_one_epoch, f_evaluate, f_train_one_epoch2
from f_tools.fun_od.f_boxes import nms
from object_detection.f_yolov1.CONFIG_YOLO1 import CFG
from object_detection.f_yolov1.nets.model_YOLOv1 import Yolo_v1
from object_detection.f_yolov1.train_eval_fun import LossHandler, PredictHandler

DATA_TRANSFORM = {
    "train": Compose([
        # ResizeKeep(CFG.IMAGE_SIZE),
        Resize(CFG.IMAGE_SIZE),
        # SSDCroppingPIL(),
        ColorJitter(),
        ToTensor(),
        RandomHorizontalFlip4TS(1),
        Normalization4TS(),
    ], CFG),
    "val": Compose([
        # ResizeKeep(cfg.IMAGE_SIZE),  # (h,w)
        Resize(CFG.IMAGE_SIZE),
        ToTensor(),
        Normalization4TS(),
    ], CFG)
}


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
    model = models.densenet121(pretrained=True)
    out_dim = model.classifier.in_features
    model = torch.nn.Sequential(*list(model.children())[:-1])  # 去除resnet的最后两层
    model = Yolo_v1(model, out_dim, cfg.GRID, cfg.NUM_CLASSES)

    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'densenet121'
    # f_look(model)
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
    loader_train, loader_val = None, None
    # 返回数据已预处理 返回np(batch,(3,640,640))  , np(batch,(x个选框,15维))
    if cfg.IS_TRAIN:
        dataset_train = VOCDataSet(
            cfg.PATH_DATA_ROOT,
            'train.txt',  # 正式训练要改这里
            DATA_TRANSFORM["train"],
            bbox2one=False,
            isdebug=cfg.DEBUG
        )

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.DATA_NUM_WORKERS,
            shuffle=True,
            pin_memory=True,  # 不使用虚拟内存 GPU要报错
            # drop_last=True,  # 除于batch_size余下的数据
            collate_fn=_collate_fn,
        )

    # loader_train = Data_Prefetcher(loader_train)
    if cfg.IS_EVAL:
        # class_to_idx = {'face': 1}
        # dataset_val = MapDataSet(cfg.PATH_DT_ROOT, cfg.PATH_DT_RES, class_to_idx, transforms=DATA_TRANSFORM['val'],
        #                          is_debug=cfg.DEBUG)
        # loader_val = torch.utils.data.DataLoader(
        #     dataset_val,
        #     batch_size=cfg.BATCH_SIZE,
        #     num_workers=cfg.DATA_NUM_WORKERS,
        #     # shuffle=True,
        #     pin_memory=True,  # 不使用虚拟内存 GPU要报错
        #     # drop_last=True,  # 除于batch_size余下的数据
        #     collate_fn=_collate_fn,
        # )
        pass

    # iter(data_loader).__next__()
    return loader_train, loader_val


def data_loader4mgpu(cfg):
    loader_train, loader_val, train_sampler = None, None, None
    # 返回数据已预处理 返回np(batch,(3,640,640))  , np(batch,(x个选框,15维))
    if cfg.IS_TRAIN:
        dataset_train = VOCDataSet(
            cfg.PATH_DATA_ROOT,
            'train.txt',  # 正式训练要改这里
            DATA_TRANSFORM["train"],
            bbox2one=False,
            isdebug=cfg.DEBUG
        )
        # 给每个rank按显示个数生成定义类 shuffle -> ceil(样本/GPU个数)自动补 -> 间隔分配到GPU
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train,
                                                                        shuffle=True,
                                                                        seed=20201114,
                                                                        )
        # 按定义为每一个 BATCH_SIZE 生成一批的索引
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, cfg.BATCH_SIZE, drop_last=True)

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            # batch_size=cfg.BATCH_SIZE,
            batch_sampler=train_batch_sampler,  # 按样本定义加载
            num_workers=cfg.DATA_NUM_WORKERS,
            # shuffle=True,
            pin_memory=True,  # 不使用虚拟内存 GPU要报错
            # drop_last=True,  # 除于batch_size余下的数据
            collate_fn=_collate_fn,
        )

    # loader_train = Data_Prefetcher(loader_train)
    if cfg.IS_EVAL:
        class_to_idx = {'face': 1}
        dataset_val = MapDataSet(cfg.PATH_DT_ROOT, cfg.PATH_DT_RES, class_to_idx, transforms=DATA_TRANSFORM['val'],
                                 is_debug=cfg.DEBUG)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val,
                                                                      shuffle=False,
                                                                      )

        loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=cfg.BATCH_SIZE,
            sampler=val_sampler,
            num_workers=cfg.DATA_NUM_WORKERS,
            # shuffle=True,
            pin_memory=True,  # 不使用虚拟内存 GPU要报错
            # drop_last=True,  # 除于batch_size余下的数据
            collate_fn=_collate_fn,
        )
        pass

    # iter(data_loader).__next__()
    return loader_train, loader_val, train_sampler


def train_eval(cfg, start_epoch, model, losser, optimizer, lr_scheduler=None,
               loader_train=None, loader_val=None, device=torch.device('cpu'),
               train_sampler=None,
               ):
    # 返回特图 h*w,4 anchors 是每个特图的长宽个 4维整框 这里是 x,y,w,h 调整系数l
    # 特图的步距
    train_loss = []
    learning_rate = []

    for epoch in range(start_epoch, cfg.END_EPOCH):
        # if epoch < 5:
        #     # 主干网一般要冻结
        #     for param in model.backbone.parameters():
        #         param.requires_grad = False
        # else:
        #     # 解冻后训练
        #     for param in model.backbone.parameters():
        #         param.requires_grad = True
        if cfg.IS_TRAIN:
            process = LossHandler(model, device, losser, cfg.GRID, cfg.NUM_CLASSES)

            flog.info('训练开始 %s', epoch + 1)
            loss = f_train_one_epoch2(
                model=model, loss=losser,
                data_loader=loader_train, loss_process=process, optimizer=optimizer,
                epoch=epoch, cfg=cfg,
                lr_scheduler=None,
                ret_train_loss=train_loss, ret_train_lr=learning_rate,
                train_sampler=train_sampler,
            )

            if lr_scheduler is not None:
                lr_scheduler.step(loss)  # 更新学习

        '''------------------模型验证---------------------'''
        if cfg.IS_EVAL:
            flog.info('验证开始 %s', epoch + 1)
            predict_handler = PredictHandler(model, device, anchors,
                                             threshold_conf=0.5, threshold_nms=0.3)
            res_eval = []
            f_evaluate(
                data_loader=loader_val,
                predict_handler=predict_handler,
                epoch=epoch,
                res_eval=res_eval)
            if cfg.IS_RUN_ONE:
                return
            # del coco_res_bboxs
    if cfg.IS_TRAIN:
        '''-------------结果可视化-----------------'''
        if len(train_loss) != 0 and len(learning_rate) != 0:
            plot_loss_and_lr(train_loss, learning_rate)
