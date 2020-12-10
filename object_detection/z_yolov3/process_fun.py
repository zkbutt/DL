import torch
from torch import optim
from torchvision import models

from f_pytorch.tools_model.f_layer_get import ModelOuts4Mobilenet_v2
from f_pytorch.tools_model.model_look import f_look_model
from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_factory import MapDataSet
from f_tools.datas.f_coco.convert_data.coco_dataset import CocoDataset
from f_tools.f_torch_tools import load_weight
from f_tools.fits.fitting.f_fit_eval_base import f_train_one_epoch4, f_evaluate4coco2
from f_tools.fun_od.f_boxes import fmatch4yolo1
from f_tools.pic.enhance.f_data_pretreatment import Compose, ColorJitter, ToTensor, RandomHorizontalFlip4TS, \
    Normalization4TS, Resize, ResizeKeep
from object_detection.z_yolov3.nets.net_yolov3 import YoloV3SPP


def cre_data_transform(cfg):
    if cfg.IS_MOSAIC:
        data_transform = {
            "train": Compose([
                # ResizeKeep(cfg.IMAGE_SIZE),  # (h,w)
                # Resize(cfg.IMAGE_SIZE),
                ColorJitter(),
                ToTensor(),
                RandomHorizontalFlip4TS(0.5),
                Normalization4TS(),
            ], cfg),
            "val": Compose([
                # ResizeKeep(cfg.IMAGE_SIZE),  # (h,w)
                Resize(cfg.IMAGE_SIZE),
                ToTensor(),
                Normalization4TS(),
            ], cfg)
        }
    else:
        data_transform = {
            "train": Compose([
                ResizeKeep(cfg.IMAGE_SIZE),  # (h,w)
                # Resize(cfg.IMAGE_SIZE),
                ColorJitter(),
                ToTensor(),
                RandomHorizontalFlip4TS(0.5),
                Normalization4TS(),
            ], cfg),
            "val": Compose([
                ResizeKeep(cfg.IMAGE_SIZE),  # (h,w)
                # Resize(cfg.IMAGE_SIZE),
                ToTensor(),
                Normalization4TS(),
            ], cfg)
        }

    return data_transform


def _collate_fn(batch_datas):
    # 数据组装
    _t = batch_datas[0][0]
    # images = torch.empty((len(batch_datas), *_t.shape), device=_t.device)
    images = torch.empty((len(batch_datas), *_t.shape)).to(_t)
    targets = []
    for i, (img, taget) in enumerate(batch_datas):
        images[i] = img
        targets.append(taget)
    return images, targets


def fdatas_l2(batch_data, device, cfg=None):
    images, targets = batch_data
    images = images.to(device)

    for target in targets:
        target['boxes'] = target['boxes'].to(device)
        target['labels'] = target['labels'].to(device)
        # target['size'] = target['size'].to(device)
        # target['image_id'] = target['image_id'].to(device)

    return images, targets


def init_model(cfg, device, id_gpu=None):
    model = models.mobilenet_v2(pretrained=True)
    model = ModelOuts4Mobilenet_v2(model)
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'mobilenet_v2'

    model = YoloV3SPP(backbone=model, nums_anc=cfg.NUMS_ANC, num_classes=cfg.NUM_CLASSES,
                      dims_rpn_in=model.dims_out, device=device, cfg=cfg, is_spp=True)
    # f_look_model(model, input=(1, 3, *cfg.IMAGE_SIZE))

    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        for name, param in model.backbone.named_parameters():
            param.requires_grad_(False)
        # 除最后的全连接层外，其他权重全部冻结
        # if "fc" not in name:
        #     param.requires_grad_(False)

    if id_gpu is not None:
        # 多GPU初始化
        is_mgpu = True
        if cfg.SYSNC_BN:
            # 不冻结权重的情况下可, 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        else:
            model.to(device)  # 这个不需要加等号
        # 转为DDP模型
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[id_gpu], find_unused_parameters=True)
    else:
        model.to(device)  # 这个不需要加等号
        is_mgpu = False
    # model.train()
    # ------------------------模型完成-------------------------------

    pg = model.parameters()
    # 最初学习率
    lr0 = 1e-3
    lrf = lr0 / 100
    # 权重衰减(如L2惩罚)(默认: 0)
    optimizer = optim.Adam(pg, lr0)
    # optimizer = optim.Adam(pg, lr0, weight_decay=5e-4)
    # 两次不上升，降低一半
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=1, verbose=True)
    # start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, None, lr_scheduler, device, is_mgpu=is_mgpu)
    start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device, is_mgpu=is_mgpu)

    model.cfg = cfg
    return model, optimizer, lr_scheduler, start_epoch


def data_loader(cfg, is_mgpu=False):
    data_transform = cre_data_transform(cfg)

    loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler = [None] * 5
    # 返回数据已预处理 返回np(batch,(3,640,640))  , np(batch,(x个选框,15维))
    if cfg.IS_TRAIN:
        dataset_train = CocoDataset(
            path_coco_target=cfg.PATH_COCO_TARGET_TRAIN,
            path_img=cfg.PATH_IMG_TRAIN,
            # mode='keypoints',
            mode='bbox',
            data_type='train2017',
            transform=data_transform['train'],
            is_mosaic=cfg.IS_MOSAIC,
            is_debug=cfg.DEBUG,
            cfg=cfg
        )
        # __d = dataset_train[0]  # 调试
        if is_mgpu:  # DataLoader 不一样
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
        else:
            loader_train = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=cfg.BATCH_SIZE,
                num_workers=cfg.DATA_NUM_WORKERS,
                # shuffle=True,
                pin_memory=True,  # 不使用虚拟内存 GPU要报错
                # drop_last=True,  # 除于batch_size余下的数据
                collate_fn=_collate_fn,
            )

    if cfg.IS_FMAP_EVAL:
        class_to_idx = {'face': 1}
        dataset_val = MapDataSet(path_imgs=cfg.PATH_EVAL_IMGS,
                                 path_eval_info=cfg.PATH_EVAL_INFO,
                                 class_to_idx=class_to_idx,
                                 transforms=data_transform['val'],
                                 is_debug=cfg.DEBUG)
        loader_val_fmap = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.DATA_NUM_WORKERS,
            # shuffle=True,
            pin_memory=True,  # 不使用虚拟内存 GPU要报错
            # drop_last=True,  # 除于batch_size余下的数据
            collate_fn=_collate_fn,
        )
    elif cfg.IS_COCO_EVAL:
        dataset_val = CocoDataset(
            path_coco_target=cfg.PATH_COCO_TARGET_EVAL,
            path_img=cfg.PATH_IMG_EVAL,
            # mode='keypoints',
            mode='bbox',
            data_type='val2017',
            transform=data_transform['val'],
            is_mosaic=False,  # 必须为 False 否则ID出错
            is_debug=cfg.DEBUG,
            cfg=cfg
        )

        if is_mgpu:
            # 给每个rank按显示个数生成定义类 shuffle -> ceil(样本/GPU个数)自动补 -> 间隔分配到GPU
            eval_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val,
                                                                           shuffle=True,
                                                                           seed=20201114,
                                                                           )
            # 按定义为每一个 BATCH_SIZE 生成一批的索引
            eval_batch_sampler = torch.utils.data.BatchSampler(eval_sampler,
                                                               int(cfg.BATCH_SIZE * 1.5),
                                                               drop_last=False,  # 不要扔掉否则验证不齐
                                                               )

            loader_val_coco = torch.utils.data.DataLoader(
                dataset_val,
                # batch_size=cfg.BATCH_SIZE,
                batch_sampler=eval_batch_sampler,  # 按样本定义加载
                num_workers=cfg.DATA_NUM_WORKERS,
                # shuffle=True,
                pin_memory=True,  # 不使用虚拟内存 GPU要报错
                # drop_last=True,  # 除于batch_size余下的数据
                collate_fn=_collate_fn,
            )
        else:
            loader_val_coco = torch.utils.data.DataLoader(
                dataset_val,
                batch_size=int(cfg.BATCH_SIZE * 1.5),
                num_workers=cfg.DATA_NUM_WORKERS,
                # shuffle=True,
                pin_memory=True,  # 不使用虚拟内存 GPU要报错
                # drop_last=True,  # 除于batch_size余下的数据
                collate_fn=_collate_fn,
            )
    # iter(data_loader).__next__()
    return loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler


def train_eval(start_epoch, model, optimizer, lr_scheduler=None,
               loader_train=None, loader_val_fmap=None, loader_val_coco=None,
               device=torch.device('cpu'), train_sampler=None, eval_sampler=None,
               tb_writer=None,
               ):
    cfg = model.cfg

    fun_datas_l2 = fdatas_l2
    for epoch in range(start_epoch, cfg.END_EPOCH):

        if cfg.IS_TRAIN:
            model.train()
            flog.info('训练开始 %s', epoch + 1)
            log_dict = f_train_one_epoch4(
                model=model,
                fun_datas_l2=fun_datas_l2,
                data_loader=loader_train,
                optimizer=optimizer, epoch=epoch,
                lr_scheduler=lr_scheduler,
                train_sampler=train_sampler,
                tb_writer=tb_writer,
                device=device,
            )

            if lr_scheduler is not None:
                lr_scheduler.step(log_dict['loss_total'])  # 更新学习

        if model.cfg.IS_COCO_EVAL:
            flog.info('COCO 验证开始 %s', epoch + 1)
            model.eval()
            # with torch.no_grad():
            mode = 'bbox'
            res_eval = []
            f_evaluate4coco2(
                model=model,
                fun_datas_l2=fun_datas_l2,
                data_loader=loader_val_coco,
                epoch=epoch,
                tb_writer=tb_writer,
                res_eval=res_eval,
                mode=mode,
                device=device,
                eval_sampler=eval_sampler,
            )
            # return
        # if model.cfg.IS_FMAP_EVAL:
        #     flog.info('FMAP 验证开始 %s', epoch + 1)
        #     res_eval = []
        #     f_evaluate4fmap(
        #         data_loader=loader_val_fmap,
        #         predict_handler=predict_handler,
        #         epoch=epoch,
        #         res_eval=res_eval)
        #     path_dt_info = loader_val_fmap.dataset.path_dt_info
        #     path_gt_info = loader_val_fmap.dataset.path_gt_info
        #     path_imgs = loader_val_fmap.dataset.path_imgs
        #     f_do_fmap(path_gt=path_gt_info, path_dt=path_dt_info, path_img=path_imgs,
        #               confidence=cfg.THRESHOLD_PREDICT_CONF,
        #               iou_map=[], ignore_classes=[], console_pinter=True,
        #               plot_res=False, animation=False)
        #     return