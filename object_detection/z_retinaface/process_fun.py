import torch
from torch import optim
from torchvision import models

from f_pytorch.tools_model.f_layer_get import ModelOut4Densenet121, ModelOuts4Mobilenet_v2
from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_factory import MapDataSet
from f_tools.datas.f_coco.convert_data.coco_dataset import CocoDataset
from f_tools.datas.f_map.map_go import f_do_fmap
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_lossfun import LossRetinaface
from f_tools.fun_od.f_anc import FAnchors_v2

from f_tools.fits.fitting.f_fit_eval_base import f_train_one_epoch3, f_evaluate4fmap, f_evaluate4coco
from f_tools.fun_od.f_boxes import nms
from f_tools.pic.enhance.f_data_pretreatment import RandomHorizontalFlip4TS, Normalization4TS, Compose, ColorJitter, \
    ToTensor, Resize
from object_detection.z_retinaface.fun_train_eval import LossHandler, PredictHandler
from object_detection.z_retinaface.nets.net_retinaface import RetinaFace


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


def init_model(cfg, device, id_gpu=None):
    # model = models.densenet121(pretrained=False)
    # cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'densenet121'
    # dims_out = [512, 1024, 1024]
    # ret_name_dict = {'denseblock2': 1, 'denseblock3': 2, 'denseblock4': 3}
    # model = ModelOut4Densenet121(model, 'features', ret_name_dict)

    model = models.mobilenet_v2(pretrained=True)
    model = ModelOuts4Mobilenet_v2(model)
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'mobilenet_v2'

    # model = models.mnasnet1_3(pretrained=True)
    # dims_out = [512, 1024, 1280]
    # ret_name_dict = {'denseblock2': 1, 'denseblock3': 2, 'denseblock4': 3}
    # model = ModelOut4Densenet121(model, 'features', ret_name_dict)

    # from f_pytorch.tools_model.model_look import f_look_model
    # f_look_model(model, input=(1, 3, 416, 416))
    # # conv 可以取 in_channels 不支持数组层

    use_keypoint = True

    model = RetinaFace(backbone=model, num_classes=cfg.NUM_CLASSES, anchor_num=cfg.NUMS_ANC[0],
                       in_channels_fpn=model.dims_out, use_keypoint=use_keypoint)
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
    model.train()

    # ------------------------自定义backbone完成-------------------------------
    anc_obj = FAnchors_v2(cfg.IMAGE_SIZE, cfg.ANC_SCALE, cfg.FEATURE_MAP_STEPS,
                          anchors_clip=cfg.ANCHORS_CLIP, is_xymid=True, is_real_size=False,
                          device=device)
    if use_keypoint:
        losser = LossRetinaface(anc_obj.ancs.to(device), cfg.LOSS_WEIGHT, cfg.NEG_RATIO, cfg)
    else:
        losser = LossRetinaface(anc_obj.ancs, cfg.LOSS_WEIGHT, cfg.NEG_RATIO, cfg)

    pg = model.parameters()
    # 最初学习率
    lr0 = 1e-3
    lrf = lr0 / 100
    # 权重衰减(如L2惩罚)(默认: 0)
    optimizer = optim.Adam(pg, lr0, weight_decay=5e-4)
    # 两次不上升，降低一半
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    start_epoch = load_weight(cfg.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device, is_mgpu=is_mgpu)

    model.cfg = cfg
    return model, losser, optimizer, lr_scheduler, start_epoch, anc_obj


def _collate_fn(batch_datas):
    _t = batch_datas[0][0]
    # images = torch.empty((len(batch_datas), *_t.shape), device=_t.device)
    images = torch.empty((len(batch_datas), *_t.shape)).to(_t)
    targets = []
    for i, (img, taget) in enumerate(batch_datas):
        images[i] = img
        targets.append(taget)
    return images, targets


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
                # ResizeKeep(cfg.IMAGE_SIZE),  # (h,w)
                Resize(cfg.IMAGE_SIZE),
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

    return data_transform


def data_loader(cfg, is_mgpu=False):
    data_transform = cre_data_transform(cfg)

    loader_train, loader_val_fmap, loader_val_coco, train_sampler = [None] * 4
    # 返回数据已预处理 返回np(batch,(3,640,640))  , np(batch,(x个选框,15维))
    if cfg.IS_TRAIN:
        dataset_train = CocoDataset(cfg.PATH_DATA_ROOT,
                                    mode='keypoints',
                                    # mode='bbox',
                                    data_type='train2017',
                                    transform=data_transform['train'],
                                    is_mosaic=cfg.IS_MOSAIC,
                                    is_debug=cfg.DEBUG,
                                    cfg=cfg
                                    )
        # __d = dataset_train[0]  # 调试
        if is_mgpu:
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
        dataset_val = CocoDataset(cfg.PATH_DATA_ROOT,
                                  # mode='keypoints',
                                  mode='bbox',
                                  data_type='val2017',
                                  transform=data_transform['val'],
                                  is_debug=cfg.DEBUG,
                                  cfg=cfg
                                  )
        loader_val_coco = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.DATA_NUM_WORKERS,
            # shuffle=True,
            pin_memory=True,  # 不使用虚拟内存 GPU要报错
            # drop_last=True,  # 除于batch_size余下的数据
            collate_fn=_collate_fn,
        )
    # iter(data_loader).__next__()
    return loader_train, loader_val_fmap, loader_val_coco, train_sampler


def train_eval(start_epoch, model, anc_obj, losser, optimizer, lr_scheduler=None,
               loader_train=None,
               loader_val_fmap=None,
               loader_val_coco=None,
               device=torch.device('cpu'),
               train_sampler=None, tb_writer=None,
               ):
    cfg = model.cfg
    for epoch in range(start_epoch, cfg.END_EPOCH):

        if cfg.IS_TRAIN:
            process = LossHandler(model, device, anchors=anc_obj.ancs, losser=losser,
                                  neg_iou_threshold=cfg.THRESHOLD_NEG_IOU)

            flog.info('训练开始 %s', epoch + 1)
            log_dict = f_train_one_epoch3(model, loader_train,
                                          loss_process=process,
                                          optimizer=optimizer,
                                          epoch=epoch,
                                          lr_scheduler=lr_scheduler,
                                          train_sampler=train_sampler,
                                          tb_writer=tb_writer,
                                          )

            if lr_scheduler is not None:
                lr_scheduler.step(log_dict['loss_total'])  # 更新学习

        '''------------------模型验证---------------------'''
        predict_handler = PredictHandler(model, device, anc_obj.ancs,
                                         threshold_conf=cfg.THRESHOLD_PREDICT_CONF,
                                         threshold_nms=cfg.THRESHOLD_PREDICT_NMS)
        if model.cfg.IS_FMAP_EVAL:
            flog.info('FMAP 验证开始 %s', epoch + 1)
            res_eval = []
            f_evaluate4fmap(
                data_loader=loader_val_fmap,
                predict_handler=predict_handler,
                epoch=epoch,
                res_eval=res_eval)
            path_dt_info = loader_val_fmap.dataset.path_dt_info
            path_gt_info = loader_val_fmap.dataset.path_gt_info
            path_imgs = loader_val_fmap.dataset.path_imgs
            f_do_fmap(path_gt=path_gt_info, path_dt=path_dt_info, path_img=path_imgs,
                      confidence=cfg.THRESHOLD_PREDICT_CONF,
                      iou_map=[], ignore_classes=[], console_pinter=True,
                      plot_res=False, animation=False)
            return

        if model.cfg.IS_COCO_EVAL:
            flog.info('COCO 验证开始 %s', epoch + 1)
            # with torch.no_grad():
            mode = 'bbox'
            res_eval = []
            f_evaluate4coco(
                data_loader=loader_val_coco,
                predict_handler=predict_handler,
                epoch=epoch,
                tb_writer=tb_writer,
                res_eval=res_eval,
                mode=mode,
                device=device,
            )
