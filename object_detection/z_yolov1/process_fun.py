import torch
from torch import optim
from torchvision import models

from f_pytorch.tools_model.f_layer_get import ModelOut4Mobilenet_v2
from f_pytorch.tools_model.model_look import f_look_model
from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_factory import MapDataSet, load_od4voc
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_gpu.f_gpu_api import model_device_init
from f_tools.fits.f_match import fmatch4yolo1
from f_tools.fits.fitting.f_fit_eval_base import f_train_one_epoch4, f_evaluate4coco2, is_mgpu
from f_tools.pic.enhance.f_data_pretreatment import Compose, ColorJitter, ToTensor, RandomHorizontalFlip4TS, \
    Normalization4TS, Resize
from object_detection.z_yolov1.nets.net_yolov1 import Yolo_v1


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
                # ResizeKeep(cfg.IMAGE_SIZE),  # 这个有边界需要修正
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
    batch = images.shape[0]

    # 匹配后最终结果 一个网格只匹配一个
    dim_out = cfg.NUM_BBOX * (4 + 1) + cfg.NUM_CLASSES
    p_yolos = torch.zeros((batch, cfg.NUM_GRID, cfg.NUM_GRID, dim_out), device=device)

    for i, target in enumerate(targets):
        # 这里是每一个图片
        boxes_one = target['boxes'].to(device)
        labels_one = target['labels'].to(device)  # ltrb
        p_yolo = fmatch4yolo1(boxes=boxes_one, labels=labels_one, num_bbox=cfg.NUM_BBOX,
                              num_class=cfg.NUM_CLASSES, grid=cfg.NUM_GRID, device=device)
        p_yolos[i] = p_yolo
        # target['size'] = target['size'].to(self.device)

    return images, p_yolos


def init_model(cfg, device, id_gpu=None):
    model = models.mobilenet_v2(pretrained=True)
    model = ModelOut4Mobilenet_v2(model)
    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'mobilenet_v2'

    model = Yolo_v1(backbone=model, dim_in=model.dim_out, grid=cfg.NUM_GRID,
                    num_classes=cfg.NUM_CLASSES, num_bbox=cfg.NUM_BBOX, cfg=cfg)
    # f_look_model(model, input=(1, 3, *cfg.IMAGE_SIZE))

    if cfg.IS_LOCK_BACKBONE_WEIGHT:
        for name, param in model.backbone.named_parameters():
            param.requires_grad_(False)
        # 除最后的全连接层外，其他权重全部冻结
        # if "fc" not in name:
        #     param.requires_grad_(False)

    model, is_mgpu = model_device_init(model, device, id_gpu, cfg)
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

    _res = load_od4voc(cfg, data_transform, is_mgpu)
    loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler = _res

    # 返回数据已预处理 返回np(batch,(3,640,640))  , np(batch,(x个选框,15维))
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
        #
