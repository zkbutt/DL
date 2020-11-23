import torch
from torch import optim

from f_pytorch.backbone_t.f_model_api import Output4Return
from f_pytorch.backbone_t.f_models.darknet import Darknet
from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_factory import VOCDataSet
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_fun_lr import f_lr_cos
from f_tools.fits.f_lossfun import LossYOLOv3
from f_tools.fun_od.f_anc import FAnchors
from f_tools.pic.enhance.data_pretreatment import Compose, ToTensor, Normalization4TS, ResizeKeep, \
    RandomHorizontalFlip4TS, ColorJitter

from f_tools.fits.f_show_fit_res import plot_loss_and_lr
from f_tools.fits.fitting.f_fit_eval_base import f_evaluate, f_train_one_epoch2
from f_tools.fun_od.f_boxes import nms
from object_detection.f_yolov3.CONFIG_YOLO3 import CFG
from object_detection.f_yolov3.nets.model_yolo3 import YoloV3SPP
from object_detection.f_yolov3.train_eval_fun import LossHandler

DATA_TRANSFORM = {
    "train": Compose([
        ResizeKeep(CFG.IMAGE_SIZE),
        # Resize(CFG.IMAGE_SIZE),
        # SSDCroppingPIL(),
        # ColorJitter(),
        ToTensor(),
        # RandomHorizontalFlip4TS(1),
        # Normalization4TS(),
    ], CFG),
    "val": Compose([
        ResizeKeep(CFG.IMAGE_SIZE),  # (h,w)
        # Resize(CFG.IMAGE_SIZE),
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


def init_model(cfg, device, id_gpu=None):
    model = Darknet(nums_layer=(1, 2, 8, 8, 4))
    return_layers = {'block3': 1, 'block4': 2, 'block5': 3}
    model = Output4Return(model, return_layers)
    dims_out = [256, 512, 1024]

    model = YoloV3SPP(model, cfg.NUMS_ANC, cfg.NUM_CLASSES, dims_out, is_spp=True)
    if id_gpu is not None:
        is_mgpu = True
        if CFG.SYSNC_BN:
            # 不冻结权重的情况下可, 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        else:
            model.to(device)  # 这个不需要加等号
        # 转为DDP模型
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[id_gpu])
    else:
        model.to(device)  # 这个不需要加等号
        is_mgpu = False
    model.train()

    cfg.SAVE_FILE_NAME = cfg.SAVE_FILE_NAME + 'darknet53'
    # ------------------------自定义backbone完成-------------------------------
    # f_look(model)

    anc_obj = FAnchors(CFG.IMAGE_SIZE, CFG.ANC_SCALE, CFG.FEATURE_MAP_STEPS, CFG.ANCHORS_CLIP, device=device)
    losser = LossYOLOv3(anc_obj, CFG)

    pg = model.parameters()
    # 最初学习率
    lr0 = 1e-3
    lrf = lr0 / 100
    optimizer = optim.SGD(pg, lr=lr0, momentum=0.937, weight_decay=0.0005, nesterov=True)
    start_epoch = load_weight(CFG.FILE_FIT_WEIGHT, model, optimizer, None, device, is_mgpu=is_mgpu)
    lr_scheduler = f_lr_cos(optimizer, start_epoch, CFG.END_EPOCH, lrf_scale=0.01)

    return model, losser, optimizer, lr_scheduler, start_epoch, anc_obj


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


def data_loader(cfg, is_mgpu=False):
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
    return loader_train, loader_val, train_sampler


def train_eval(cfg, start_epoch, model, anc_obj, losser, optimizer, lr_scheduler=None,
               loader_train=None, loader_val=None, device=torch.device('cpu'),
               train_sampler=None,
               ):
    train_loss = []
    learning_rate = []

    for epoch in range(start_epoch, cfg.END_EPOCH):

        if cfg.IS_TRAIN:
            process = LossHandler(model, device, anc_obj, losser, cfg)

            flog.info('训练开始 %s', epoch + 1)
            loss = f_train_one_epoch2(model=model,
                                      data_loader=loader_train, loss_process=process, optimizer=optimizer,
                                      epoch=epoch, cfg=cfg,
                                      lr_scheduler=None,
                                      ret_train_loss=train_loss, ret_train_lr=learning_rate,
                                      train_sampler=train_sampler,
                                      )

            if lr_scheduler is not None:
                lr_scheduler.step()  # 更新学习

        '''------------------模型验证---------------------'''
        if cfg.IS_EVAL:
            flog.info('验证开始 %s', epoch + 1)
            predict_handler = PredictHandler(model, device, anc_obj,
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
