import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.f_coco.convert_data.coco_dataset import CocoDataset
from f_tools.f_torch_tools import save_weight

from f_tools.fits.f_show_fit_res import plot_loss_and_lr
from f_tools.fits.fitting.f_fit_eval_base import f_train_one_epoch, f_evaluate
from f_tools.fun_od.f_boxes import nms
from object_detection.f_retinaface.CONFIG_F_RETINAFACE import CFG
from object_detection.f_retinaface.nets.mobilenet025 import MobileNetV1
from object_detection.f_retinaface.nets.retinaface import RetinaFace
from f_tools.datas.data_pretreatment import Compose, Resize, ColorJitter, ToTensor, RandomHorizontalFlip4TS, \
    Normalization4TS

from object_detection.f_retinaface.utils.train_eval_fun import LossHandler, PredictHandler

'''
这个文件处理大流水
'''
# if CFG.IS_VISUAL:
#     # 遍历降维 self.anc([1, 16800, 4])
#     _t = p_boxes.clone()
#     _t[:, ::2] = _t[:, ::2] * CFG.IMAGE_SIZE[0]
#     _t[:, 1::2] = _t[:, 1::2] * CFG.IMAGE_SIZE[1]
#     _t = _t[:20, :]
#     show_od4ts(img_ts.squeeze(0), p_boxes, torch.ones(200))
DATA_TRANSFORM = {
    "train": Compose([
        # ResizeKeep(cfg.IMAGE_SIZE),  # (h,w)
        Resize(CFG.IMAGE_SIZE),
        ColorJitter(),
        ToTensor(),
        RandomHorizontalFlip4TS(1),
        Normalization4TS(),
    ]),
    "val": Compose([
        # ResizeKeep(cfg.IMAGE_SIZE),  # (h,w)
        Resize(CFG.IMAGE_SIZE),
        ToTensor(),
        Normalization4TS(),
    ])
}




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

    if cfg.IS_TRAIN:
        dataset_train = CocoDataset(cfg.PATH_DATA_ROOT, 'keypoints', 'train2017',
                                    device, DATA_TRANSFORM['train'],
                                    is_debug=cfg.DEBUG)

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
        dataset_val = CocoDataset(cfg.PATH_DATA_ROOT, 'bboxs', 'val2017',
                                  device, DATA_TRANSFORM['val'],
                                  is_debug=cfg.DEBUG)

        loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.DATA_NUM_WORKERS,
            # shuffle=True,
            pin_memory=True,  # 不使用虚拟内存 GPU要报错
            # drop_last=True,  # 除于batch_size余下的数据
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
        if cfg.IS_EVAL:
            flog.info('验证开始 %s', epoch + 1)
            predict_handler = PredictHandler(model, device, )

            coco_res_bboxs = []
            # del coco_res_bboxs
            f_evaluate(
                model=model,
                data_loader=loader_val,
                anchors=anchors,
                device=device,
                epoch=epoch,
                coco_res_bboxs=coco_res_bboxs
            )
    if cfg.IS_TRAIN:
        '''-------------结果可视化-----------------'''
        if len(train_loss) != 0 and len(learning_rate) != 0:
            plot_loss_and_lr(train_loss, learning_rate)
