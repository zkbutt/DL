#!/usr/bin/env Python
# coding=utf-8
import sys
import time
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.data import DataLoader

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_pretreatment import Compose, Resize
from f_tools.datas.f_coco.coco_eval import CocoEvaluator
from f_tools.datas.f_coco.convert_data.coco_dataset import CocoDataset
from f_tools.datas.f_coco.convert_data.dataset2coco_obj import voc2coco_obj
from f_tools.f_torch_tools import save_weight, load_weight
from f_tools.fun_od.f_anc import AnchorsFound
from object_detection.f_retinaface.CONFIG_F_RETINAFACE import CFG
from object_detection.f_retinaface.nets.retinaface_training import MultiBoxLoss, DataGenerator, detection_collate
from object_detection.f_retinaface.train_fun import fit_one_epoch, evaluate
from object_detection.f_retinaface.utils.process_fun import init_model, data_loader


def _collate_fn(batch_datas):
    _t = batch_datas[0][0]
    # images = torch.empty((len(batch_datas), *_t.shape), device=_t.device)
    images = torch.empty((len(batch_datas), *_t.shape)).to(_t)
    targets = []
    for i, (img, taget) in enumerate(batch_datas):
        images[i] = img
        targets.append(taget)
    return images, targets


if __name__ == "__main__":
    '''
    04:54
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    flog.info('模型当前设备 %s', device)

    if CFG.DEBUG:
        CFG.DATA_NUM_WORKERS = 0

    training_dataset_path = r'M:\AI\datas\widerface\train/label.txt'
    # training_dataset_path = r'/home/bak3t/bak299g/AI/datas/widerface/train/label.txt'
    torch.multiprocessing.set_sharing_strategy('file_system')  # 多进程开文件

    model = init_model(CFG)
    model.to(device)  # 这个不需要加等号

    # start_epoch = 0
    # file_fit_weight = "model_data/Retinaface_mobilenet0.25.pth"
    # # 权重初始模式
    # state_dict = torch.load(file_fit_weight, map_location=device)
    # model_dict = model.state_dict()
    # keys_missing, keys_unexpected = model.load_state_dict(state_dict)
    # flog.info('加载成功 %s', file_fit_weight)
    # if len(keys_missing) > 0 or len(keys_unexpected):
    #     flog.error('missing_keys %s', keys_missing)
    #     flog.error('unexpected_keys %s', keys_unexpected)

    # if start_epoch < 25:
    #     for param in model.body.parameters():
    #         param.requires_grad = False
    # else:
    #     for param in model.body.parameters():
    #         param.requires_grad = True

    if CFG.IS_TRAIN:
        model.train()
        lr = 1e-3  # lr = 1e-4
        optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
        start_epoch = 0
        # start_epoch = load_weight(CFG.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device)

        anchors = AnchorsFound(CFG.IMAGE_SIZE, CFG.ANCHORS_SIZE, CFG.FEATURE_MAP_STEPS, CFG.ANCHORS_CLIP).get_anchors()
        anchors = anchors.to(device)

        box_loss = MultiBoxLoss(CFG.NUM_CLASSES, 0.35, 7, device)

        # loader_train, loader_val = data_loader(CFG, device)
        train_dataset = DataGenerator(training_dataset_path, CFG.IMAGE_SIZE, CFG.DEBUG)
        loader_train = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, pin_memory=True,
                                  num_workers=CFG.DATA_NUM_WORKERS,
                                  # drop_last=True,
                                  collate_fn=detection_collate)

        for epoch in range(start_epoch, CFG.END_EPOCH):
            loss = fit_one_epoch(model, box_loss, epoch, len(loader_train), loader_train, CFG.END_EPOCH, anchors, CFG,
                                 device, optimizer, lr_scheduler)
            lr_scheduler.step(loss)
    if CFG.IS_EVAL:
        model.eval()
        start_epoch = load_weight(CFG.FILE_FIT_WEIGHT, model, None, None, device)

        data_transform = Compose([
            # ResizeKeep(cfg.IMAGE_SIZE),  # (h,w)
            Resize(CFG.IMAGE_SIZE),
        ])
        # dataset_val = CocoDataset(CFG.PATH_DATA_ROOT, 'bboxs', 'val2017',
        #                           device, data_transform,
        #                           is_debug=CFG.DEBUG)
        #
        # loader_val = torch.utils.data.DataLoader(
        #     dataset_val,
        #     batch_size=CFG.BATCH_SIZE,
        #     num_workers=0,
        #     # shuffle=True,
        #     pin_memory=True,  # 不使用虚拟内存 GPU要报错
        #     drop_last=False,  # 除于batch_size余下的数据
        #     collate_fn=_collate_fn,
        # )
        train_dataset = DataGenerator(training_dataset_path, CFG.IMAGE_SIZE, CFG.DEBUG)
        loader_val = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, pin_memory=True,
                                num_workers=CFG.DATA_NUM_WORKERS,
                                drop_last=True, collate_fn=detection_collate)

        evaluate(model, loader_val, device)
