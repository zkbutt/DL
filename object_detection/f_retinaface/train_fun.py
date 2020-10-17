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
from f_tools.fun_od.f_boxes import xywh2ltrb
from f_tools.pic.f_show import show_od_keypoints4ts, show_od_keypoints4np, show_od4np
from object_detection.f_retinaface.CONFIG_F_RETINAFACE import CFG
from object_detection.f_retinaface.nets.retinaface_training import MultiBoxLoss, DataGenerator, detection_collate
from object_detection.f_retinaface.utils.process_fun import init_model, data_loader
import numpy as np


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(model, box_loss, epoch, end_epoch, loader, Epoch, anchors, cfg, device, optimizer, lr_scheduler):
    total_r_loss = 0
    total_c_loss = 0
    total_landmark_loss = 0

    start_time = time.time()

    scaler = GradScaler()
    with tqdm(total=end_epoch, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict,
              mininterval=0.3,
              # dynamic_ncols=True,
              ncols=200,
              ) as pbar:
        # flog.debug('开始加载数据 %s', )
        for iteration, batch in enumerate(loader):
            if iteration >= end_epoch:
                break
            # imgs_ts:(batch, 3, 640, 640)
            # targets: list_batch个(x, 15)
            imgs_np, targets = batch

            if CFG.IS_VISUAL:
                for img, target in zip(imgs_np, targets):
                    # 遍历降维
                    # _t = anchors.view(-1, 4).copy()
                    _t = anchors.view(-1, 4).clone()
                    _t[:, ::2] = _t[:, ::2] * CFG.IMAGE_SIZE[0]
                    _t[:, 1::2] = _t[:, 1::2] * CFG.IMAGE_SIZE[1]
                    img_t = np.transpose(img, (1, 2, 0)).copy()
                    # show_od_keypoints4np(img_t, _t[:, :4], _t[:, 4:14], target[:, -1])
                    show_od4np(img_t, xywh2ltrb(_t), torch.ones(999))

            # _targets = []
            # for target in targets:
            #     try:
            #         _t = torch.cat([target['bboxs'], target['labels'][:, None], target['keypoints']], dim=1)
            #     except Exception as e:
            #         flog.error('%s %s %s %s', target['bboxs'].shape, target['labels'].shape, target['keypoints'].shape,
            #                    e)
            #         sys.exit()
            #     _targets.append(_t.to(device))
            # 这里不可能处理到一致
            with torch.no_grad():
                imgs_ts = torch.tensor(imgs_np).type(torch.float).to(device)
                targets = [torch.from_numpy(ann).type(torch.float).to(device) for ann in targets]

            with autocast():
                # forward
                # imgs_np = imgs_np.type(torch.cuda.HalfTensor)
                out = model(imgs_ts)
                r_loss, c_loss, landm_loss = box_loss(out, anchors, targets)
                loss = 2 * r_loss + c_loss + landm_loss

            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()  # # 查看是否要更新scaler
            # loss.backward()
            # optimizer.step()

            total_c_loss += c_loss.item()
            total_r_loss += r_loss.item()
            total_landmark_loss += landm_loss.item()
            waste_time = time.time() - start_time

            pbar.set_postfix(**{'Conf Loss': total_c_loss / (iteration + 1),
                                'Regression Loss': total_r_loss / (iteration + 1),
                                'LandMark Loss': total_landmark_loss / (iteration + 1),
                                'lr': get_lr(optimizer),
                                's/step': waste_time})
            pbar.update(1)
            start_time = time.time()

    if not cfg.DEBUG:
        save_weight(
            path_save=cfg.PATH_SAVE_WEIGHT,
            model=model,
            name=cfg.SAVE_FILE_NAME,
            loss=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch)
    return (total_c_loss + total_r_loss + total_landmark_loss) / (end_epoch + 1)


@torch.no_grad()
def evaluate(model, data_loader, device):
    '''将验证集改为coco_gt'''
    # coco_gt = voc2coco_obj(data_loader.dataset)
    # coco_evaluator = CocoEvaluator(data_loader.dataset.coco, ['bbox'])

    for epoch, datas in enumerate(data_loader):
        with tqdm(total=len(data_loader), desc=f'Epoch {epoch + 1}/{len(data_loader)}', postfix=dict,
                  mininterval=0.3,
                  # dynamic_ncols=True,
                  ncols=200,
                  ) as pbar:
            images, targets = datas[0], datas[1]
            images = torch.tensor(images).type(torch.float).to(device)
            targets = [torch.from_numpy(ann).type(torch.float).to(device) for ann in targets]
            out = model(images)
            # 多outputs数据组装 个结果切换到CPU
            outputs = []
            for index, (bboxes_out, labels_out, scores_out) in enumerate(results):
                info = {"boxes": bboxes_out.to(cpu_device),
                        "labels": labels_out.to(cpu_device),
                        "scores": scores_out.to(cpu_device),
                        "height_width": targets[index]["height_width"]}
                outputs.append(info)

            res = dict()
            for index in range(len(outputs)):
                info = {targets[index]["image_id"].item(): outputs[index]}
                res.update(info)

            coco_evaluator.update(res)
