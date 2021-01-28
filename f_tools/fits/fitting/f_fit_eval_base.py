import json
import math
import os
import sys
import tempfile
import time
import datetime

import cv2
import torch
from collections import defaultdict, deque

from PIL import Image
from pycocotools.cocoeval import COCOeval
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.f_map.convert_data.extra.intersect_gt_and_dr import f_fix_txt, f_recover_gt
from f_tools.datas.f_map.map_go import f_do_fmap
from f_tools.f_torch_tools import save_weight
import torch.distributed as dist

from f_tools.fits.f_gpu.f_gpu_api import dict_all_gather, get_rank, fis_mgpu, is_main_process
from f_tools.fun_od.f_boxes import ltrb2ltwh, xywh2ltrb
from f_tools.pic.f_show import f_show_od4ts, f_plot_od4pil, f_show_od4pil, f_plot_od4pil_keypoints, f_plt_od_pil, \
    f_plt_show_pil, show_pic_ts, f_plt_od_np, f_plt_show_cv


def calc_average_loss(value, log_dict):
    '''
    计算出多 GPU 的平均loss 直接修改 log_dict
    :param value:
    :param log_dict:
    :return:
    '''
    if fis_mgpu():
        num_gpu = torch.distributed.get_world_size()  # 总GPU个数
    else:
        return log_dict['l_total']

    with torch.no_grad():  # 多GPU重算loss
        torch.distributed.all_reduce(value)  # 所有设备求和
        value /= num_gpu
        value = value.detach().item()
        _t = log_dict['l_total']  # 这个是GPU的损失值
        log_dict['l_total'] = value  # 变成平均值
        # log_dict['lgpu'] = _t
        return value


def f_train_one_epoch4(model, data_loader, optimizer, epoch,
                       fun_datas_l2=None,
                       lr_scheduler=None,
                       tb_writer=None,
                       train_sampler=None,
                       device=None,
                       ):
    cfg = model.cfg
    end_epoch = cfg.END_EPOCH
    print_freq = cfg.PRINT_FREQ
    forward_count = cfg.FORWARD_COUNT

    if train_sampler is not None:
        train_sampler.set_epoch(epoch)  # 使每一轮多gpu获取的数据不一样

    if fis_mgpu():
        # flog.debug('get_rank %s 这里等待', get_rank())
        torch.distributed.barrier()  # 等待GPU

    metric_logger = MetricLogger(is_show_log=True, delimiter=" ")  # 日志记录器
    # if is_main_process():  # 多gpu主进程 这里要报错
    #     metric_logger = MetricLogger(is_show_log=True, delimiter=" ")  # 日志记录器
    # else:
    #     metric_logger = MetricLogger(is_show_log=False, delimiter=" ")

    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch:[{}/{}]'.format(epoch + 1, end_epoch)

    # ---半精度训练1---
    # enable_amp = True if "cuda" in device.type else False
    scaler = GradScaler(enabled=cfg.IS_MIXTURE_FIX)
    for i, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header, epoch)):
        # lr 初始化 LR变更时使用
        if epoch < cfg.EPOCH_WARMUP:  # warmup 热身训练
            tmp_lr = cfg.LR0 * pow((i + epoch * len(data_loader)) * 1. / (cfg.EPOCH_WARMUP * len(data_loader)), 4)
            # tmp_lr = 1e-6 + (cfg.LR0 - 1e-6) * (i + epoch * len(data_loader)) / (cfg.EPOCH_WARMUP * len(data_loader))
            for param_group in optimizer.param_groups:
                param_group['lr'] = tmp_lr

        elif epoch == cfg.EPOCH_WARMUP:  # 相等时强制恢复一次
            tmp_lr = cfg.LR0
            for param_group in optimizer.param_groups:
                param_group['lr'] = tmp_lr

        # ---半精度训练2---
        with autocast(enabled=cfg.IS_MIXTURE_FIX):
            #  完成  数据组装完成   模型输入输出    构建展示字典及返回值
            if fun_datas_l2 is not None:
                img_ts4, g_targets = fun_datas_l2(batch_data, device, cfg, epoch, model)

            loss_total, log_dict = model(img_ts4, g_targets)

            if not math.isfinite(loss_total):  # 当计算的损失为无穷大时停止训练
                flog.critical("Loss is {}, stopping training".format(loss_total))
                flog.critical(log_dict)
                sys.exit(1)

            _ = calc_average_loss(loss_total, log_dict)  # 更新log_dict 各显卡损失平均
            loss_total *= 1. / forward_count  # 分解批次 整合一起
            # ---半精度训练2  完成---

        # ---半精度训练3---
        # torch.autograd.set_detect_anomaly(True)
        scaler.scale(loss_total).backward()

        '''多GPU输出均值 对 loss_total 进行修正'l_total'  'loss_g' 直接修改 log_dict'''

        # 每训练n批图片更新一次权重
        if i % forward_count == 0:
            scaler.step(optimizer)
            scaler.update()  # 查看是否要更新scaler
            optimizer.zero_grad()

        # 这里记录日志输出 直接用字典输入
        metric_logger.update(**log_dict)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

        if tb_writer is not None and not cfg.LOSS_EPOCH and is_main_process():
            # 主进程写入
            for k, v, in log_dict.items():
                tb_writer.add_scalar('loss_iter/%s' % k, v, len(data_loader) * epoch + i + 1)

    # 这里返回的是该设备的平均loss ,不是所有GPU的
    log_dict_avg = {}
    log_dict_avg['lr'] = metric_logger.meters['lr'].value
    for k, v in log_dict.items():
        log_dict_avg[k] = metric_logger.meters[k].avg

    if lr_scheduler is not None:
        # 每批的LR更新
        # flog.warning('更新 lr_scheduler loss:%s', log_dict['l_total'])
        # lr_scheduler.step(log_dict['l_total'])  # 更新学习
        lr_scheduler.step(epoch)  # 更新学习

    if fis_mgpu() and not is_main_process():
        # 只有0进程才需要保存
        return log_dict_avg

    # if epoch in cfg.lr_scheduler:
    #     cfg.LR0 = cfg.LR0 * 0.1
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = cfg.LR0

    if tb_writer is not None and cfg.LOSS_EPOCH:
        # 每一epoch
        # flog.warning('写入 tb_writer %s', log_dict_avg)
        # now_lr = optimizer.param_groups[0]["lr"]
        # tb_writer.add_scalar('Train/lr', now_lr, epoch + 1)
        for k, v, in log_dict_avg.items():
            tb_writer.add_scalar('loss_epoch/%s' % k, v, epoch + 1)

    # epoch保存
    if ((epoch + 1) % cfg.NUM_SAVE_INTERVAL) == 0 or cfg.IS_FORCE_SAVE:
        flog.info('训练完成正在保存模型...')
        save_weight(
            path_save=cfg.PATH_SAVE_WEIGHT,
            model=model,
            name=cfg.SAVE_FILE_NAME,
            loss=log_dict_avg['l_total'],
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch)

        if cfg.IS_FORCE_SAVE:
            flog.info('IS_FORCE_SAVE 完成')
            sys.exit(-1)

    return log_dict_avg


@torch.no_grad()
def f_evaluate4coco3(model, data_loader, epoch, fun_datas_l2=None,
                     res_eval=None, tb_writer=None, ann_type='bbox',
                     device=None, eval_sampler=None, is_keep=False, ):
    '''

    :param ann_type:  ['segm', 'bbox', 'keypoints']
    '''
    if eval_sampler is not None:
        eval_sampler.set_epoch(epoch)  # 使每一轮多gpu获取的数据不一样

    cfg = model.cfg
    res_ = {}
    ids_coco = []
    num_no_pos = 0

    if fis_mgpu():
        # flog.debug('get_rank %s 这里等待', get_rank())
        torch.distributed.barrier()

    pbar = tqdm(data_loader, desc='%s' % epoch, postfix=dict, mininterval=0.1)
    for batch_data in pbar:
        # torch.Size([5, 3, 416, 416])
        img_ts4, g_targets = fun_datas_l2(batch_data, device, cfg, epoch, model)
        # 处理size 和 ids 用于coco
        images, targets = batch_data

        '''提取真实ID及尺寸'''
        _sizes = []  # 用于修复box
        _ids = []
        for target in targets:  # 通过target 提取ID 和 size
            ids_coco.append(target['image_id'])  # 加入总ID
            _ids.append(target['image_id'])
            _s = target['size']
            if is_keep:  # keep修复
                max1 = max(_s)
                _s = [max1, max1]
            if isinstance(_s, torch.Tensor):
                _sizes.append(_s.clone().detach())  # tnesor
            else:
                _sizes.append(torch.tensor(_s))  # tnesor

            # if cfg.IS_VISUAL:
            #     coco_gt = data_loader.dataset.coco
            #     f_show_coco_pics(coco_gt, data_loader.dataset.path_img, ids_img=[target['image_id']])

        ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores = model(img_ts4, g_targets)
        if p_labels is None or len(p_labels) == 0:
            num_no_pos += len(data_loader)
            flog.info('本批没有目标 num_no_pos 3次后出 当前: %s', num_no_pos)
            # if num_no_pos > 3:  # 如果3个批都没有目标则放弃
            #     return
            # else:  # 没有目标就下一个
            #     num_no_pos += 1
            pbar.set_description("未-%s" % num_no_pos)
            continue

        _res_t = {}  # 每一批的结果
        # 每一张图的 id 与批次顺序保持一致 选出匹配
        for i, (size, image_id) in enumerate(zip(_sizes, _ids)):
            mask = ids_batch == i  # 构建 batch 次的mask
            if torch.any(mask):  # 如果最终有目标存在 将写出info中
                if cfg.IS_VISUAL:
                    img_ts = img_ts4[i]
                    # flog.debug('nms后 预测共有多少个目标: %s' % p_boxes_ltrb[mask].shape[0])
                    # from f_tools.pic.enhance.f_data_pretreatment import f_recover_normalization4ts
                    # img_ts = f_recover_normalization4ts(img_ts)
                    # from torchvision.transforms import functional as transformsF
                    # img_pil = transformsF.to_pil_image(img_ts).convert('RGB')
                    # # 处理完后尺寸
                    # _size = torch.tensor(cfg.IMAGE_SIZE * 2)
                    # p_boxes_ltrb_f = p_boxes_ltrb[mask].cpu() * _size
                    # f_plt_od(img_pil, p_boxes_ltrb_f,
                    #          g_boxes_ltrb=targets[i]['boxes'].cpu(),  # gbox 默认不归一化
                    #          ids2classes=data_loader.dataset.ids_classes,
                    #          labels=p_labels[mask],
                    #          scores=p_scores[mask].tolist(),
                    #          is_recover_size=False
                    #          )

                    _size = torch.tensor(cfg.IMAGE_SIZE * 2)
                    coco = data_loader.dataset.coco
                    img_info = coco.loadImgs([image_id])
                    file_img = os.path.join(data_loader.dataset.path_img, img_info[0]['file_name'])

                    img_np = cv2.imread(file_img)
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                    # import skimage.io as io
                    # h,w,c
                    # img_np = io.imread(file_img)
                    whwh = np.tile(np.array(img_np.shape[:2][::-1]), 2)

                    p_boxes_ltrb_f = p_boxes_ltrb[mask].cpu() * whwh
                    f_plt_od_np(img_np, p_boxes_ltrb_f,
                                g_boxes_ltrb=targets[i]['boxes'].cpu() / _size * whwh,  # gbox 默认不归一化
                                ids2classes=data_loader.dataset.ids_classes,
                                labels=p_labels[mask],
                                scores=p_scores[mask].tolist(),
                                is_recover_size=False
                                )

                # 恢复真实尺寸(原装未处理) coco需要 ltwh
                boxes_ltwh = ltrb2ltwh(p_boxes_ltrb[mask] * size.repeat(2)[None])
                _res_t[image_id] = {
                    'boxes': boxes_ltwh,
                    'labels': p_labels[mask],
                    'scores': p_scores[mask],
                }
                # 更新进度条值
                d = {
                    'pos': len(boxes_ltwh),
                    'max': round(p_scores[mask].max().item() * 100, 1),
                    # 'min': round(p_scores.min().item(), 1),
                    'mean': round(p_scores[mask].mean().item() * 100, 1),
                }
                pbar.set_postfix(**d)

            else:
                # flog.warning('没有预测出框 %s', files_txt)
                num_no_pos += 1
                pbar.set_description("未-%s" % num_no_pos)

        if len(_res_t) > 0:
            res_.update(_res_t)  # 扩展单个

    # if len(res_)
    # 重组数据
    res_coco = []

    '''这里处理多 GPU 数据同步 '''
    if fis_mgpu():
        # 这里未考虑 res_ 为空的优化
        d = {}
        d['res_'] = res_
        d['ids_coco'] = ids_coco
        d['num_no_pos'] = num_no_pos
        data_list = dict_all_gather(d)  # 任务类型同步
        if not is_main_process():
            # 其它 GPU 进程退出
            flog.debug('get_rank %s 已退出', get_rank())
            return None
        res_.clear()  # 重组多GPU的数据
        ids_coco.clear()
        num_no_pos = 0
        for d in data_list:
            res_.update(d['res_'])
            ids_coco.extend(d['ids_coco'])
            num_no_pos += d['num_no_pos']

    for i, (image_id, target) in enumerate(res_.items()):
        labels = target['labels'].type(torch.int).tolist()
        boxes_ltwh = target['boxes'].tolist()
        score = target['scores'].tolist()
        for i in range(len(labels)):
            res_coco.append({"image_id": image_id, "category_id": labels[i], "bbox": boxes_ltwh[i], "score": score[i]})

    maps_val = []
    if len(res_coco) > 0:
        coco_gt = data_loader.dataset.coco
        # 第一个元素指示操作该临时文件的安全级别，第二个元素指示该临时文件的路径
        _, tmp = tempfile.mkstemp()
        json.dump(res_coco, open(tmp, 'w'))
        cocoDt = coco_gt.loadRes(tmp)
        coco_eval_obj = COCOeval(coco_gt, cocoDt, ann_type)
        coco_eval_obj.params.imgIds = ids_coco
        coco_eval_obj.evaluate()
        coco_eval_obj.accumulate()
        coco_eval_obj.summarize()
        maps_val.append(coco_eval_obj.stats[1])
        maps_val.append(coco_eval_obj.stats[7])

        if tb_writer is not None:
            titles = [
                '[IoU=0.50:0.95]', '[IoU=0.50]', '[IoU=0.75]',
                'small', 'medium', 'large',
            ]
            tb_writer.add_scalar('mAP/num_no_pos', num_no_pos, epoch + 1)  # 未检出的图片数
            for i, title in zip(range(6), titles):
                _d = {
                    'Precision': coco_eval_obj.stats[i],
                    'Recall': coco_eval_obj.stats[i + 6],
                }
                tb_writer.add_scalars('mAP/%s' % title, _d, epoch + 1)
    else:
        if tb_writer is not None:
            tb_writer.add_scalar('mAP/num_no_pos', num_no_pos, epoch + 1)  # 未检出的图片数
        maps_val = [0, 0]
    return maps_val


# def _polt_boxes(img_pil, p_boxes_ltrb, szie_scale4bbox, p_scores, p_labels, labels_lsit):
#     if p_boxes_ltrb is not None:
#         flog.debug('一共有 %s 个目标', p_boxes_ltrb.shape[0])
#         p_boxes = p_boxes_ltrb * szie_scale4bbox
#         img_pil = f_plot_od4pil(img_pil, p_boxes, p_scores, p_labels, labels_lsit)
#     return img_pil


def _polt_keypoints(img_pil, p_boxes_ltrb, szie_scale4bbox, p_keypoints, szie_scale4landmarks,
                    p_scores, p_labels, labels_lsit):
    if p_boxes_ltrb is not None:
        flog.debug('一共有 %s 个目标', p_boxes_ltrb.shape[0])
        p_boxes = p_boxes_ltrb * szie_scale4bbox
        p_keypoints = p_keypoints * szie_scale4landmarks
        img_pil = f_plot_od4pil_keypoints(img_pil, p_boxes, p_keypoints, p_scores, p_labels, labels_lsit)
    return img_pil


def f_prod_pic4one(img_np, data_transform, model, size_ts, labels_lsit, is_keeep=False, cfg=None,
                   gboxes_ltrb=None, target=None):
    img_ts_one, boxes, labels = data_transform(img_np)
    img_ts4 = img_ts_one.unsqueeze_(0)
    size_input = size_ts
    if is_keeep:
        max1 = max(size_ts)
        size_input = torch.tensor([max1, max1])

    # 用于恢复bbox及ke
    szie_scale4bbox = size_input.repeat(2)

    '''---------------预测开始--------------'''
    ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores = model(img_ts4)

    if p_labels is None or len(p_labels) == 0:
        _text = '未检测出来，id为：%s'
        if target is not None:
            _text = _text % target['image_id']
        flog.warning(_text)
        plabels_text = None
        p_scores_float = None
        p_boxes_ltrb = p_boxes_ltrb
    else:
        plabels_text = []
        p_scores_float = []
        for i, label in enumerate(p_labels):
            plabels_text.append(labels_lsit[int(label.item())])
            p_scores_float.append(p_scores[i].item())
        p_boxes_ltrb = p_boxes_ltrb * szie_scale4bbox

    f_plt_show_cv(img_np, gboxes_ltrb=gboxes_ltrb,
                  pboxes_ltrb=p_boxes_ltrb,
                  plabels_text=plabels_text,
                  p_scores_float=p_scores_float,
                  )


def f_prod_pic4file(file_img, model, labels_lsit, data_transform, is_keeep=False, cfg=None):
    img_pil = Image.open(file_img).convert('RGB')
    f_prod_pic4one(img_pil, model, labels_lsit, data_transform, is_keeep=is_keeep, cfg=cfg)


def f_prod_pic4keypoints(file_img, model, labels_lsit, data_transform, is_keeep=False, cfg=None):
    img_pil = Image.open(file_img).convert('RGB')
    size_input = img_pil.size
    if is_keeep:
        max1 = max(size_input)
        size_input = [max1, max1]

    # 用于恢复bbox及ke
    szie_scale4bbox = torch.Tensor(size_input * 2)
    szie_scale4landmarks = torch.Tensor(size_input * 5)
    img_ts = data_transform['val'](img_pil)[0][None]

    '''---------------预测开始--------------'''
    ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores = model(img_ts)
    img_pil = _polt_keypoints(img_pil, p_boxes_ltrb, szie_scale4bbox, p_keypoints, szie_scale4landmarks,
                              p_scores, p_labels, labels_lsit)
    img_pil.show()


def f_prod_vodeo(cap, data_transform, model, labels_lsit, is_keeep=False):
    fps = 0.0
    count = 0

    while True:
        start_time = time.time()
        '''---------------数据加载及处理--------------'''
        ref, img_np = cap.read()  # 读取某一帧 ref是否成功
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # 格式转变，BGRtoRGB
        img_pil = Image.fromarray(img_np, mode="RGB")
        size_input = img_pil.size
        if is_keeep:
            max1 = max(size_input)
            size_input = [max1, max1]

        szie_scale4bbox = torch.Tensor(size_input * 2)
        szie_scale4landmarks = torch.Tensor(size_input * 5)
        img_ts = data_transform['val'](img_pil)[0][None]

        '''---------------预测开始--------------'''
        ids_batch, p_boxes_ltrb, p_labels, p_scores = model(img_ts)

        img_pil = _polt_boxes(img_pil, p_boxes_ltrb, szie_scale4bbox, p_scores, p_labels, labels_lsit)
        img_np = np.array(img_pil)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        # print("fps= %.2f" % (fps))
        count += 1
        img_np = cv2.putText(img_np, "fps= %.2f count=%s" % (fps, count), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                             (0, 255, 0), 2)
        # 极小数
        fps = (fps + (1. / max(sys.float_info.min, time.time() - start_time))) / 2
        cv2.imshow("video", img_np)

        c = cv2.waitKey(1) & 0xff  # 输入esc退出
        if c == 27:
            cap.release()
            break


def f_prod_vodeo4keypoints(cap, data_transform, model, labels_lsit, is_keeep=False):
    fps = 0.0
    count = 0

    while True:
        start_time = time.time()
        '''---------------数据加载及处理--------------'''
        ref, img_np = cap.read()  # 读取某一帧 ref是否成功
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # 格式转变，BGRtoRGB
        img_pil = Image.fromarray(img_np, mode="RGB")
        size_input = img_pil.size
        if is_keeep:
            max1 = max(size_input)
            size_input = [max1, max1]

        szie_scale4bbox = torch.Tensor(size_input * 2)
        szie_scale4landmarks = torch.Tensor(size_input * 5)
        img_ts = data_transform['val'](img_pil)[0][None]

        '''---------------预测开始--------------'''
        ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores = model(img_ts)
        img_pil = _polt_keypoints(img_pil, p_boxes_ltrb, szie_scale4bbox, p_keypoints, szie_scale4landmarks,
                                  p_scores, p_labels, labels_lsit)

        img_np = np.array(img_pil)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        # print("fps= %.2f" % (fps))
        count += 1
        img_np = cv2.putText(img_np, "fps= %.2f count=%s" % (fps, count), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                             (0, 255, 0), 2)
        # 极小数
        fps = (fps + (1. / max(sys.float_info.min, time.time() - start_time))) / 2
        cv2.imshow("video", img_np)

        c = cv2.waitKey(1) & 0xff  # 输入esc退出
        if c == 27:
            cap.release()
            break


def handler_fmap(self, batch_data, path_dt_info, idx_to_class):
    '''
    生成 dt_info
        tvmonitor 0.471781 0 13 174 244
        cup 0.414941 274 226 301 265
    :param batch_data:
    :param path_dt_info:
    :param idx_to_class:
    :return:
    '''
    # (batch,3,640,640)   list(batch{'size','boxes','labels'}) 转换到GPU设备
    images, targets = _preprocessing_data(batch_data, self.device, mode='bbox')
    sizes = []
    files_txt = []
    for target in targets:
        sizes.append(target['size'])
        files_txt.append(os.path.join(path_dt_info, target['name_txt']))

    idxs, p_boxes, p_labels, p_scores = self.predicting4many(images)

    for batch_index, (szie, file_txt) in enumerate(zip(sizes, files_txt)):
        mask = idxs == batch_index  # 构建 batch 次的mask
        if torch.any(mask):  # 如果最终有目标存在 将写出info中
            lines_write = []
            for label, score, bbox in zip(p_labels[mask], p_scores[mask], p_boxes[mask]):
                _bbox = [str(i.item()) for i in list((bbox * szie.repeat(2)).type(torch.int64).data)]
                bbox_str = ' '.join(_bbox)
                _line = idx_to_class[label.item()] + ' ' + str(score.item()) + ' ' + bbox_str + '\n'
                lines_write.append(_line)
            with open(file_txt, "w") as f:
                f.writelines(lines_write)
        else:
            # flog.warning('没有预测出框 %s', files_txt)
            pass
    return p_labels, p_scores, p_boxes, sizes, idxs


def f_evaluate4fmap(model, data_loader, is_keeep, cfg):
    '''
    1 通过 voc2fmap.py 创建 gtinfo
    2 设置 IS_FMAP_EVAL 开启 指定 PATH_EVAL_IMGS 和 PATH_EVAL_INFO 文件夹
    3 加载 classes_ids_voc
    4 支持 degbug 设置
    5 生成 dt_info label conf ltrb真实值
            tvmonitor 0.471781 0 13 174 244
            cup 0.414941 274 226 301 265
    6 f_fix_txt(gt_path, DR_PATH)  使gt 与 dt 一致
    7 f_recover_gt(gt_path)  # 恢复 gt
    8 到 D:\tb\tb\ai_code\DL\f_tools\datas\f_map\output  查看结果 查打开 f_do_fmap plot_res
    9 恢复在 CONFIG中

    :param model:
    :param data_loader:
    :param is_keeep:
    :return:
    '''
    path_dt_info = data_loader.dataset.path_dt_info
    ids2classes = data_loader.dataset.ids2classes

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test: "
    print_freq = max(int(len(data_loader) / 5), 1)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        _dataset = data_loader.dataset
        img_ts4, targets = batch_data
        sizes = []
        files_txt = []
        for target in targets:
            size_ = target['size']
            if is_keeep:  # keep修复
                max1 = max(size_)
                size_ = [max1, max1]
            sizes.append(torch.tensor(size_))  # tesor
            # sizes.append(target['size'])
            files_txt.append(os.path.join(path_dt_info, target['name_txt']))

        ids_batch, p_boxes_ltrb, p_labels, p_scores = model(img_ts4)

        for i, (szie, file_txt) in enumerate(zip(sizes, files_txt)):
            mask = ids_batch == i  # 构建 batch 次的mask
            if torch.any(mask):  # 如果最终有目标存在 将写出info中
                lines_write = []
                for label, score, bbox in zip(p_labels[mask], p_scores[mask], p_boxes_ltrb[mask]):
                    _bbox = [str(i.item()) for i in list((bbox * szie.repeat(2)).type(torch.int64).data)]
                    bbox_str = ' '.join(_bbox)
                    _line = ids2classes[str(int(label.item()))] + ' ' + str(score.item()) + ' ' + bbox_str + '\n'
                    lines_write.append(_line)
                with open(file_txt, "w") as f:
                    f.writelines(lines_write)
            else:
                flog.warning('没有预测出框 %s', files_txt)
                pass

        end_time = time.time() - start_time
        metric_logger.update(eval_time=end_time)  # 这个填字典 添加的字段
    flog.info('dt_info 生成完成')

    # path_dt_info = data_loader.dataset.path_dt_info
    path_gt_info = data_loader.dataset.path_gt_info
    path_imgs = data_loader.dataset.path_imgs
    # f_recover_gt(path_gt_info)
    f_fix_txt(path_gt_info, path_dt_info)

    f_do_fmap(path_gt=path_gt_info, path_dt=path_dt_info, path_img=path_imgs,
              confidence=cfg.THRESHOLD_PREDICT_CONF,
              iou_map=[], ignore_classes=[], console_pinter=True,
              plot_res=False, animation=True)


class SmoothedValue(object):
    """
    记录一系列统计量
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)  # deque简单理解成加强版list
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not fis_mgpu():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):  # @property 是装饰器，这里可简单理解为增加median属性(只读)
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, is_show_log=True, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.is_show_log = is_show_log

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, epoch=1):
        if not self.is_show_log:
            # 多GPU主进程才显示日志
            return
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}/{2}]',
                                           'eta: {eta}',
                                           '{meters}',
                                           'time: {time}',
                                           'data: {data}',
                                           'mem: {memory:.0f}'])
        else:
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}]',
                                           'eta: {eta}',
                                           '{meters}',
                                           'time: {time}',
                                           'data: {data}',
                                           ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                # eta_second = iter_time.global_avg * (len(iterable) - i)
                eta_second = int(round(iter_time.global_avg * (len(iterable) - i), 0))
                eta_string = str(datetime.timedelta(seconds=eta_second))
                if torch.cuda.is_available():
                    # flog.debug(log_msg.format(i + 1, len(iterable),
                    print(log_msg.format(i + 1, len(iterable), len(iterable) * epoch + i + 1,
                                         eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time),  # 模型迭代时间(含数据加载) SmoothedValue对象
                                         data=str(data_time),  # 取数据时间 SmoothedValue对象
                                         # r_time=str(int(iter_time.value * (len(iterable) - i))),
                                         memory=torch.cuda.max_memory_allocated() / MB))  # 只能取第一个显卡
                else:
                    # flog.debug(log_msg.format(i, len(iterable),
                    print(log_msg.format(i, len(iterable),
                                         eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time),
                                         # r_time=str(int(iter_time.value * (len(iterable) - i))),
                                         data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # 每批时间
        # flog.debug('{} Total time: {} ({:.4f} s / it)'.format(header,
        print('{} Total time: {} ({:.4f} s / it)'.format(header,
                                                         total_time_str,
                                                         total_time / len(iterable)))


if __name__ == '__main__':
    pass
