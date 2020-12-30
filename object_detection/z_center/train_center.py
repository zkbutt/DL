import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])

import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_fit_fun import init_od, base_set, custom_set, train_eval4od
from f_tools.fits.f_match import match4center
from f_tools.fun_od.f_boxes import ltrb2xywh
from object_detection.z_center.CONFIG_CENTER import CFG
from object_detection.z_center.process_fun import init_model, data_loader4voc, data_loader4widerface, \
    data_loader4raccoon200


def fdatas_l2(batch_data, device, cfg=None):
    '''

    :param batch_data:
        images : torch.Size([5, 3, 512, 512])
        targets :
            boxes : ltrb
            labels :
            'keypoints' : 关键点数量
    :param device:
    :param cfg:
    :return:
    '''
    images, targets = batch_data
    images = images.to(device)
    size = images.shape[-2:]
    fsize = torch.tensor(size, device=device).floor_divide(cfg.FEATURE_MAP_STEP)  # 整除
    batch = len(targets)

    # num_class + 中间给关键点 + xy偏移 + wh偏移
    if cfg.NUM_KEYPOINTS > 0:
        dim = cfg.NUM_CLASSES + cfg.NUM_KEYPOINTS * 2 + 4
    else:
        dim = cfg.NUM_CLASSES + 4
    targets_center = torch.zeros((batch, fsize[0], fsize[1], dim), dtype=torch.float, device=device)

    for i in range(batch):
        boxes_ltrb = targets[i]['boxes'].to(device)
        boxes_xywh = ltrb2xywh(boxes_ltrb)
        labels = targets[i]['labels'].to(device)
        if cfg.NUM_KEYPOINTS > 0:
            keypoints = targets[i]['keypoints'].to(device)
        else:
            keypoints = None
        # 共享内存加速
        match4center(boxes_xywh, labels, fsize, targets_center[i], cfg.NUM_KEYPOINTS, keypoints)

        if cfg.IS_VISUAL:
            # -------------------目标可视化--------------------
            import json
            # json_file = open(os.path.join(cfg.PATH_DATA_ROOT, 'ids_classes_widerface_proj.json'), 'r', encoding='utf-8')
            # json_file = open(os.path.join(cfg.PATH_DATA_ROOT, 'ids_classes_voc_proj.json'), 'r', encoding='utf-8')
            # ids2classes = json.load(json_file, encoding='utf-8')  # json key是字符
            ids2classes = {'1': 'aaaa'}

            from torchvision.transforms import functional as transformsF
            from f_tools.pic.enhance.f_data_pretreatment import f_recover_normalization4ts
            img_ts = images[i]
            img_ts = f_recover_normalization4ts(img_ts)
            img_pil = transformsF.to_pil_image(img_ts).convert('RGB')
            print([ids2classes[str(int(label))] for label in labels])
            print('fsize:', fsize)
            # img_pil.show()

            '''plt画图部分'''
            from matplotlib import pyplot as plt
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
            plt.rcParams['axes.unicode_minus'] = False
            data_hot = torch.zeros_like(targets_center[i, :, :, 0])
            for label in labels.unique():
                print(ids2classes[str(int(label))])
                torch.max(data_hot, targets_center[i, :, :, label - 1], out=data_hot)
            plt.imshow(data_hot)
            plt.imshow(img_pil.resize(fsize), alpha=0.7)
            plt.colorbar()
            # x,y表示横纵坐标，color表示颜色：'r':红  'b'：蓝色 等，marker:标记，edgecolors:标记边框色'r'、'g'等，s：size大小
            xys_f = boxes_xywh[:, :2] * fsize
            plt.scatter(xys_f[:, 0], xys_f[:, 1], color='r', s=5)

            boxes_ltrb_f = boxes_ltrb * fsize.repeat(2)
            current_axis = plt.gca()
            for i, box_ltrb_f in enumerate(boxes_ltrb_f):
                l, t, r, b = box_ltrb_f
                # ltwh
                current_axis.add_patch(plt.Rectangle((l, t), r - l, b - t, color='green', fill=False, linewidth=2))
                current_axis.text(l, t - 2, ids2classes[str(int(labels[i]))], size=8, color='white',
                                  bbox={'facecolor': 'green', 'alpha': 0.6})
            plt.show()

    return images, targets_center


'''
python /AI/temp/tmp_pycharm/DL/object_detection/z_center/train_center.py
'''

if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    init_od()
    device, cfg, ids2classes = base_set(CFG)
    custom_set(cfg)

    cfg.NUM_CLASSES = 1  # 这里与数据集相关
    model, optimizer, lr_scheduler, start_epoch = init_model(cfg, device, id_gpu=None)

    # loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler = data_loader4voc(
    #     model.cfg,
    #     is_mgpu=False,
    #     ids2classes=ids2classes)

    # loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler = data_loader4widerface(
    #     model.cfg,
    #     is_mgpu=False,
    # )

    loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler = data_loader4raccoon200(
        model.cfg,
        is_mgpu=False,
    )

    train_eval4od(start_epoch=start_epoch, model=model, optimizer=optimizer,
                  fdatas_l2=fdatas_l2, lr_scheduler=lr_scheduler,
                  loader_train=loader_train, loader_val_fmap=loader_val_fmap, loader_val_coco=loader_val_coco,
                  device=device, train_sampler=None, eval_sampler=None,
                  tb_writer=None,
                  )
    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
