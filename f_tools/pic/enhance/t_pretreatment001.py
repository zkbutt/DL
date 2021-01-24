import os

import cv2
import torch
from pycocotools.coco import COCO
import numpy as np

from f_tools.GLOBAL_LOG import flog
from f_tools.fun_od.f_boxes import ltwh2ltrb
from f_tools.pic.enhance.f_data_pretreatment4np import SSDAugmentation, RandomMirror, Expand, Normalize, \
    ToPercentCoords, RandomSaturation, ConvertFromInts, RandomSampleCrop, cre_transform_resize4np
from f_tools.pic.f_show import f_plt_od_np, f_plt_show_cv, f_plt_show_ts

if __name__ == '__main__':
    # coco_obj加载
    path_root = r'M:/AI/datas/VOC2007'  # 自已的数据集
    file_json = os.path.join(path_root, 'coco/annotations') + '/instances_type3_train_1096.json'  # 1096
    path_img = os.path.join(path_root, 'train', 'JPEGImages')

    coco_obj = COCO(file_json)
    id_ = 1
    image_info = coco_obj.loadImgs(id_)[0]
    file_img = os.path.join(path_img, image_info['file_name'])
    img_np = cv2.imread(file_img)

    annotation_ids = coco_obj.getAnnIds(id_)
    bboxs = np.zeros((0, 4), dtype=np.float32)  # np创建 空数组 高级
    labels = []

    coco_anns = coco_obj.loadAnns(annotation_ids)
    for a in coco_anns:
        # skip the annotations with width or height < 1
        if a['bbox'][2] < 1 or a['bbox'][3] < 1:
            flog.warning('标记框有问题 %s', a)
            continue

        labels.append(a['category_id'])
        bbox = np.zeros((1, 4), dtype=np.float32)
        bbox[0, :4] = a['bbox']
        bboxs = np.append(bboxs, bbox, axis=0)

    bboxs_ltrb = ltwh2ltrb(bboxs)


    # f_plt_show_cv(img_np, gboxes_ltrb=torch.tensor(bboxs_ltrb), labels_text=labels, is_recover_size=False)

    # fd = RandomMirror()
    # fd = Expand(mean=(0.406, 0.456, 0.485))
    # fd = RandomSampleCrop()
    # fd = Normalize(mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229))
    # fd = ToPercentCoords()
    # fd = ConvertFromInts()

    class cfg:
        pass


    cfg.IMAGE_SIZE = (448, 448)
    cfg.PIC_MEAN = (0.406, 0.456, 0.485)
    cfg.PIC_STD = (0.225, 0.224, 0.229)
    fd = cre_transform_resize4np(cfg)['train']

    img, boxes, classes = fd(img_np, bboxs_ltrb, np.array(labels))
    # fd = RandomSaturation()
    # img, boxes, classes = fd(img_np, bboxs_ltrb, np.array(labels))

    # f_plt_show_cv(img, gboxes_ltrb=torch.tensor(bboxs_ltrb), pboxes_ltrb=boxes,
    #               labels_text=labels,
    #               is_recover_size=False)

    f_plt_show_ts(img, gboxes_ltrb=torch.tensor(bboxs_ltrb), pboxes_ltrb=boxes,
                  labels_text=labels,
                  is_recover_size=False)
    # ssd_augmentation = SSDAugmentation()
    # ssd_augmentation(img_np, bboxs_ltrb, np.array(labels))
