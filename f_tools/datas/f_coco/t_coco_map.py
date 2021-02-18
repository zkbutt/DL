import json
import tempfile

import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from f_tools.datas.f_coco.coco_api import t_coco_json2
from f_tools.datas.f_coco.coco_eval import CocoEvaluator


def t_cre_pdatas(coco_eval, ids):
    '''

    :param coco_eval:
    :param ids:
    :return:
    '''
    coco_gt = coco_eval.coco_gt
    res = {}
    for id in ids:
        img_info = coco_gt.loadImgs([id])  # 这里原始返回list
        annIds = coco_gt.getAnnIds(imgIds=img_info[0]['id'])
        anns = coco_gt.loadAnns(annIds)
        boxes_ltwh = []
        labels = []
        scores = []
        for ann in anns:
            # coco默认的是 ltwh list真实值
            boxes_ltwh.append(ann['bbox'])
            labels.append(ann['category_id'])
            scores.append(1)
        boxes_ltwh = torch.tensor(np.array(boxes_ltwh))
        # bbox_xywh = ltrb2xywh(ltwh2ltrb(boxes_ltwh))
        res[id] = {
            'boxes': boxes_ltwh,
            'labels': torch.tensor(np.array(labels)),
            'scores': torch.tensor(np.array(scores)),
        }

    coco_eval.update(res)


def ft1(coco_gt):
    ids = coco_gt.getImgIds()
    coco_eval = CocoEvaluator(coco_gt, ['bbox'])

    t_cre_pdatas(coco_eval, ids)

    coco_eval.synchronize_between_processes()
    coco_eval.accumulate()  # 积累
    coco_eval.summarize()  # 总结
    # [0.9964964476743241, 0.9964964476743239, 0.9964964476743239, 1.0, 1.0, 0.9964562827964212, 0.7050492610837438, 0.9983579638752053, 0.9997947454844007, 1.0, 1.0, 0.9997541789577189]
    result_info = coco_eval.coco_eval['bbox'].stats.tolist()


def ft2(coco_gt):
    coco_eval_data1 = []
    ids = coco_gt.getImgIds()
    for id in ids:
        img_info = coco_gt.loadImgs([id])  # 这里原始返回list
        annIds = coco_gt.getAnnIds(imgIds=img_info[0]['id'])
        anns = coco_gt.loadAnns(annIds)
        for ann in anns:
            coco_eval_data1.append({"image_id": ann['image_id'],
                                    "category_id": ann['category_id'],
                                    "bbox": ann['bbox'],
                                    "score": 0.8})
    _, tmp = tempfile.mkstemp()
    json.dump(coco_eval_data1, open(tmp, 'w'))

    coco_dt = coco_gt.loadRes(tmp)
    coco_eval_obj = COCOeval(coco_gt, coco_dt, 'bbox')
    # coco_eval_obj.params.imgIds = [0]
    coco_eval_obj.evaluate()
    coco_eval_obj.accumulate()
    coco_eval_obj.summarize()
    # return coco_eval_data1


if __name__ == '__main__':
    coco_gt, path_img = t_coco_json2()
    # coco_gt = COCO('./coco_annotation_t.json')

    # ft1(coco_gt)
    ft2(coco_gt)
    # coco_eval_data1 = ft2(coco_gt)
    #
    # path_img = r'M:/AI/datas/VOC2007/val/JPEGImages'
    # f_show_coco_pics(coco_gt, path_img, ids_img=[0])

    # coco_eval_data1 = []
    # coco_eval_data1.append({"image_id": 0, "category_id": 12, "bbox": tolist, "score": 0.8})
    # coco_eval_data1.append({"image_id": 0, "category_id": 12, "bbox": [48, 240, 147, 131], "score": 0.8})
    # coco_eval_data1.append({'id': 0, 'image_id': 0, 'category_id': 1, 'bbox': [449, 330, 122, 149], "score": 0.8})

    # annIds = coco_gt.getAnnIds(imgIds=[0])
    # coco_eval_data1 = coco_gt.loadAnns(annIds)  # annotation 对象
    # coco_eval_data1[0]['score'] = 0.7

    # print(coco_dt)
