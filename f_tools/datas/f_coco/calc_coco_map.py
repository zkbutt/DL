import os

import torch
import numpy as np

from f_tools.datas.f_coco.coco_eval import CocoEvaluator
from f_tools.datas.f_coco.convert_data.coco_dataset import CocoDataset


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


if __name__ == '__main__':
    # pylab.rcParams['figure.figsize'] = (10.0, 8.0)
    # print(torch.get_num_threads())  # CPU核心数

    path_root = r'M:\temp\11\voc_coco\coco'  # 自已的数据集
    data_type = 'val2017'  # 自动会添加 imgages
    mode = 'bbox' # keypoints
    path_img = os.path.join(path_root, 'images', data_type)

    dataset = CocoDataset(path_root, mode, data_type)
    coco_gt = dataset.coco

    ids = coco_gt.getImgIds()
    coco_eval = CocoEvaluator(coco_gt, [mode])

    t_cre_pdatas(coco_eval, ids)

    coco_eval.synchronize_between_processes()
    coco_eval.accumulate()  # 积累
    coco_eval.summarize()  # 总结
    # [0.9964964476743241, 0.9964964476743239, 0.9964964476743239, 1.0, 1.0, 0.9964562827964212, 0.7050492610837438, 0.9983579638752053, 0.9997947454844007, 1.0, 1.0, 0.9997541789577189]
    result_info = coco_eval.coco_eval[mode].stats.tolist()

    d = 1
