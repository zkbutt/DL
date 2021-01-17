import os

import torch
from PIL import Image
from tqdm import tqdm

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.f_coco.coco_api import f_show_coco_pics
from f_tools.datas.f_coco.convert_data.coco_dataset import load_dataset_coco, CocoDataset
from f_tools.datas.f_coco.convert_data.csv2coco import to_coco, to_coco_v2
import numpy as np

from f_tools.fun_od.f_boxes import xywh2ltrb, ltwh2ltrb

if __name__ == '__main__':
    '''
    coco出来是 ltwh
    '''
    mode = 'bbox'  # bbox segm keypoints caption
    path_host = 'M:'
    # path_host = ''
    # path_root = path_host + r'/AI/datas/VOC2012'  # 自已的数据集
    path_root = path_host + r'/AI/datas/VOC2007'  # 自已的数据集

    # type = 'train'
    # data_type = 'train_5011'  # train2017 val2017 自动匹配json文件名

    type_img = 'val'
    type_json = 'test'
    data_type = 'val_1980'  # train2017 val2017 自动匹配json文件名
    data_type = 'test_2972'  # train2017 val2017 自动匹配json文件名

    name = 'type3' + '_' + type_json  # 文件名
    s_ids_cats = [3, 8, 12]  # 汽车 1284 ,dog 1341, 人person 9583
    nums_cat = [1000, 1000, 1000]  # 类型的最大数量

    path_coco_target = os.path.join(path_root, 'coco/annotations')
    path_img = os.path.join(path_root, type_img, 'JPEGImages')

    dataset = CocoDataset(path_coco_target, path_img, mode, data_type,
                          s_ids_cats=s_ids_cats,
                          nums_cat=nums_cat,
                          )

    classes_ids = {}  # name int
    ids_classes = {}  # name int
    classes_name = []
    for i, cat_id in enumerate(dataset.s_ids_cats, start=1):
        cat_name = dataset.ids_classes[cat_id]
        classes_ids[cat_name] = i
        ids_classes[cat_id] = cat_name
        classes_name.append(cat_name)

    file = os.path.join(path_root, 'classes_name_cats.txt')
    if not os.path.exists(file):
        with open(file, 'w') as f:
            f.write(' '.join(classes_name))

    coco = dataset.coco

    annotations = []
    no_match = []
    for id_img in tqdm(dataset.ids_img, desc='文件ID遍历'):
        # f_show_coco_pics(coco, path_img, ids_img=[id_img])

        info_img = coco.loadImgs(id_img)[0]
        file_name = info_img['file_name']
        ids_ann = coco.getAnnIds(id_img)
        infos_ann = coco.loadAnns(ids_ann)

        file_img = os.path.join(path_img, file_name)
        img_pil = Image.open(file_img).convert('RGB')

        for info_ann in infos_ann:
            bbox_xywh = info_ann['bbox']
            bbox_ltrb = ltwh2ltrb(torch.tensor(bbox_xywh)[None]).squeeze(0).tolist()
            id_cat = info_ann['category_id']
            if id_cat in dataset.s_ids_cats:
                # 确保只有选择的这几类
                cat_name = ids_classes[id_cat]
                _l = [file_name, *bbox_ltrb, cat_name]
                annotations.append(_l)

                # show_bbox4pil(img_pil, torch.tensor(bbox_ltrb))
            else:
                no_match.append([info_img, info_ann])
    if len(annotations) == 0:
        raise Exception('len(annotations)==0 没有这个类型 %s' % s_ids_cats)
    # flog.debug('no_match : %s', no_match)
    annotations = np.array(annotations)
    # # 文件名, ltrb + keys 类型名

    to_coco_v2(annotations, classes_ids, path_img, path_root, mode, is_copy=False, is_move=False,
               type=name + '_' + str(len(dataset.ids_img)))
