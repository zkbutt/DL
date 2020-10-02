import json
import os

import xmltodict
from PIL import Image
import numpy as np

from object_detection.coco_t.csv2coco import Csv2CocoInstances


def f1(file_txt, path_xml, path_img, classes_ids_voc):
    file_annotations_dict = {}

    with open(file_txt) as f:
        # 读每一行加上路径和扩展名---完整路径
        for line in f.readlines():
            name_file = line.strip()
            file_xml = os.path.join(path_xml, name_file + ".xml")

            # dict(文件名:np(框个数,5)...)   5是 bbox4+lable1
            boxes = np.empty(shape=(0, 5), dtype=np.float)

            with open(file_xml) as file:
                str_xml = file.read()
            doc = xmltodict.parse(str_xml)
            _objs = doc['annotation']['object']

            if isinstance(_objs, dict):
                xmin = float(_objs['bndbox']['xmin'])
                ymin = float(_objs['bndbox']['ymin'])
                xmax = float(_objs['bndbox']['xmax'])
                ymax = float(_objs['bndbox']['ymax'])
                lable = classes_ids_voc[_objs['name']]
                boxes = np.concatenate((boxes, np.array([xmin, ymin, xmax, ymax, lable])[None]), axis=0)
            else:
                for obj in _objs:
                    # 可能有多个目标
                    xmin = float(obj['bndbox']['xmin'])
                    ymin = float(obj['bndbox']['ymin'])
                    xmax = float(obj['bndbox']['xmax'])
                    ymax = float(obj['bndbox']['ymax'])
                    lable = classes_ids_voc[obj['name']]
                    boxes = np.concatenate((boxes, np.array([xmin, ymin, xmax, ymax, lable])[None]), axis=0)
            file_annotations_dict[name_file + '.jpg'] = boxes
    coco_generate = Csv2CocoInstances(path_img, classes_ids_voc, file_annotations_dict)


if __name__ == '__main__':
    '''
    训练集与验证集在一起
    '''
    path_classname = './_file/classes_ids_voc.json'

    with open(path_classname, 'r', encoding='utf-8') as f:
        classes_ids_voc = json.load(f)  # 文件转dict 或list

    file_name = ['train_s.txt', 'val_s.txt']
    # file_name = ['train.txt', 'val.txt']
    type = 'val2017'
    file_txt = os.path.join(r'M:\AI\datas\VOC2012\trainval', file_name[0])
    path_xml = r'M:\AI\datas\VOC2012\trainval\Annotations'  # 这是xml文件所在的地址
    path_img = r'M:\AI\datas\VOC2012\trainval\JPEGImages'  # 真实图片路径
    mode = 'bboxs'
    f1(file_txt, path_xml, path_img, classes_ids_voc)
