import glob
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from tqdm import tqdm

from f_tools.datas.f_coco.convert_data.coco_dataset import load_dataset_coco
from f_tools.f_general import show_time
from f_tools.fun_od.f_boxes import ltrb2xywh
from f_tools.fun_od.kmeans_anc.kmeans import kmeans, avg_iou


def load_voc_boxes():
    '''
    读取将所有框按ltrb 归一化成框组合
    :param path:
    :return:
    '''
    # path_annotations = r"M:/AI/datas/VOC2012/train/Annotations"
    path_annotations = r"/AI/datas/VOC2012/train/Annotations"
    boxes_wh = []
    for xml_file in tqdm(glob.glob("{}/*xml".format(path_annotations))):  # 获取路径下文件名
        tree = ET.parse(xml_file)

        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))

        for obj in tree.iter("object"):
            xmin = float(obj.findtext("bndbox/xmin")) / width
            ymin = float(obj.findtext("bndbox/ymin")) / height
            xmax = float(obj.findtext("bndbox/xmax")) / width
            ymax = float(obj.findtext("bndbox/ymax")) / height

            boxes_wh.append([xmax - xmin, ymax - ymin])  # w,h
    path_img = '/AI/datas/VOC2012/train/JPEGImages'
    calc_pic_mean(path_img)
    return np.array(boxes_wh)


def load_coco_boxes(keep=False, is_calc_pic_mean=True):
    '''
    读取将所有框按ltrb 归一化成框组合
    :param path:
    :return:
    '''
    mode = 'bbox'  # bbox segm keypoints caption
    path_img, dataset = load_dataset_coco(mode)
    # path_img, dataset = load_raccoon(mode)
    print('len(dataset)', len(dataset))
    coco = dataset.coco

    boxes_wh = []
    for data in tqdm(dataset):
        # print(data)
        img_pil, target = data
        boxes_ltrb = target['boxes']
        boxes_xywh = ltrb2xywh(boxes_ltrb)
        if not keep:
            wh_ = target['size'].max().repeat(2)[None]
        else:
            wh_ = target['size'][None]
        bb = (boxes_xywh[:, 2:] / wh_).tolist()
        for b in bb:
            boxes_wh.append(b)
    if is_calc_pic_mean:
        calc_pic_mean(path_img)
    return np.array(boxes_wh)


def calc_anc_size(data, clusters, size_in, is_calc_pic_mean=True):
    out_np = kmeans(data, k=clusters)  # 输出5,2
    print("Accuracy: {:.2f}%".format(avg_iou(data, out_np) * 100))
    # 计算尺寸大小排序后的索引
    a = out_np[:, 0] * out_np[:, 1]
    indexs = np.argsort(a)  # 默认升序
    # indexs = np.argsort(-a)  # 降序
    print("Boxes:\n {}".format(out_np.round(3)[indexs].tolist()))
    print("Boxes:\n {}".format(out_np[indexs]))
    print("size:\n {}".format((out_np * size_in)[indexs]))
    ratios = np.around(out_np[:, 0] / out_np[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))


def calc_pic_mean(path_img):
    img_file_names = os.listdir(path_img)
    m_list, s_list = [], []
    for img_filename in tqdm(img_file_names, desc='计算图片均值'):
        img = cv2.imread(path_img + '/' + img_filename)
        img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print("mean bgr = ", m[0].tolist())
    print("std bgr = ", s[0].tolist())
    print("mean rgb= ", m[0][::-1].tolist())
    print("std rgb= ", s[0][::-1].tolist())


if __name__ == '__main__':
    '''
    VOC2012 17125 6 Accuracy: 63.18%
    VOC2012 17125 9 Accuracy: 67.88%
    VOC2012 17125 18 Accuracy: 75.18%
    输出anc的归一化比例
    '''
    clusters = 5
    size_in = [416, 416]
    # size_in = [512, 512]

    '''-------------加载box--------------'''
    # data = load_voc_boxes() # voc直接计算,类型数据
    # data = load_coco_boxes(keep=False, is_calc_pic_mean=False)  # 基本用这个 coco_dataset
    data = load_coco_boxes(keep=False, is_calc_pic_mean=True)  # 基本用这个 coco_dataset
    # 计算anc尺寸 和图片均值
    show_time(calc_anc_size, data, clusters, size_in)

    '''计算显示'''
    size_ = 19
    array = np.array([[0.074, 0.074], [0.162, 0.146], [0.314, 0.3], [0.452, 0.506], [0.729, 0.635]]) * size_
    print(array)
