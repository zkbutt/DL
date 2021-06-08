import json
import os

import cv2
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab

from f_tools.GLOBAL_LOG import flog
import skimage.io as io

CLSID2CATID = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16,
               15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31,
               27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43,
               39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56,
               51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72,
               63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85,
               75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

CATID2CLSID = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14,
               17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26,
               32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38,
               44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50,
               57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62,
               73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74,
               86: 75, 87: 76, 88: 77, 89: 78, 90: 79}


def f_show_coco_net_pic():
    # 加载公交车示例
    id = 233727
    img_info = coco.loadImgs(id)[0]

    img = io.imread(img_info['coco_url'])
    flog.debug('加载图片成功 %s', img_info)

    # 获取该图片的所有标注的id
    annIds = coco.getAnnIds(imgIds=img_info['id'])
    anns = coco.loadAnns(annIds)  # annotation 对象
    flog.debug('anns %s', anns)

    plt.axis('off')
    plt.imshow(img)
    coco.showAnns(anns)  # 显示标注
    plt.show()


def f_open_cocoimg(path_img, coco, img_id):
    img_info = coco.loadImgs([img_id])
    img_pil = Image.open(os.path.join(path_img, img_info[0]['file_name'])).convert('RGB')
    return img_pil


def f_show_coco_pics(coco, path_img, ids_img=None):
    '''
    遍历所有图片打开查看
    :param coco:
    :param path_img:
    :return:
    '''

    # id = 1
    # imgIds = coco.getImgIds(imgIds=[id])
    if not ids_img:
        ids = coco.getImgIds()
    else:
        ids = ids_img
    for id in ids:
        img_info = coco.loadImgs([id])  # 这里原始返回list
        # 本地加载 h,w,c
        # img = io.imread(os.path.join(path_img, img_info[0]['file_name']))
        file = os.path.join(path_img, img_info[0]['file_name'])
        if not os.path.exists(file):
            raise Exception('path不存在%s' % file)

        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # flog.debug('imgsize %s', img.shape[:2][::-1])
        # 加载图片基本信息 h w id filename
        # 获取该图片的所有标注的id
        annIds = coco.getAnnIds(imgIds=img_info[0]['id'])
        anns = coco.loadAnns(annIds)  # annotation 对象

        # img_pil = f_open_cocoimg(path_img, coco, id)
        # for ann in anns:
        #     box_ltwh = np.array(ann['bbox'])[None]
        #     print(box_ltwh)  # ltwh
        #     box_ltrb = ltwh2ltrb(box_ltwh)
        #     show_anc4pil(img_pil, box_ltrb)

        flog.warning('宽高:%s x %s' % (img.shape[1], img.shape[0]))
        plt.axis('off')
        plt.imshow(img)
        coco.showAnns(anns)  # 显示标注
        plt.show()
        # plt.savefig("test.png")


def f_look_coco_type(coco):
    '''
    查看coco的类别分布 类别名	图片数量	标注框数量
    :param coco:
    :return:
    '''
    cats = coco.loadCats(coco.getCatIds())
    cat_names = [cat['name'] for cat in cats]
    for cat_name in cat_names:
        catId = coco.getCatIds(catNms=cat_name)
        imgId = coco.getImgIds(catIds=catId)
        annId = coco.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)
        print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))


def t_coco_json():
    path_root = r'm:/AI/datas/widerface/coco'  # 自已的数据集
    # data_type = 'val2017'  # 自动会添加 imgages
    data_type = 'train2017'  # 自动会添加 imgages
    mode = 'bbox'
    if mode == 'bbox':
        name_file = 'instances'
    elif mode == 'keypoints':
        name_file = 'person_keypoints'
    else:
        raise Exception('mode 错误', mode)
    file_json = '{}/annotations/{}_{}.json'.format(path_root, name_file, data_type)
    # file_ann = '{}/annotations/person_keypoints_{}.json'.format(path_root, data_type)
    # 图片的根目录
    path_img = os.path.join(path_root, 'images', data_type)
    # 初始化标注数据的 COCO api
    coco = COCO(file_json)
    return coco, path_img


def t_coco_json2():
    path_host = 'M:'
    path_root = path_host + r'/AI/datas/VOC2007'
    # path_root = path_host + r'/AI/datas/coco2017'
    # path_root = path_host + r'/AI/datas/raccoon200'
    # file_json = path_root + '/annotations/instances_val2017.json'
    file_json = path_root + '/coco/annotations/instances_type4_train_994.json'
    # file_json = path_root + '/coco/annotations/instances_train_5011.json'
    # file_json = path_root + '/coco/annotations/instances_type3_train_1096.json'
    # file_json = path_root + '/coco/annotations/instances_train_5011.json'
    # path_img = path_root + '/VOCdevkit/JPEGImages'
    # path_img = path_root + '/images/train2017'
    path_img = path_root + '/train/JPEGImages'
    coco = COCO(file_json)
    return coco, path_img


def cre_coco_cls_file(path):
    '''
    生成 coco_cls 文件
    :param path:
    :return:
    '''
    coco, path_img = t_coco_json2()
    cats = coco.loadCats(coco.getCatIds())
    cls_names = []
    ids_cls = {}
    cls_ids = {}
    for i, ds in enumerate(cats):
        cls_names.append(ds['name'])
        ids_cls[ds['id']] = ds['name']
        cls_ids[ds['name']] = ds['id']

    fh = open(os.path.join(path, 'classes_name.txt'), 'w', encoding='utf-8')
    fh.write('&&'.join(cls_names))
    fh.close()

    with open(os.path.join(path, 'ids_cls.json'), 'w', encoding='utf-8') as f:
        json.dump(ids_cls, f, ensure_ascii=False, )

    with open(os.path.join(path, 'cls_ids.json'), 'w', encoding='utf-8') as f:
        data = json.dumps(cls_ids)
        f.write(data)


if __name__ == '__main__':
    '''
    coco出来是 ltwh
    关联 coco_dataset
    '''
    cre_coco_cls_file('M:\AI\datas\coco2017')

    pylab.rcParams['figure.figsize'] = (8.0, 10.0)
    # 加载coco数据集
    # coco, path_img = t_coco_json()
    coco, path_img = t_coco_json2()

    ''' 查看 数据 集 '''
    # f_show_coco_pics(coco, path_img, ids_img=[279])
    # f_show_coco_pics(coco, path_img)

    # 查看具体类名 ID从1开始 [{'id': 1, 'name': 'aeroplane'}, {'id': 2, 'name': 'bicycle'}, {'id': 3, 'name': 'bird'}, {'id': 4, 'name': 'boat'}, {'id': 5, 'name': 'bottle'}, {'id': 6, 'name': 'bus'}, {'id': 7, 'name': 'car'}, {'id': 8, 'name': 'cat'}, {'id': 9, 'name': 'chair'}, {'id': 10, 'name': 'cow'}, {'id': 11, 'name': 'diningtable'}, {'id': 12, 'name': 'dog'}, {'id': 13, 'name': 'horse'}, {'id': 14, 'name': 'motorbike'}, {'id': 15, 'name': 'person'}, {'id': 16, 'name': 'pottedplant'}, {'id': 17, 'name': 'sheep'}, {'id': 18, 'name': 'sofa'}, {'id': 19, 'name': 'train'}, {'id': 20, 'name': 'tvmonitor'}]
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]  # 单独获取 类别名称（category name）
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    # 大类 自已的数据集没有大类
    # nms = set([cat['supercategory'] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join(nms)))

    # [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
    dataset = coco.dataset  # 获取整个标注文件json对象
    # print(dataset)

    '''---------------分析--------------'''
    ids = coco.getImgIds()
    for i in ids:
        _img_info = coco.loadImgs(i)[0]
        # print(_img_info)
        w, h = _img_info['width'], _img_info['height']
    # f_look_coco_type(coco)
