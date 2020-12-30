import os

from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab

from f_tools.GLOBAL_LOG import flog
from f_tools.fun_od.f_boxes import ltwh2ltrb
from f_tools.pic.f_show import show_anc4pil
import skimage.io as io


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
        img = io.imread(os.path.join(path_img, img_info[0]['file_name']))
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

        print('宽高', img.shape[1], img.shape[0])
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
    mode = 'keypoints'
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


if __name__ == '__main__':
    '''
    coco出来是 ltwh
    '''
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)
    coco, path_img = t_coco_json()

    f_show_coco_pics(coco, path_img, ids_img=[279])

    # 查看具体类 ID从1开始 [{'id': 1, 'name': 'aeroplane'}, {'id': 2, 'name': 'bicycle'}, {'id': 3, 'name': 'bird'}, {'id': 4, 'name': 'boat'}, {'id': 5, 'name': 'bottle'}, {'id': 6, 'name': 'bus'}, {'id': 7, 'name': 'car'}, {'id': 8, 'name': 'cat'}, {'id': 9, 'name': 'chair'}, {'id': 10, 'name': 'cow'}, {'id': 11, 'name': 'diningtable'}, {'id': 12, 'name': 'dog'}, {'id': 13, 'name': 'horse'}, {'id': 14, 'name': 'motorbike'}, {'id': 15, 'name': 'person'}, {'id': 16, 'name': 'pottedplant'}, {'id': 17, 'name': 'sheep'}, {'id': 18, 'name': 'sofa'}, {'id': 19, 'name': 'train'}, {'id': 20, 'name': 'tvmonitor'}]
    # cats = coco.loadCats(coco.getCatIds())
    # nms = [cat['name'] for cat in cats]  # 单独获取 类别名称（category name）
    # print('COCO categories: \n{}\n'.format(' '.join(nms)))

    # 大类 自已的数据集没有大类
    # nms = set([cat['supercategory'] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join(nms)))

    # [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
    dataset = coco.dataset  # 获取整个标注文件json对象
    # print(dataset)

    '''---------------分析--------------'''
    # ids = coco.getImgIds()
    # for i in ids:
    #     _img_info = coco.loadImgs(i)[0]
    #     # print(_img_info)
    #     w, h = _img_info['width'], _img_info['height']
    # f_look_coco_type(coco)
