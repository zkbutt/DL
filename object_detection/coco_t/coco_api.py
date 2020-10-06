import json
import os

from PIL import ImageFont
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from pycocotools.cocoeval import COCOeval

from f_tools.GLOBAL_LOG import flog
from f_tools.f_general import NpEncoder


# class FCOCO(COCO):
#
#     def __init__(self, annotation_file=None):
#         super(FCOCO, self).__init__(annotation_file)
#
#     def loadRes(self, resFile):
#         return super().loadRes(resFile)


def f查看类别():
    '''返回类别信息'''
    # 获取指定名称的类别序号（找category name为 x 的 category id）
    catIds = coco.getCatIds(catNms=['person', 'dog', 'skateboard'])
    # 指定类别 catIds 获取类别ID对应的图片ID list   如果没有会返回0
    imgIds = coco.getImgIds(catIds=catIds)
    # # 随机选择一个ID
    ids_ = imgIds[np.random.randint(0, len(imgIds))]
    # 加载图片基本信息  ["license", "file_name","coco_url", "height", "width", "date_captured", "id"]
    return coco.loadImgs(ids_)[0]


def f查看网络图片():
    img_info = f查看类别()

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


def t_coco_pic(coco, path_img, id_img=None):
    '''
    遍历所有图片打开查看
    :param coco:
    :param path_img:
    :return:
    '''
    # id = 1
    # imgIds = coco.getImgIds(imgIds=[id])
    if not id_img:
        ids = coco.getImgIds()
    else:
        ids = [id_img]
    for id in ids:
        img_info = coco.loadImgs([id])  # 这里原始返回list
        # 本地加载 h,w,c
        img = io.imread(os.path.join(path_img, img_info[0]['file_name']))
        # 加载图片基本信息 h w id filename
        # 获取该图片的所有标注的id
        annIds = coco.getAnnIds(imgIds=img_info[0]['id'])
        anns = coco.loadAnns(annIds)  # annotation 对象
        plt.axis('off')
        plt.imshow(img)
        coco.showAnns(anns)  # 显示标注
        plt.show()


def t_show(img, anns):
    s = 7
    plt.figure(figsize=(4 * s, 3 * s))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(img)
    coco.showAnns(anns)
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    coco_kps.showAnns(annKps)


def coco_eval(coco_res_dict, coco_gt, epoch=0, mode='bbox'):
    '''

    :param coco_res_dict: coco标准的结果dict
    :param coco_gt:
    :param epoch:
    :param mode:
    :return:
    '''
    # file = './file/coco_{}_{}.json'.format(mode, epoch)
    # # flog.debug('coco_res_dict path:%s', os.path.abspath(file))
    # with open(file, 'w', encoding='utf-8') as f:
    #     json.dump(coco_res_dict, f, cls=NpEncoder, ensure_ascii=False)
    if coco_res_dict is None or len(coco_res_dict) == 0:
        flog.error('出错拉 coco_res_dict:%s coco_gt:%s epoch:%s ', coco_res_dict, coco_gt, epoch)
        return

    coco_p = coco_gt.loadRes(coco_res_dict)
    coco_eval = COCOeval(coco_gt, coco_p, mode)
    if mode == 'keypoints':
        coco_eval.params.kpt_oks_sigmas = coco_eval.params.kpt_oks_sigmas[:5]  # 5个关键点进行参数修正

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def show_od4coco(img, target, coco):
    fig, ax = plt.subplots(figsize=(12.80, 7.20))
    size = 4
    font_dict = {
        'family': 'serif',
        'style': 'italic',  # ['normal','italic','oblique']
        'weight': 'normal',  # ['light','normal','medium','semibold','bold','heavy','black']
        'color': 'red',
        # 'bold': 1,
        'size': 6
    }
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    cats = coco.loadCats(target['labels'])

    for cat, bbox, keypoint in zip(cats, target['bboxs'], target['keypoints']):
        skeleton = cat['skeleton']
        keypoint = keypoint
        print()

        # coco的关键点连接骨架  index 1开始

        sks = [[15, 13], [3, 5], [4, 6]]
        kp = np.array([0, 0, 0, 236, 240, 2, 254, 238, 2])

        # ltrb -> ltwh
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                             linewidth=2,
                             color="r",
                             fill=False)
        ax.add_patch(rect)

        # text_width, text_height = font.getsize(cat['name'])
        plt.text(bbox[0], bbox[1] - font_dict['size'], cat['name'], color='white',
                 verticalalignment='top', horizontalalignment='left',
                 fontdict=font_dict
                 )

        if keypoint[0] != 0 and keypoint[1] != 0:
            x = keypoint[0::2]
            y = keypoint[1::2]
            # 画骨架线
            for sk in skeleton:
                sk = np.array(sk) - 1
                plt.plot(x[sk], y[sk], linewidth=size - 2, color='red')
            # 画标注点
            plt.plot(x, y, 'o', markersize=size, markerfacecolor='k', markeredgecolor='aqua', markeredgewidth=size - 3)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)

    # dataDir = 'M:\datas\coco2017'
    path_root = r'd:\t001\coco'  # 自已的数据集
    # dataType = 'val2017'  # 自动会添加 imgages
    data_type = 'train2017'
    # annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    file_ann = '{}/annotations/person_keypoints_{}.json'.format(path_root, data_type)

    # 初始化标注数据的 COCO api
    coco = COCO(file_ann)

    # 查看具体类
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]  # 单独获取 类别名称（category name）
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    # 大类 自已的数据集没有大类
    # nms = set([cat['supercategory'] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join(nms)))

    # f查看网络图片()
    path_save_img = os.path.join(path_root, 'images', data_type)
    t_coco_pic(coco, path_save_img)

    # [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
    dataset = coco.dataset  # 获取整个标注文件
    print(dataset)

    '''---------------评估--------------'''
    # f评估()
    # 原图图片
    # _img_info = self.coco.loadImgs(image_id)[0]
    # w, h = _img_info['width'], _img_info['height']
