from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from pycocotools.cocoeval import COCOeval

from f_tools.GLOBAL_LOG import flog


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


def 查看本地图片():
    id = 1
    # 指定返回一张索引
    imgIds = coco.getImgIds(imgIds=[id])
    bzs = coco.loadImgs([id])  # 这里原始返回list
    # 本地加载 h,w,c
    img = io.imread('%s/images/%s/%s' % (dataDir, dataType, bzs[0]['file_name']))
    # 加载图片基本信息 h w id filename
    # 获取该图片的所有标注的id
    annIds = coco.getAnnIds(imgIds=bzs[0]['id'])
    anns = coco.loadAnns(annIds)  # annotation 对象
    plt.axis('off')
    plt.imshow(img)
    coco.showAnns(anns)  # 显示标注
    plt.show()


def f_show(img, anns):
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


def f评估():
    path_res = ''
    cocoGt = COCO(annFile)  # 标注文件的路径及文件名，json文件形式
    cocoDt = cocoGt.loadRes('my_result.json')  # 自己的生成的结果的路径及文件名，json文件形式
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")  # segm表示分割  bbox目标检测 keypoints关键点检测
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)

    # dataDir = 'D:\down\AI\datas\coco2017'
    dataDir = r'd:\t001\coco'  # 自已的数据集
    dataType = 'val2017'  # 自动会添加 imgages
    # dataType = 'train2017'
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    # annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)

    # 初始化标注数据的 COCO api
    coco = COCO(annFile)

    # 查看具体类
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]  # 单独获取 类别名称（category name）
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    # 大类 自已的数据集没有大类
    # nms = set([cat['supercategory'] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join(nms)))

    # f查看网络图片()

    查看本地图片()

    # f_show(img)

    # [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
    dataset = coco.dataset  # 获取整个标注文件
    print(dataset)

    '''---------------评估--------------'''
    # f评估()
