import os
import numpy as np
import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_pretreatment import Compose, Resize, RandomHorizontalFlip, ToTensor
from f_tools.fun_od.f_anc import AnchorsFound
from f_tools.fun_od.f_boxes import pos_match
from object_detection.coco_t.coco_api import t_coco_pic
from object_detection.coco_t.coco_dataset import CocoDataset

if __name__ == '__main__':
    path_data_root = r'M:\AI\datas\widerface\coco'
    batch_size = 64

    data_transform = {
        "train": Compose([
            Resize((640, 640)),  # (h,w)
            # RandomHorizontalFlip(1),
            ToTensor(),
        ]),
        "val": Compose([ToTensor()])
    }

    dataset_train = CocoDataset(path_data_root, 'keypoints', 'train2017')

    # def collate_fn(batch_datas):
    #     _t = batch_datas[0][0]
    #     # images = torch.empty((len(batch_datas), *_t.shape), device=_t.device)
    #     images = torch.empty((len(batch_datas), *_t.shape)).to(_t)
    #     targets = []
    #     for i, (img, taget) in enumerate(batch_datas):
    #         images[i] = img
    #         targets.append(taget)
    #     return images, targets
    #
    #
    # loader_train = torch.utils.data.DataLoader(
    #     dataset_train,
    #     batch_size=batch_size,
    #     num_workers=0,
    #     shuffle=True,
    #     # pin_memory=True,  # 不使用虚拟内存 GPU要报错
    #     drop_last=True,  # 除于batch_size余下的数据
    #     collate_fn=collate_fn,
    # )

    flog.debug('len %s', len(dataset_train))
    coco = dataset_train.coco
    id_img = 9227
    # id_img = 12411

    # img_info = coco.loadImgs([id_img])  # 这里原始返回list
    # annIds = coco.getAnnIds(imgIds=img_info[0]['id'])
    # anns = coco.loadAnns(annIds)  # annotation 对象
    # for ann in anns:
    #     if np.all(np.array(ann['bbox']) == 0, axis=0):
    #         flog.error('出错 %s %s', img_info, ann['bbox'])

    # 查看指定图片
    # imgs_info = coco.loadImgs(id_img)[0]
    # path_save_img = os.path.join(dataset_train.path_root, 'images', dataset_train.data_type)
    # t_coco_pic(coco, path_save_img,id_img)

    for id in range(len(coco.getImgIds())):
        img_info = coco.loadImgs([id])  # 这里原始返回list
        # 本地加载 h,w,c
        # img = io.imread(os.path.join(path_img, img_info[0]['file_name']))
        # 加载图片基本信息 h w id filename
        # 获取该图片的所有标注的id
        annIds = coco.getAnnIds(imgIds=img_info[0]['id'])
        anns = coco.loadAnns(annIds)  # annotation 对象
        for ann in anns:
            if np.all(np.array(ann['bbox']) == 0, axis=0):
                flog.error('出错 %s %s', img_info, ann['bbox'])

    # print(imgs_info)
    # print(coco.getCatIds(id_img))
    # print(coco.loadCats(coco.getCatIds()))
    # print(coco.loadAnns(id_img))

    # ANCHORS_SIZE = [[16, 32], [64, 128], [256, 512]]
    # FEATURE_MAP_STEPS = [8, 16, 32]  # 特图的步距
    # ANCHORS_VARIANCE = [0.1, 0.2]  # 修复系数 中心0.1 长宽0.2
    #
    # anchors = AnchorsFound((640, 640), ANCHORS_SIZE, FEATURE_MAP_STEPS).get_anchors()
    # device = torch.device('cuda:0')
    # bboxs = torch.tensor([0, 0, 0, 0], dtype=torch.float).to(device)
    # pos_match(anchors.to(device), bboxs[None].to(device), 0.5)
