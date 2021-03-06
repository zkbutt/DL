import os

import cv2
import torch
import numpy as np

from f_tools.fun_od.f_boxes import ltwh2ltrb
from f_tools.pic.f_show import f_show_od_ts4plt, f_plt_box2, f_show_od_np4plt, show_bbox_keypoints4ts

torch.set_printoptions(linewidth=320, sci_mode=False, precision=5, profile='long')
np.set_printoptions(linewidth=320, suppress=True,
                    formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.f_coco.coco_api import f_show_coco_pics, f_open_cocoimg
from f_tools.pic.enhance.f_mosaic import f_mosaic_pics_ts


class CustomCocoDataset(Dataset):
    def __init__(self, file_json, path_img, mode, transform=None, is_mosaic=False,
                 is_mosaic_keep_wh=False, is_mosaic_fill=False, is_debug=False, cfg=None,
                 s_ids_cats=None, nums_cat=None,
                 is_img_np=True, is_boxesone=True):
        '''

        :param file_json:
        :param path_img:
        :param mode:
        :param transform: 这个需与 is_img_np 配对使用
        :param is_mosaic:
        :param is_mosaic_keep_wh:
        :param is_mosaic_fill:
        :param is_debug:
        :param cfg:
        :param s_ids_cats:
        :param nums_cat:  每一个分类的最大图片数量
        :param is_img_np: 为Ture img_np 为False 为pil  这个需与 transform 配对使用
        :param is_boxesone: 归一化gt 尺寸
        '''
        self.file_json = file_json
        self.transform = transform
        self.mode = mode
        self.coco_obj = COCO(file_json)
        if s_ids_cats is not None:
            flog.warning('指定coco类型 %s', coco.loadCats(s_ids_cats))
            self.s_ids_cats = s_ids_cats
            ids_img = []

            # 限制每类的最大个数
            if nums_cat is None:
                for idc in zip(s_ids_cats):
                    # 类型对应哪些文件 可能是一张图片多个类型
                    ids_ = self.coco_obj.getImgIds(catIds=idc)
                    ids_img += ids_
            else:
                # 限制每类的最大个数
                for idc, num_cat in zip(s_ids_cats, nums_cat):
                    # 类型对应哪些文件 可能是一张图片多个类型
                    ids_ = self.coco_obj.getImgIds(catIds=idc)[:num_cat]
                    # ids_ = self.coco.getImgIds(catIds=idc)[:1000]
                    ids_img += ids_
                    # print(ids_)  # 这个只支持单个元素

            self.ids_img = list(set(ids_img))  # 去重
        else:
            self.ids_img = self.coco_obj.getImgIds()  # 所有图片的id 画图数量

        #  self.classes_ids self.classes_ids
        self._init_load_classes()  # 除了coco数据集,其它不管

        self.is_debug = is_debug
        self.is_mosaic = is_mosaic
        self.cfg = cfg
        self.path_img = path_img
        self.is_mosaic_keep_wh = is_mosaic_keep_wh
        self.is_mosaic_fill = is_mosaic_fill
        self.is_img_np = is_img_np  # img_np False 是img_pil
        if not os.path.exists(path_img):
            raise Exception('coco path_img 路径不存在', path_img)

    def __len__(self):
        # flog.debug('__len__ %s', )
        if self.is_debug:
            return 20
        if self.is_mosaic:
            return len(self.ids_img) // 4
        return len(self.ids_img)

    def open_img_tar(self, index):
        img = self.load_image(index)
        image_id = self.ids_img[index]
        # bboxs, labels,keypoints

        tars_ = self.load_anns(index, img_wh=img.shape[:2][::-1])
        if tars_ is None:  # 没有标注返回空
            return None

        # 动态构造target
        target = {}
        l_ = ['boxes', 'labels', 'keypoints']
        target['image_id'] = image_id
        if self.is_img_np:
            target['size'] = img.shape[:2][::-1]  # (w,h)
        else:
            target['size'] = img.size  # (w,h) # 原图数据

        # 根据标注模式 及字段自动添加 target['boxes', 'labels', 'keypoints']
        for i, tar in enumerate(tars_):
            target[l_[i]] = tar
        return img, target

    def do_mosaic(self, idx):
        imgs = []
        boxs = []
        labels = []
        for i in range(idx * 4, idx * 4 + 4):
            img_pil, target = self.open_img_tar(i)
            imgs.append(img_pil)  # list(img_pil)
            boxs.append(target["boxes"])
            labels.append(target["labels"])
        img_pil_mosaic, boxes_mosaic, labels = f_mosaic_pics_ts(imgs, boxs, labels,
                                                                self.cfg.IMAGE_SIZE,
                                                                is_mosaic_keep_wh=self.is_mosaic_keep_wh,
                                                                is_mosaic_fill=self.is_mosaic_fill,
                                                                )
        target = {}
        target["boxes"] = boxes_mosaic  # 输出左上右下
        target["labels"] = labels
        target["size"] = self.cfg.IMAGE_SIZE
        return img_pil_mosaic, target

    def __getitem__(self, index):
        '''

        :param index:
        :return: tensor or np.array 根据 out: 默认ts or other is np
            img: h,w,c
            target:
            coco原装是 ltwh
            dict{
                image_id: int,
                bboxs: ts n4 原图 ltwh -> ltrb
                labels: ts n,
                keypoints: ts n,10
                size: wh
            }
        '''
        # 这里生成的是原图尺寸的 target 和img_np_uint8 (375, 500, 3)
        if self.is_mosaic and self.mode == 'bbox':
            res = self.do_mosaic(index)
        else:
            res = self.open_img_tar(index)

        if res is None:
            flog.error('这个图片没有标注信息 id为%s', index)
            return self.__getitem__(index + 1)

        img, target = res

        if len(target['boxes']) != len(target['labels']):
            flog.warning('!!! 数据有问题 1111  %s %s %s ', target, len(target['boxes']), len(target['labels']))

        '''---------------cocoAPI测试 查看图片在归一化前------------------'''
        # 这个用于调试
        # if self.cfg.IS_VISUAL_PRETREATMENT:
        #     可视化参数 is_mosaic 这个用不起
        #     f_show_coco_pics(self.coco_obj, self.path_img, ids_img=[index])

        if target['boxes'].shape[0] == 0:
            flog.warning('数据有问题 重新加载 %s', index)
            return self.__getitem__(index + 1)

        if self.transform is not None:
            img, target = self.transform(img, target)
            # if self.is_img_np:
            #     # 输入 ltrb 原图
            #     # f_plt_show_cv(img, gboxes_ltrb=target['boxes'])
            #     # img, boxes, labels = self.transform(img, target['boxes'], target['labels'])
            #     img, target = self.transform(img, target)
            #     # 这里会刷新 boxes, labels
            #     # f_plt_show_cv(img, gboxes_ltrb=boxes)
            # else:
            #     # 预处理输入 PIL img 和 np的target
            #     img, target = self.transform(img, target)

        if len(target['boxes']) != len(target['labels']):
            flog.warning('!!! 数据有问题 ttttttttt  %s %s %s ', target, len(target['boxes']), len(target['labels']))

        # target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float)
        # target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
        target['size'] = torch.tensor(target['size'], dtype=torch.float)  # 用于恢复真实尺寸
        # if self.mode == 'keypoints':
        #     target['keypoints'] = torch.tensor(target['keypoints'], dtype=torch.float)

        if target['boxes'].shape[0] == 0:
            flog.debug('二次检查出错 %s', index)
            return self.__getitem__(index + 1)

        if len(target['boxes']) != len(target['labels']):
            flog.warning('!!! 数据有问题 22222  %s %s %s ', target, len(target['boxes']), len(target['labels']))
        # flog.warning('数据debug 有问题 %s %s %s ', target, len(target['boxes']), len(target['labels']))
        return img, target

    def load_image(self, index):
        '''

        :param index:
        :return:
        '''
        image_info = self.coco_obj.loadImgs(self.ids_img[index])[0]
        file_img = os.path.join(self.path_img, image_info['file_name'])
        if not os.path.exists(file_img):
            raise Exception('file_img 加载图片路径错误', file_img)

        if self.is_img_np:
            img = cv2.imread(file_img)
        else:
            img = Image.open(file_img).convert('RGB')  # 原图数据
        return img

    def load_anns(self, index, img_wh):
        '''
        ltwh --> ltrb
        :param index:
        :return:
            bboxs: np(num_anns, 4)
            labels: np(num_anns)
        '''
        # annotation_ids = self.coco.getAnnIds(self.image_ids[index], iscrowd=False)
        annotation_ids = self.coco_obj.getAnnIds(self.ids_img[index])  # ann的id
        # anns is num_anns x 4, (x1, x2, y1, y2)
        bboxs_np = np.zeros((0, 4), dtype=np.float32)  # np创建 空数组 高级
        labels = []
        keypoints_np = []
        # skip the image without annoations
        if len(annotation_ids) == 0:
            return None

        coco_anns = self.coco_obj.loadAnns(annotation_ids)
        for a in coco_anns:
            x, y, box_w, box_h = a['bbox']  # ltwh
            # 得 ltrb
            x1 = max(0, x)  # 修正lt最小为0 左上必须在图中
            y1 = max(0, y)
            x2 = min(img_wh[0] - 1, x1 + max(0, box_w - 1))  # 右下必须在图中
            y2 = min(img_wh[1] - 1, y1 + max(0, box_h - 1))
            if a['area'] > 0 and x2 >= x1 and y2 >= y1:
                a['bbox'] = [x1, y1, x2, y2]  # 修正 并写入ltrb
            else:
                flog.warning('标记框有问题 %s 跳过', a)
                continue

            bbox = np.zeros((1, 4), dtype=np.float32)
            bbox[0, :4] = a['bbox']

            if self.mode == 'keypoints':
                '''
                # 如果关键点在物体segment内，则认为可见.
           		# v=0 表示这个关键点没有标注（这种情况下x=y=v=0）
           		# v=1 表示这个关键点标注了但是不可见(被遮挡了）
           		# v=2 表示这个关键点标注了同时也可见
                '''
                keypoints = self.handle_keypoints(a)
                if keypoints is None:
                    flog.warning('全0 keypoints %s 跳过')
                    continue
                keypoints_np.append(keypoints)

            # 全部通过后添加
            bboxs_np = np.append(bboxs_np, bbox, axis=0)
            labels.append(self.ids_old_new[a['category_id']])

        # bboxs = ltwh2ltrb(bboxs) # 前面 已转
        if bboxs_np.shape[0] == 0:
            flog.error('这图标注 不存在 %s', coco_anns)
            return None
            # raise Exception('这图标注 不存在 %s', coco_anns)

        # 转tensor
        if self.mode == 'bbox':
            return [torch.tensor(bboxs_np, dtype=torch.float), torch.tensor(labels, dtype=torch.float)]
        elif self.mode == 'keypoints':
            keypoints_np = np.array(keypoints_np)  # list转np
            # 有标注的情况下 keypoints_np 一定有
            return [torch.tensor(bboxs_np, dtype=torch.float),
                    torch.tensor(labels, dtype=torch.float),
                    torch.tensor(keypoints_np, dtype=torch.float)]

    def _init_load_classes(self):
        '''
        self.classes_ids :  {'Parade': 1}
        self.ids_classes :  {1: 'Parade'}
        self.ids_new_old {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20}
        self.ids_old_new {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20}
        :return:
        '''
        # [{'id': 1, 'name': 'aeroplane'}, {'id': 2, 'name': 'bicycle'}, {'id': 3, 'name': 'bird'}, {'id': 4, 'name': 'boat'}, {'id': 5, 'name': 'bottle'}, {'id': 6, 'name': 'bus'}, {'id': 7, 'name': 'car'}, {'id': 8, 'name': 'cat'}, {'id': 9, 'name': 'chair'}, {'id': 10, 'name': 'cow'}, {'id': 11, 'name': 'diningtable'}, {'id': 12, 'name': 'dog'}, {'id': 13, 'name': 'horse'}, {'id': 14, 'name': 'motorbike'}, {'id': 15, 'name': 'person'}, {'id': 16, 'name': 'pottedplant'}, {'id': 17, 'name': 'sheep'}, {'id': 18, 'name': 'sofa'}, {'id': 19, 'name': 'train'}, {'id': 20, 'name': 'tvmonitor'}]
        categories = self.coco_obj.loadCats(self.coco_obj.getCatIds())
        categories.sort(key=lambda x: x['id'])  # 按id升序 [{'id': 1, 'name': 'Parade'}]

        # coco ids is not from 1, and not continue ,make a new index from 0 to 79, continuely
        # 重建index 从1-80
        # classes_ids:   {names:      new_index}
        # coco_ids:  {new_index:  coco_index}
        # coco_ids_inverse: {coco_index: new_index}
        self.classes_ids, self.ids_new_old, self.ids_old_new = {}, {}, {}
        # 解决中间有断格的情况
        for c in categories:  # 修正从1开始
            self.ids_new_old[len(self.classes_ids) + 1] = c['id']
            self.ids_old_new[c['id']] = len(self.classes_ids) + 1
            self.classes_ids[c['name']] = len(self.classes_ids) + 1  # 这个是新索引 {'Parade': 0}

        # ids_classes: {new_index:  names}
        self.ids_classes = {}  # index 1 开始
        for k, v in self.classes_ids.items():
            self.ids_classes[v] = k

    def handle_keypoints(self, coco_ann):
        '''

        :param coco_ann: 这里只会是一维标签
        :return:
        '''
        k_ = np.array(coco_ann['keypoints'])

        '''
        widerface 不需要加 mask
        这个是标注是否可用的的定义 
        # 如果关键点在物体segment内，则认为可见.
        # v=0 表示这个关键点没有标注（这种情况下x=y=v=0）
        # v=1 表示这个关键点标注了但是不可见(被遮挡了）
        # v=2 表示这个关键点标注了同时也可见
        '''
        inds = np.arange(2, len(k_), 3)  # 每隔两个选出 [ 2  5  8 11 14]
        mask = np.ones(len(coco_ann['keypoints']), dtype=np.float32)
        mask[inds] = 1
        _t = k_[inds] != 2
        if _t.any() and _t.sum() != len(_t):
            # 有 且 不是全部的进入
            _s = '标签不全为2 %s' % k_
            flog.error(_s)
            # raise Exception(_s)

        inds = np.arange(2, len(k_), 3)  # 每隔两个选出
        ones = np.ones(len(coco_ann['keypoints']), dtype=np.float32)
        ones[inds] = 0
        nonzero = np.nonzero(ones)  # 取非零索引
        keypoint_np = k_[nonzero]  # 选出10个数
        if np.all(keypoint_np == 0):
            flog.warning('出现全0 keypoints %s' % coco_ann)
            return None
        return keypoint_np


def load_dataset_coco(mode, transform=None):
    # -------voc--------
    # path_root = r'M:/AI/datas/VOC2012'  # 自已的数据集
    # file_json = os.path.join(path_root, 'coco/annotations') + '/instances_type3_train.json' # 2776

    # path_root = r'M:/AI/datas/VOC2007'  # 自已的数据集
    # file_json = os.path.join(path_root, 'coco/annotations') + '/instances_type3_train_1096.json'  # 1096
    # file_json = os.path.join(path_root, 'coco/annotations') + '/instances_type4_train_753.json'  # 1096
    # path_img = os.path.join(path_root, 'train', 'JPEGImages')

    ''' 这两个都要改 '''
    # file_json = os.path.join(path_root, 'coco/annotations') + '/instances_type4_train_994.json'  # 1096
    # path_img = os.path.join(path_root, 'train', 'JPEGImages')

    # path_root = r'M:/AI/datas/widerface'  # 自已的数据集
    # file_json = os.path.join(path_root, 'coco/annotations') + '/person_keypoints_train2017.json'
    # path_img = os.path.join(path_root, 'coco/images/train2017')

    # path_root = r'E:\AI\datas\coco2017'  # 自已的数据集
    # file_json = os.path.join(path_root, 'annotations', 'person_keypoints_val2017.json')
    # path_img = os.path.join(path_root, 'val2017_5000')

    # path_root = r'M:\AI\datas\face_98'  # 自已的数据集
    # file_json = os.path.join(path_root, 'annotations', 'keypoints_test_2500_2118.json')
    # path_img = os.path.join(path_root, 'images_test_2118')

    # file_json = os.path.join(path_root, 'annotations', 'keypoints_train_7500_5316.json')
    # path_img = os.path.join(path_root, 'images_train_5316')

    path_root = r'M:\AI\datas\face_5'
    file_json = r'M:/AI/datas/face_5/annotations/keypoints_train_10000_10000.json'
    path_img = os.path.join(path_root, 'images_13466')

    dataset = CustomCocoDataset(
        file_json=file_json,
        path_img=path_img,
        mode=mode,
        transform=transform,
        # cfg=cfg,
    )
    return path_img, dataset


def t_other():
    global coco_obj, target, img_np_tensor
    coco_obj = dataset.coco_obj
    '''检测dataset'''
    dataset_ = dataset[1]
    for img, target in dataset:
        # print(img, target['boxes'], target['labels'])
        # f_plt_show_cv(img, target['boxes'])
        glabels_text = []
        for i in target['labels'].long():
            glabels_text.append(dataset.ids_classes[i.item()])

        f_show_od_ts4plt(img, target['boxes'], is_recover_size=True, glabels_text=glabels_text)
        pass

    '''打开某一个图'''
    img_id = coco_obj.getImgIds()[0]
    img_np_tensor = f_open_cocoimg(path_img, coco_obj, img_id=img_id)
    img_np_tensor.show()
    '''------------------- 获取指定类别名的id ---------------------'''
    ids_cat = coco_obj.getCatIds()
    print(ids_cat)
    infos_cat = coco_obj.loadCats(ids_cat)
    for info_cat in infos_cat:
        # {'id': 1, 'name': 'aeroplane'}
        ids = coco_obj.getImgIds(catIds=info_cat['id'])
        print('类型对应有多少个图片', info_cat['id'], info_cat['name'], len(ids))
    # ids_cat = coco.getCatIds(catNms=['aeroplane', 'bottle'])
    # ids_cat = coco.getCatIds(catIds=[1, 3])
    # ids_cat = coco_obj.getCatIds(catIds=[1])
    # print(ids_cat)
    # infos_cat = coco.loadCats(ids=[1, 5])
    # print(infos_cat)  # 详细类别信息 [{'id': 1, 'name': 'aeroplane'}, {'id': 2, 'name': 'bicycle'}]
    '''获取指定类别id的图片id'''
    ids_img = []
    for idc in ids_cat:
        ids_ = coco_obj.getImgIds(catIds=idc)
        ids_img += ids_
        # print(ids_)  # 这个只支持单个元素
    ids_img = list(set(ids_img))  # 去重
    '''查看图片信息  '''
    infos_img = coco_obj.loadImgs(ids_img[0])
    print(infos_img)  # [{'height': 281, 'width': 500, 'id': 1, 'file_name': '2007_000032.jpg'}]
    ids_ann = coco_obj.getAnnIds(imgIds=infos_img[0]['id'])
    info_ann = coco_obj.loadAnns(ids_ann)  # annotation 对象
    '''获取数据集类别数'''
    flog.debug(coco_obj.loadCats(coco_obj.getCatIds()))
    '''显示标注'''
    # f_show_coco_pics(coco_obj, path_img)


def t_keypoint():
    global data, img_np_tensor, target
    for data in dataset:
        # coco dataset可视化
        # print(data)
        # 如果有 transform 则返回的基本是统一的 tensor3d torch.Size([3, 448, 448])
        img_np_tensor, target = data  # 没有transform 则是原图 np(1385, 1024, 3) whc bgr
        # 这个是用 coco api
        # print(target)

        f_show_coco_pics(coco, path_img, ids_img=[target['image_id']])
        # f_show_od_np4plt(img_np_tensor, target['boxes'], is_recover_size=False)  # 显示原图
        # f_show_od_ts4plt(img_np_tensor, target['boxes'], is_recover_size=True)  # 需要恢复box

        print(len(target['boxes']), len(target['keypoints']))
        # keypoints 用 这个 这里全是ts
        show_bbox_keypoints4ts(img_np_tensor,
                               target['boxes'],
                               target['keypoints'],
                               torch.ones(target['boxes'].shape[0]),  # 创建一个scores
                               is_recover_size=True
                               )  # 需要恢复box
        pass


def t_bbox(dataset, transform):
    for data in dataset:
        # coco dataset可视化
        # print(data)
        # 如果有 transform 则返回的基本是统一的 tensor3d torch.Size([3, 448, 448])
        img_np_tensor, target = data  # 没有transform 则是原图 np(1385, 1024, 3) whc bgr
        # 这个是用 coco api
        # print(target)

        # 各种 coco 模式统一用这个
        f_show_coco_pics(coco, path_img, ids_img=[target['image_id']])
        # 这里输出的是np或tensor
        if transform is None:
            f_show_od_np4plt(img_np_tensor, target['boxes'], is_recover_size=False)  # 显示原图
        else:
            f_show_od_ts4plt(img_np_tensor, target['boxes'], is_recover_size=True)  # 需要恢复box
        pass


if __name__ == '__main__':
    '''
    用于测试  coco出来是 ltwh
    '''
    mode = 'bbox'  # bbox segm keypoints caption
    # mode = 'keypoints'  # bbox segm keypoints caption
    from f_tools.pic.enhance.f_data_pretreatment4np import DisposePicSet4SSD, cre_transform_resize4np


    class cfg:
        pass


    cfg.IMAGE_SIZE = (448, 448)
    cfg.PIC_MEAN = (0.406, 0.456, 0.485)
    cfg.PIC_STD = (0.225, 0.224, 0.229)
    cfg.KEEP_SIZE = True  # 用 keypoints 需要设置为True
    cfg.USE_BASE4NP = False  # 基础处理
    cfg.IS_VISUAL_PRETREATMENT = False  # 用于dataset提取时测试
    transform = cre_transform_resize4np(cfg)['train']
    # transform = SSDAugmentation(size=(448, 448))
    path_img, dataset = load_dataset_coco(mode, transform=transform)
    # path_img, dataset = load_dataset_coco(mode)
    # path_img, dataset = load_cat(mode)

    print('len(dataset)', len(dataset))
    coco = dataset.coco_obj
    # 可视化代码
    # t_keypoint(dataset,transform)
    t_bbox(dataset, transform)
    # t_other(dataset,transform)
