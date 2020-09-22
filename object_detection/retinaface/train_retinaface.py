from __future__ import print_function

import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from f_tools.fits.fitting.f_fit_retinaface import train_one_epoch, evaluate
from f_tools.fun_od.f_anc import AnchorsFound
from object_detection.f_fit_tools import sysconfig, load_data4widerface, load_weight, save_weight
from object_detection.retinaface.CONFIG_RETINAFACE import PATH_SAVE_WEIGHT, PATH_DATA_ROOT, DEBUG, IMAGE_SIZE, \
    MOBILENET025, PATH_FIT_WEIGHT, NEGATIVE_RATIO, NUM_CLASSES, NEG_IOU_THRESHOLD, END_EPOCHS, \
    PRINT_FREQ, BATCH_SIZE
from object_detection.retinaface.nets.retinaface import RetinaFace
from object_detection.retinaface.nets.retinaface_training import MultiBoxLoss
from object_detection.retinaface.utils.box_utils import ltrb2xywh


class LossProcess(torch.nn.Module):
    '''
    前向和反向 loss过程函数
    '''

    def __init__(self, model, anchors, cls_loss, device):
        super(LossProcess, self).__init__()
        self.args = {
            'anchors': anchors,
            'fun_loss': cls_loss,
        }
        self.model = model
        self.device = device

    def forward(self, batch_data):
        '''
        需注意多GPU时返回值处理
        完成  数据组装完成   模型输入输出    构建展示字典及返回值
        loss_total, log_dict = self.forward(model, device, batch_data, kwargs)

          这个是loss计算过程  前向 后向
          :param batch_data: tuple(images,targets)
              images: <class 'tuple'>: (batch, 3, 640, 640)
              targets: list[batch,(23,15)]
          :return:
              loss_total: 这个用于优化
              show_dict: total_losses 这个key必须 训练的返回值
          '''
        # -----------------------输入模型前的数据处理------------------------
        images, targets = batch_data

        # with torch.no_grad():  # np -> tensor 这里不需要求导
        # tuple(np(batch,3,640,640)) -> torch.Size([8, 3, 640, 640])
        images = torch.tensor(images).type(torch.float).to(self.device)
        # list(batch,tensor(x,15(4+10+1)))
        targets = [torch.tensor(target).type(torch.float).to(self.device) for target in targets]
        # -----------------------数据组装完成------------------------

        if self.device != torch.device("cpu"):
            torch.cuda.synchronize(self.device)
        model_time = time.time()  # 开始时间

        # ---------------模型输入输出 损失计算要变 ----------------------
        # tuple(特图层个,(框,类别,关键点))
        #       框 torch.Size([batch, 16800, 4])
        #       类别 torch.Size([batch, 16800, 2])
        #       关键点 torch.Size([batch, 16800, 10])
        out = self.model(images)  # 这里要变

        if self.training:
            # 返回list(tensor)
            r_loss, c_loss, landm_loss = self.args['fun_loss'](out, self.args['anchors'], targets, device)
            loss_total = 2 * r_loss + c_loss + landm_loss  # 这个用于优化

            # -----------------构建展示字典及返回值------------------------
            # 多GPU时结果处理 reduce_dict 方法
            # losses_dict_reduced = reduce_dict(losses_dict)
            show_dict = {
                "total_losses": loss_total,
                "r_loss": r_loss,
                "c_loss": c_loss,
                "landm_loss": landm_loss,
            }
            return loss_total, show_dict
        else:
            if self.device != torch.device("cpu"):
                torch.cuda.synchronize(self.device)
            model_time = time.time() - model_time  # 结束时间

            # 当使用GPU时通过同步测试时间
            cpu_device = torch.device("cpu")
            outputs = []
            for index, (bboxes_out, labels_out, scores_out) in enumerate(out):
                info = {"boxes": bboxes_out.to(cpu_device),
                        "labels": labels_out.to(cpu_device),
                        "scores": scores_out.to(cpu_device),
                        "height_width": targets[index]["height_width"]}
                outputs.append(info)
            return out, model_time

    def to_coco(self, ds):
        ''' 80个类别
        {
            "info": { # 这个是表头
                    "description": "Example Dataset", # 数据集描述
                    "url": "https://github.com/waspinator/pycococreator", # 下载地址
                    "version": "0.1.0",
                    "year": 2018,
                    "contributor": "waspinator", # 提供者
                    "date_created": "2015-01-27 09:11:52.357475"  datetime.datetime.utcnow().isoformat(' ')
                }, # 不用
            "licenses": [
                 {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }
            ...], # 不用
            "images": [{
                    "id": int,# 图片的ID编号（每张图片ID是唯一的）
                    "width": 360,#宽
                    "height": 640,#高
                    "file_name": "COCO_val2014_000000391895.jpg",# 图片名
                    "license": 3,
                    "flickr_url": "http:\/\/farm9.staticflickr.com\/8186\/8119368305_4e622c8349_z.jpg",,# flickr网路地址
                    "coco_url": "http:\/\/mscoco.org\/images\/391895",# 网路地址路径
                    "date_captured": '2013-11-14 11:18:45' # 数据获取日期
                }
            ...],  #图片
            "annotations": [
                {
                    "id": 1768, # 图片的ID编号（每张图片ID是唯一的）
                    "image_id": 289343,  #图片序列
                    "category_id": 18, # 对应的图片ID（与images中的ID对应）
                    "segmentation": RLE or [polygon],# 对象的边界点（边界多边形，此时iscrowd=0）
                    "area": 702.1057499999998, # 区域面积宽*宽
                    "bbox": [l,t,width,height],
                    "iscrowd": 0 or 1, #是否为polygon
                },
            ...], #训练集（或者测试集）中bounding box的数量
            "categories": [
                {
                    "supercategory": "person",
                    "id": 1,# # 类对应的id （0 默认为背景）
                    "name": "person"
                },
                {
                    "supercategory": "bicycle",
                    "id": 2,
                    "name": "bicycle"
                }
            ...] #类别的数量
        }
        :return:
        '''
        dataset = {'images': [], 'categories': [], 'annotations': []}
        categories = set()
        ann_id = 1
        for img_idx in range(len(ds)):  # 遍历所有验证集
            img, targets = ds[img_idx]
            img_dict = {}
            img_dict['id'] = img_idx
            img_dict['height'] = IMAGE_SIZE[1]
            img_dict['width'] = IMAGE_SIZE[0]
            dataset['images'].append(img_dict)

            # 归一化要恢复 lrtb -> xywh
            bboxes = targets[:, :4]
            bboxes = ltrb2xywh(bboxes)
            bboxes = bboxes.tolist()
            num_objs = len(bboxes)
            for i in range(num_objs):
                ann = {}
                ann['image_id'] = img_idx  # 图片id
                ann['bbox'] = bboxes[i]
                ann['category_id'] = 1  # 只有一个类别
                categories.add(1)
                ann['area'] = bboxes[i][2] * IMAGE_SIZE[0] * bboxes[i][3] * IMAGE_SIZE[1]
                ann['iscrowd'] = 0
                ann['id'] = i
                dataset['annotations'].append(ann)
                ann_id += 1
        dataset['categories'] = [{'id': i} for i in sorted(categories)]
        return dataset


if __name__ == "__main__":
    '''------------------系统配置---------------------'''
    claxx = MOBILENET025  # 这里根据实际情况改
    device = sysconfig(PATH_SAVE_WEIGHT)
    if DEBUG:
        device = torch.device("cpu")

    '''---------------数据加载及处理--------------'''
    # 返回数据已预处理 返回np(batch,(3,640,640))  , np(batch,(x个选框,15维))
    data_loader = load_data4widerface(PATH_DATA_ROOT, IMAGE_SIZE, BATCH_SIZE, isdebug=True)
    # iter(data_loader).__next__()

    '''------------------模型定义---------------------'''
    model = RetinaFace(claxx.MODEL_NAME,
                       claxx.PATH_MODEL_WEIGHT,
                       claxx.IN_CHANNELS, claxx.OUT_CHANNEL,
                       claxx.RETURN_LAYERS)
    # if torchvision._is_tracing() 判断训练模式
    # self.training 判断训练模式
    model.train()  # 启用 BatchNormalization 和 Dropout
    model.to(device)

    cls_loss = MultiBoxLoss(NUM_CLASSES, NEGATIVE_RATIO, NEG_IOU_THRESHOLD)
    # 权重衰减(如L2惩罚)(默认: 0)
    optimizer = optim.Adam(model.parameters(), 1e-3, weight_decay=5e-4)
    # 在发现loss不再降低或者acc不再提高之后，降低学习率
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    # feadre权重加载
    start_epoch = load_weight(PATH_FIT_WEIGHT, model, optimizer, lr_scheduler)

    '''------------------模型训练---------------------'''
    # 返回特图 h*w,4 anchors 是每个特图的长宽个 4维整框 这里是 x,y,w,h 调整系数
    anchors = AnchorsFound(IMAGE_SIZE, claxx.ANCHORS_SIZE, claxx.FEATURE_MAP_STEPS, claxx.ANCHORS_CLIP).get_anchors()
    anchors.to(device)

    train_loss = []
    learning_rate = []
    val_map = []

    for epoch in range(start_epoch, END_EPOCHS):
        if epoch < 5:
            model.train()

            # 主干网一般要冻结
            for param in model.body.parameters():
                param.requires_grad = False

            process = LossProcess(model, anchors, cls_loss, device)
            loss = train_one_epoch(data_loader, process, optimizer, epoch,
                                   PRINT_FREQ, train_loss=train_loss, train_lr=learning_rate,
                                   )

            lr_scheduler.step(loss)  # 更新学习

            '''------------------模型验证---------------------'''
            model.eval()

            coco_text = evaluate(data_loader, process, PRINT_FREQ, mAP_list=val_map)

            # 每个epoch保存
            save_weight(PATH_SAVE_WEIGHT,
                        model,
                        os.path.basename(__file__),
                        optimizer,
                        lr_scheduler,
                        epoch)

    for epoch in range(epoch_start, epoch_end):
        loss = fit_one_epoch(model, cls_loss, epoch, data_loader, epoch_end, anchors, device)
        lr_scheduler.step(loss)

    if True:
        # --------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        # --------------------------------------------#
        lr = 1e-4
        batch_size = 4
        epoch_end = 25
        Unfreeze_Epoch = 50

        optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
        train_dataset = DataGenerator(training_dataset_path, img_dim)
        data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=detection_collate)
        epoch_size = train_dataset.get_len() // batch_size
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        for param in model.body.parameters():
            param.requires_grad = True

        for epoch in range(epoch_end, Unfreeze_Epoch):
            loss = fit_one_epoch(net, cls_loss, epoch, epoch_size, data_loader, Unfreeze_Epoch, anchors, cfg, Cuda)
            lr_scheduler.step(loss)
