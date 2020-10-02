import os

import torch
import torch.optim as optim

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_factory import Data_Prefetcher
from f_tools.datas.data_pretreatment import Compose, RandomHorizontalFlip, ToTensor, Resize
from f_tools.fits.f_lossfun import KeypointsLoss, PredictOutput
from f_tools.fits.f_show import plot_loss_and_lr
from f_tools.fits.fitting.f_fit_retinaface import f_train_one_epoch, f_evaluate
from f_tools.fun_od.f_anc import AnchorsFound
from f_tools.fun_od.f_boxes import pos_match
from object_detection.coco_t.coco_api import coco_eval
from object_detection.coco_t.coco_dataset import CocoDataset
from object_detection.f_fit_tools import sysconfig, load_weight, save_weight
from object_detection.retinaface.CONFIG_RETINAFACE import PATH_SAVE_WEIGHT, PATH_DATA_ROOT, DEBUG, IMAGE_SIZE, \
    MOBILENET025, PATH_FIT_WEIGHT, NEGATIVE_RATIO, NEG_IOU_THRESHOLD, END_EPOCHS, \
    PRINT_FREQ, BATCH_SIZE, VARIANCE, LOSS_COEFFICIENT, RESNET50
from object_detection.retinaface.nets.retinaface import RetinaFace


class LossProcess(torch.nn.Module):
    '''
    前向和反向 loss过程函数
    '''

    def __init__(self, model, anchors, losser):
        super(LossProcess, self).__init__()
        self.anchors = anchors
        self.model = model
        self.losser = losser

    def forward(self, batch_data):
        '''

        :param batch_data: tuple(images,targets)
            images:tensor(batch,c,h,w)
            list( # batch个
                target: dict{
                        image_id: int,
                        bboxs: np(num_anns, 4),
                        labels: np(num_anns),
                        keypoints: np(num_anns,10),
                    }
                )
        :return:
            loss_total: 这个用于优化
            show_dict:  log_dict
        '''
        # -----------------------输入模型前的数据处理 开始------------------------
        images, targets = batch_data
        # -----------------------输入模型前的数据处理 完成------------------------

        '''
           模型输出 tuple(# 预测器 框4 类别2 关键点10
               torch.Size([batch, 16800, 4]) # 框
               torch.Size([batch, 16800, 2]) # 类别 只有一个类是没有第三维
               torch.Size([batch, 16800, 10]) # 关键点
           )
        '''
        out = self.model(images)
        bboxs_p = out[0]
        labels_p = out[1]
        keypoints_p = out[2]

        '''---------------------与输出进行维度匹配及类别匹配-------------------------'''
        num_batch = images.shape[0]
        num_ancs = self.anchors.shape[0]

        gbboxs = torch.Tensor(num_batch, num_ancs, 4).to(images)  # torch.Size([5, 16800, 4])
        glabels = torch.Tensor(num_batch, num_ancs).to(images)  # 这个只会存在一维 无论多少类
        gkeypoints = torch.Tensor(num_batch, num_ancs, 10).to(images)
        '''---------------------与输出进行维度匹配及类别匹配-------------------------'''
        for index in range(num_batch):
            bboxs = targets[index]['bboxs']  # torch.Size([40, 4])
            labels = targets[index]['labels']  # torch.Size([40,2])

            id_ = int(targets[index]['image_id'].item())
            info = dataset_train.coco.loadImgs(id_)[0] # 查看图片信息
            flog.debug(info)

            keypoints = targets[index]['keypoints']  # torch.Size([40, 10])
            '''
            masks: 正例的 index 布尔
            bboxs_ids : 正例对应的bbox的index
            '''
            label_neg_mask, anc_bbox_ind = pos_match(self.anchors, bboxs, NEG_IOU_THRESHOLD)

            # new_anchors = anchors.clone().detach()
            # 将bbox取出替换anc对应位置 ,根据 bboxs 索引list 将bboxs取出与anc 形成维度对齐 便于后面与anc修复算最终偏差 ->[anc个,4]
            # 只计算正样本的定位损失,将正例对应到bbox标签 用于后续计算anc与bbox的差距
            match_bboxs = bboxs[anc_bbox_ind]
            match_keypoints = keypoints[anc_bbox_ind]

            # 构建正反例label 使原有label保持不变
            # labels = torch.zeros(num_ancs, dtype=torch.int64)  # 标签默认为同维 类别0为负样本
            # 正例保持原样 反例置0
            match_labels = labels[anc_bbox_ind]
            # match_labels[label_neg_mask] = torch.tensor(0).to(labels)
            match_labels[label_neg_mask] = 0.  # 这里没有to设备
            # labels[label_neg_ind] = 0 #  这是错误的用法
            glabels[index] = match_labels

            gbboxs[index] = match_bboxs
            gkeypoints[index] = match_keypoints
        '''---------------------与输出进行维度匹配及类别匹配  完成-------------------------'''

        # ---------------损失计算 ----------------------
        # log_dict用于显示
        loss_total, log_dict = self.losser(bboxs_p, gbboxs, labels_p, glabels, keypoints_p, gkeypoints)

        # -----------------构建展示字典及返回值------------------------
        # 多GPU时结果处理 reduce_dict 方法
        # losses_dict_reduced = reduce_dict(losses_dict)
        show_dict = {"loss_total": loss_total.detach().item(), }
        show_dict.update(**log_dict)
        return loss_total, show_dict


class ForecastProcess(PredictOutput):

    def __init__(self, model, ancs, img_size, coco, variance=(0.1, 0.2), iou_threshold=0.5, max_output=100,
                 mode='bboxs'):
        '''

        :param model:
        :param ancs:
        :param img_size:
        :param coco:
        :param variance:
        :param iou_threshold:
        :param max_output:
        :param mode: bboxs   keypoints
        '''
        super().__init__(ancs, img_size, variance, iou_threshold, max_output)
        self.model = model
        self.coco = coco
        self.mode = mode

    def forward(self, batch_data, epoch):
        # -----------------------输入模型前的数据处理 开始------------------------
        images, targets = batch_data
        # -----------------------输入模型前的数据处理 完成------------------------

        '''
           模型输出 tuple(# 预测器 框4 类别2 关键点10
               torch.Size([batch, 16800, 4]) # 框
               torch.Size([batch, 16800]) # 类别 只有一个类是没有第三维
               torch.Size([batch, 16800, 10]) # 关键点
           )
        '''
        out = self.model(images)
        bboxs_p = out[0]
        labels_p = out[1]
        keypoints_p = out[2]
        # imgs_rets 每一个张的最终输出 ltrb  list([bboxes_out, scores_out, labels_out, other_in] * batch个 )
        imgs_rets = super().forward(bboxs_p, labels_p, keypoints_p, mode='ltwh')
        # 组装结果json
        coco_json4bbox = []
        coco_json4keypoints = []
        '''结果要求ltwh'''

        for i, ret in enumerate(imgs_rets):  # 这里遍历每一张图片
            image_id = int(targets[i]['image_id'].cpu().item())

            w, h = self.img_size  # 使用模型进入尺寸

            # np高级
            bboxs, scores, labels = ret[0], ret[1], ret[2]
            img_size = torch.tensor((w, h)).to(bboxs)
            bboxs = bboxs * img_size[None].repeat(1, 2)

            if self.mode == 'bboxs':
                for bbox, score, label, keypoint in zip(bboxs, scores, labels):
                    _t_bbox = {}
                    _t_bbox['image_id'] = image_id
                    _t_bbox['category_id'] = label.cpu().item()
                    _t_bbox['bbox'] = list(bbox.cpu().numpy())
                    _t_bbox['score'] = score.cpu().item()
                    coco_json4bbox.append(_t_bbox)

                coco_eval(coco_json4bbox, self.coco, epoch, 'bbox')

            elif self.mode == 'keypoints':
                keypoints = ret[3]
                keypoints = keypoints * img_size[None].repeat(1, 5)
                coco_keypoints = torch.zeros(
                    (int(keypoints.shape[0]), int(keypoints.shape[-1] / 2 + keypoints.shape[-1]))).to(keypoints)
                coco_keypoints[:, 2::3] = 2
                axis1, axis2 = torch.where(coco_keypoints != 2)
                coco_keypoints[:, torch.unique(axis2)] = keypoints

                # tensor也可以直接遍历
                for bbox, score, label, keypoint in zip(bboxs, scores, labels, coco_keypoints):
                    _t_bbox = {}
                    _t_bbox['image_id'] = image_id
                    _t_bbox['category_id'] = label.cpu().item()
                    _t_bbox['bbox'] = list(bbox.cpu().numpy())
                    _t_bbox['score'] = score.cpu().item()
                    coco_json4bbox.append(_t_bbox)
                    _t_keypoints = {}
                    _t_keypoints['image_id'] = image_id
                    _t_keypoints['category_id'] = label.cpu().item()
                    _t_keypoints['keypoints'] = list(keypoint.cpu().numpy())
                    _t_keypoints['score'] = score.cpu().item()
                    coco_json4keypoints.append(_t_keypoints)

                coco_eval(coco_json4bbox, self.coco, epoch, 'bbox')
                coco_eval(coco_json4keypoints, self.coco, epoch, 'keypoints')


if __name__ == "__main__":
    '''
    执行train.py报错RuntimeError: cannot perform reduction function max on tensor with no elements because the operation does not have an identity
    '''

    '''------------------系统配置---------------------'''
    # claxx = RESNET50  # 这里根据实际情况改
    claxx = MOBILENET025  # 这里根据实际情况改
    device = sysconfig(PATH_SAVE_WEIGHT)
    if DEBUG:
        device = torch.device("cpu")

    '''---------------数据加载及处理--------------'''
    # 返回数据已预处理 返回np(batch,(3,640,640))  , np(batch,(x个选框,15维))
    data_transform = {
        "train": Compose([
            Resize(IMAGE_SIZE),  # (h,w)
            RandomHorizontalFlip(1),
            ToTensor(),
        ]),
        "val": Compose([ToTensor()])
    }

    dataset_train = CocoDataset(PATH_DATA_ROOT, 'keypoints', 'train2017', device, data_transform['train'])
    dataset_val = CocoDataset(PATH_DATA_ROOT, 'bboxs', 'val2017', device, data_transform['train'])

    '''
      归一化后 toTensor
          image_mean = [0.485, 0.456, 0.406]
          image_std = [0.229, 0.224, 0.225]
    '''


    def collate_fn(batch_datas):
        '''
        loader输出数据组装
        :param batch_datas:
            list(
                batch(
                    img: (h,w,c),
                    target: dict{
                        image_id: int,
                        bboxs: tensor(num_anns, 4),
                        labels: tensor(num_anns,类别数),
                        keypoints: tensor(num_anns,10),
                    }
                )
            )
        :return:
            images:tensor(batch,c,h,w)
            list( # batch个
                target: dict{
                        image_id: int,
                        bboxs: tensor(num_anns, 4),
                        labels: tensor(num_anns,类别数),
                        keypoints: tensor(num_anns,10),
                    }
                )
        '''
        _t = batch_datas[0][0]
        # images = torch.empty((len(batch_datas), *_t.shape), device=_t.device)
        images = torch.empty((len(batch_datas), *_t.shape)).to(_t)
        targets = []
        for i, (img, taget) in enumerate(batch_datas):
            images[i] = img
            targets.append(taget)
        return images, targets


    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        num_workers=0,
        # shuffle=True,
        # pin_memory=True,  # 不使用虚拟内存 GPU要报错
        drop_last=True,  # 除于batch_size余下的数据
        collate_fn=collate_fn,
    )

    # loader_train = Data_Prefetcher(loader_train)

    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE,
        num_workers=0,
        # shuffle=True,
        # pin_memory=True,  # 不使用虚拟内存 GPU要报错
        drop_last=True,  # 除于batch_size余下的数据
        collate_fn=collate_fn,
    )

    # iter(data_loader).__next__()

    '''------------------模型定义---------------------'''
    num_classes = len(dataset_train.coco.getCatIds())  # 根据数据集取类别数
    model = RetinaFace(claxx.MODEL_NAME,
                       claxx.PATH_MODEL_WEIGHT,
                       claxx.IN_CHANNELS, claxx.OUT_CHANNEL,
                       claxx.RETURN_LAYERS, claxx.ANCHOR_NUM,
                       num_classes)
    # if torchvision._is_tracing() 判断训练模式
    # self.training 判断训练模式
    model.train()  # 启用 BatchNormalization 和 Dropout
    model.to(device)  # 模型装入显存

    # 生成正方形anc
    anchors = AnchorsFound(IMAGE_SIZE, claxx.ANCHORS_SIZE, claxx.FEATURE_MAP_STEPS, claxx.ANCHORS_CLIP).get_anchors()
    anchors = anchors.to(device)

    losser = KeypointsLoss(anchors, NEGATIVE_RATIO, VARIANCE, LOSS_COEFFICIENT)
    # 权重衰减(如L2惩罚)(默认: 0)
    optimizer = optim.Adam(model.parameters(), 1e-3, weight_decay=5e-4)
    # 在发现loss不再降低或者acc不再提高之后，降低学习率
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    # feadre权重加载
    start_epoch = load_weight(PATH_FIT_WEIGHT, model, optimizer, lr_scheduler)

    '''------------------模型训练---------------------'''

    # 返回特图 h*w,4 anchors 是每个特图的长宽个 4维整框 这里是 x,y,w,h 调整系数
    # 特图的步距

    train_loss = []
    learning_rate = []
    val_map = []

    for epoch in range(start_epoch, END_EPOCHS):
        # if epoch < 5:
        #     # 主干网一般要冻结
        #     for param in model.body.parameters():
        #         param.requires_grad = False
        # else:
        #     # 解冻后训练
        #     for param in model.body.parameters():
        #         param.requires_grad = True

        process = LossProcess(model, anchors, losser)

        flog.info('训练开始 %s', epoch)
        loss = f_train_one_epoch(loader_train, process, optimizer, epoch,
                                 PRINT_FREQ,
                                 ret_train_loss=train_loss, ret_train_lr=learning_rate,
                                 )

        lr_scheduler.step(loss)  # 更新学习

        '''------------------模型验证---------------------'''
        forecast_process = ForecastProcess(model, anchors, IMAGE_SIZE, dataset_val.coco, 'bboxs')
        f_evaluate(loader_val, forecast_process, epoch, PRINT_FREQ)

        # 每个epoch保存
        save_weight(PATH_SAVE_WEIGHT,
                    model,
                    os.path.basename(__file__),
                    optimizer,
                    lr_scheduler,
                    epoch)

    '''-------------结果可视化-----------------'''
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate)
    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
