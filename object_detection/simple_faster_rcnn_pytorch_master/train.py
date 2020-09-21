from __future__ import absolute_import
# though cupy is not used but without this line, it raise errors...
# import cupy as cp
import os
import numpy as np
# import ipdb # 调试工具
import matplotlib
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from object_detection.simple_faster_rcnn_pytorch_master.utils.config import opt
from object_detection.simple_faster_rcnn_pytorch_master.data.dataset import Dataset, TestDataset, inverse_normalize
from object_detection.simple_faster_rcnn_pytorch_master.model.faster_rcnn_vgg16 import FasterRCNNVGG16
from object_detection.simple_faster_rcnn_pytorch_master.trainer import FasterRCNNTrainer
from object_detection.simple_faster_rcnn_pytorch_master.utils import array_tool as at
from object_detection.simple_faster_rcnn_pytorch_master.utils.vis_tool import visdom_bbox
from object_detection.simple_faster_rcnn_pytorch_master.utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
# import resource

# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')  # 在PyCharm中不显示绘图


# 评估函数
def eval(dataloader, faster_rcnn, test_num=10000):
    '''

    :param dataloader:
    :param faster_rcnn:
    :param test_num:
    :return:
    '''
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()  # 预测结果List包含，框，类别，分数。
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()  # 验证的gt框，类别，difficult疑问。
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]  # list[tensor,tensor] -> val
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])  # 输入图片以及图片的尺寸进入模型得到预测结果
        gt_bboxes += list(gt_bboxes_.numpy())  # 真实框，存放N张图片的bboxes的list,[N,(X,4)]
        gt_labels += list(gt_labels_.numpy())  # 真实类别，存放N张图片的labels的list，[N,(X,)?]
        gt_difficults += list(gt_difficults_.numpy())  # 真实difficult,存放N张图片的bboxes的难易程度，[N,(X,)]
        pred_bboxes += pred_bboxes_  # 预测框,存放N张图片的预测bboxes的list,[N,(X,4)]
        pred_labels += pred_labels_  # 预测类别,存放N张图片的预测labels的list，[N,(X,)]
        pred_scores += pred_scores_  # 预测分数,存放N张图片的预测label的分数的list,[N,(X,)]
        if ii == test_num: break  # 限制数量，预测test_num张图片

    result = eval_detection_voc(  # 用voc评估方法评估
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)  # 获得config设置信息
    np.set_printoptions(suppress=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    '''------------------数据加载及处理---------------------'''
    dataset = Dataset(opt)  # 传入opt，利用设置的数据集参数来创建训练数据集
    print('load data')
    # 用创建的训练数据集创建训练DataLoader,代码仅支持batch_size=1
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            # pin_memory=True, # 不使用虚拟内存
                            # num_workers=opt.num_workers
                            )
    testset = TestDataset(opt)  # 传入opt，利用设置的数据集参数来加载测试数据集
    # 用创建的测试数据集创建训练DataLoader,代码仅支持batch_size=1
    test_dataloader = DataLoader(testset,
                                 batch_size=1,
                                 # num_workers=opt.test_num_workers,
                                 shuffle=False,
                                 pin_memory=True
                                 )

    '''------------------模型定义---------------------'''
    faster_rcnn = FasterRCNNVGG16()  # 创建以vgg为backbone的FasterRCNN网络
    # summary(faster_rcnn, (3, 500, 500))

    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn)  # 把创建好的FasterRCNN网络放入训练器
    # trainer.to(device)

    if opt.load_path:  # 若有FasterRCNN网络的预训练加载，则加载load_path权重
        trainer.load(opt.load_path)  # 训练器加载权重
        print('load pretrained model from %s' % opt.load_path)
    # trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0  # 初始化best_map，训练时用于判断是否需要保存模型
    lr_ = opt.lr  # 得到预设的学习率
    for epoch in range(opt.epoch):  # 开始训练，训练次数为opt.epoch
        # img torch.Size([1, 3, 600, 901])
        # bbox_ torch.Size([1, 2, 4])
        # label_ tensor([[6, 8]], dtype=torch.int32)
        # scale float
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):  # 读数据时报utf8!!!
            scale = at.scalar(scale)  # 变标量 可以修改dataset直接输出标量
            # bbox是gt_box坐标(ymin, xmin, ymax, xmax)
            # label是类别的下标VOC_BBOX_LABEL_NAMES
            # img是图片，代码仅支持batch_size=1的训练

            # img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()  # 使用gpu训练
            img, bbox, label = img.float(), bbox_, label_
            trainer.train_step(img, bbox, label, scale)  # 预处理完毕，进入模型

            # if (ii + 1) % opt.plot_every == 0:  # 可视化内容，（跳过）
            #     if os.path.exists(opt.debug_file):
            #         # ipdb.set_trace()
            #         pass
            #
            #     # plot loss
            #     trainer.vis.plot_many(trainer.get_meter_data())
            #
            #     # plot groud truth bboxes
            #     ori_img_ = inverse_normalize(at.tonumpy(img[0]))
            #     gt_img = visdom_bbox(ori_img_,
            #                          at.tonumpy(bbox_[0]),
            #                          at.tonumpy(label_[0]))
            #     trainer.vis.img('gt_img', gt_img)
            #
            #     # plot predicti bboxes
            #     _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
            #     pred_img = visdom_bbox(ori_img_,
            #                            at.tonumpy(_bboxes[0]),
            #                            at.tonumpy(_labels[0]).reshape(-1),
            #                            at.tonumpy(_scores[0]))
            #     trainer.vis.img('pred_img', pred_img)
            #
            #     # rpn confusion matrix(meter)
            #     trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
            #     # roi confusion matrix
            #     trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())

        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)  # 训练一个epoch评估一次
        trainer.vis.plot('test_map', eval_result['map'])  # 可视化内容，（跳过）
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']  # 获得当前的学习率
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),  # 日志输出学习率，map，loss
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)  # 可视化内容，（跳过）

        if eval_result['map'] > best_map:  # 若这次评估的map大于之前最大的map则保存模型
            best_map = eval_result['map']  # 保存模型的map信息
            best_path = trainer.save(best_map=best_map)  # 调用保存模型函数
        if epoch == 9:  # 若训练到第9个epoch则加载之前最好的模型并且减低学习率继续训练
            trainer.load(best_path)  # 加载模型
            trainer.faster_rcnn.scale_lr(opt.lr_decay)  # 降低学习率
            lr_ = lr_ * opt.lr_decay  # 获得当前学习率

        if epoch == 13:  # 13个epoch停止训练
            break


if __name__ == '__main__':
    # import fire
    #
    # fire.Fire()
    train()
