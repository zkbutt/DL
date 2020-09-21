from __future__ import print_function
import os
import time
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader

from f_tools.GLOBAL_LOG import flog
from object_detection.f_fit_tools import sysconfig
from object_detection.retinaface.CONFIG_RETINAFACE import PATH_SAVE_WEIGHT, PATH_DATA_ROOT, IMG_SIZE_IN, BATCH_SIZE
from object_detection.retinaface.nets.retinaface import RetinaFace
from object_detection.retinaface.nets.retinaface_training import DataGenerator, MultiBoxLoss, detection_collate
from object_detection.retinaface.utils.anchors import Anchors


def get_lr(optimizer):  # 获取优化器lr
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(model, cls_loss, epoch, data_loader, epoch_end, anchors, device):
    total_r_loss = 0
    total_c_loss = 0
    total_landmark_loss = 0

    start_time = time.time()
    with tqdm(total=len(data_loader), desc=f'Epoch {epoch + 1}/{epoch_end}', postfix=dict, mininterval=0.3) as pbar:
        for i, (images, targets) in enumerate(data_loader):
            # if i >= epoch_size:
            #     break
            with torch.no_grad():  # np -> tensor
                # torch.Size([8, 3, 640, 640])
                images = Variable(torch.from_numpy(images).type(torch.float)).to(device)
                # n,15(4+10+1)
                targets = [Variable(torch.from_numpy(target).type(torch.float)).to(device) for target in targets]
            optimizer.zero_grad()
            # forward
            out = model(images)  # tuple(torch.Size([8, 16800, 4]),torch.Size([8, 16800, 2]),torch.Size([8, 16800, 10]))
            r_loss, c_loss, landm_loss = cls_loss(out, anchors, targets, device)
            loss = 2 * r_loss + c_loss + landm_loss  # r_loss放大2倍

            loss.backward()
            optimizer.step()

            total_c_loss += c_loss.item()
            total_r_loss += r_loss.item()
            total_landmark_loss += landm_loss.item()
            waste_time = time.time() - start_time

            pbar.set_postfix(**{'Conf Loss': total_c_loss / (i + 1),
                                'Regression Loss': total_r_loss / (i + 1),
                                'LandMark Loss': total_landmark_loss / (i + 1),
                                'lr': get_lr(optimizer),
                                's/step': waste_time})
            pbar.update(1)
            start_time = time.time()

    flog.info('Saving state, iter:', str(epoch + 1))
    sava_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch}
    torch.save(sava_dict, os.path.join(PATH_MODEL_WEIGHT, 'retinaface{}.pth'.format(epoch + 1)))
    return (total_c_loss + total_r_loss + total_landmark_loss) / (len(data_loader))


if __name__ == "__main__":
    '''------------------系统配置---------------------'''
    # training_dataset_path = './data/widerface/train/label.txt'
    Use_Data_Loader = True
    device = sysconfig(PATH_SAVE_WEIGHT)
    device = torch.device("cpu")

    '''---------------数据加载及处理--------------'''
    train_dataset = DataGenerator(PATH_DATA_ROOT, IMG_SIZE_IN, test=True)
    # iter(train_dataset).__next__()
    data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0,
                             shuffle=True,
                             pin_memory=True,  # 不使用虚拟内存
                             drop_last=True,  # 除于batch_size余下的数据
                             collate_fn=detection_collate)
    # for iteration, batch in enumerate(gen, 1):
    #     print(iteration)
    #     pass
    # 数据查看
    # iter(gen).__next__()
    # prefetcher = DataPrefetcher(gen)
    # data, label = prefetcher.__next__()

    '''------------------模型定义---------------------'''
    model = RetinaFace(IN_CHANNELS_FPN, OUT_CHANNEL, RETURN_LAYERS)
    # if torchvision._is_tracing() 判断训练模式
    # self.training 判断训练模式
    model.train()  # 启用 BatchNormalization 和 Dropout
    model.to(device)

    # 模型分析
    # print(model)
    # import tensorwatch as tw
    # img = tw.draw_model(model, [1, 3, 1024, 1024])
    # img.save(r'pic_model.jpg')
    # # 查看网络的统计结果
    # args = tw.model_stats(model, [1, 3, 640, 640])
    # print(args)

    lr = 1e-3
    batch_size = 8
    epoch_start = 0
    epoch_end = 25

    cls_loss = MultiBoxLoss(num_classes, NEGATIVE_RATIO)
    # 权重衰减(如L2惩罚)(默认: 0)
    optimizer = optim.Adam(model.parameters(), LR, weight_decay=5e-4)
    # 在发现loss不再降低或者acc不再提高之后，降低学习率
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    if PATH_MODEL_WEIGHT and os.path.exists(PATH_MODEL_WEIGHT):
        flog.info('Loading weights into state dict...')
        checkpoint = torch.load(PATH_MODEL_WEIGHT)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        flog.warning('已加载权重文件为 %s', PATH_MODEL_WEIGHT)

    '''------------------模型训练---------------------'''
    anchors = Anchors(IMAGE_SIZE, ANCHORS_SIZE, FEATURE_MAP_STEPS, ANCHORS_CLIP).get_anchors()
    anchors.to(device)

    #   冻结一定部分训练
    # for param in model.body.parameters():
    #     param.requires_grad = False

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
