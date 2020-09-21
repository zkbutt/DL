import os

import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_factory import VOCDataSet, WiderfaceDataSet
import numpy as np
from collections import defaultdict, deque


def sysconfig(path_save_weight, device=None):
    '''

    :param path_save_weight:
    :return:
    '''
    if not device:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    flog.info('模型当前设备 %s', device)

    np.set_printoptions(suppress=True)  # 关闭科学计数
    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(path_save_weight):
        try:
            os.makedirs(path_save_weight)
        except Exception as e:
            flog.error(' %s %s', path_save_weight, e)
            exit(-1)
    return device


def load_data4voc(data_transform, path_data_root, batch_size=2, bbox2one=False, isdebug=False):
    '''

    :param data_transform:
    :param path_data_root:
    :param batch_size:
    :param bbox2one:  是否gt框进行归一化
    :return:
    '''
    if isdebug:
        file_name = ['train_s.txt', 'val_s.txt']
    else:
        file_name = ['train.txt', 'val.txt']
    VOC_root = os.path.join(path_data_root, 'trainval')
    # ---------------------data_set生成---------------------------
    train_data_set = VOCDataSet(
        VOC_root,
        file_name[0],  # 正式训练要改这里
        data_transform["train"],
        bbox2one=bbox2one)
    # iter(train_data_set).__next__()  # VOC2012DataSet 测试
    class_dict = train_data_set.class_dict
    flog.debug('class_dict %s', class_dict)

    # 默认通过 torch.stack() 进行拼接
    '''
    一次两张图片使用3504的显存
    '''
    train_data_loader = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # windows只能为0
        collate_fn=lambda batch: tuple(zip(*batch)),  # 输出多个时需要处理
        pin_memory=True,
    )

    val_data_set = VOCDataSet(
        VOC_root, file_name[1],
        data_transform["val"],
        bbox2one=bbox2one,
    )
    val_data_set_loader = torch.utils.data.DataLoader(
        val_data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: tuple(zip(*batch)),  # 输出多个时需要处理
        pin_memory=True,
    )
    # images, targets = iter(train_data_loader).__next__()
    # show_pic_ts(images[0], targets[0]['labels'], classes=class_dict)

    return train_data_loader, val_data_set_loader


def load_data4widerface(path_data_root, img_size_in, batch_size=2, mode='train', isdebug=False, look=False):
    '''

    :param path_data_root:
    :param img_size_in:
    :param batch_size:
    :param mode:  train val test 暂时debug只支持 train
    :param isdebug:
    :param look:
    :return:
        images: np(batch,(3,640,640))
        targets: np(batch,(x个选框,15维))  4+1+10
    '''

    def detection_collate(batch):
        '''

        :param batch: list(batch,(tuple((3,640,640),(x个选框,15维))...))
        :return:np(batch,(3,640,640))   np(batch,(x个选框,15维))
        '''
        images = []
        targets = []
        for img, box in batch:
            if len(box) == 0:
                continue
            images.append(img)
            targets.append(box)
        images = np.array(images)
        targets = np.array(targets)
        return images, targets

    train_dataset = WiderfaceDataSet(path_data_root, img_size_in, mode=mode, isdebug=isdebug, look=look)
    # iter(train_dataset).__next__()
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True,  # 不使用虚拟内存
        drop_last=True,  # 除于batch_size余下的数据
        collate_fn=detection_collate,
    )
    # 数据查看
    # iter(data_loader).__next__()
    return data_loader


def load_weight(path_weight, model, optimizer=None, lr_scheduler=None):
    start_epoch = 0
    if path_weight and os.path.exists(path_weight):
        checkpoint = torch.load(path_weight)
        model.load_state_dict(checkpoint['model'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        flog.warning('已加载 feadre 权重文件为 %s', path_weight)
    else:
        # raise Exception(' 未加载 feadre权重文件 ')
        flog.warning(' 未加载 feadre权重文件 %s', path_weight)
    return start_epoch


def save_weight(path_save, model, name, optimizer=None, lr_scheduler=None, epoch=0):
    '''

    :param path_save: 前面检查以防止后来保存不起
    :param model:
    :param name:
    :param optimizer:
    :param lr_scheduler:
    :param epoch:
    :return:
    '''
    sava_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
        'epoch': epoch}
    file_weight = os.path.join(path_save, (name + '-{}.pth').format(epoch))
    torch.save(sava_dict, file_weight)


if __name__ == '__main__':
    pass
