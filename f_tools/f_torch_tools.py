import os
from collections import OrderedDict

import torch

from f_tools.GLOBAL_LOG import flog


def load_weight(file_weight, model, optimizer=None, lr_scheduler=None, device=torch.device('cpu'), is_mgpu=False):
    start_epoch = 0

    # model_dict = model.state_dict() # 获取模型每层的参数阵
    # if True:
    #     model_dict = model.state_dict()
    #     pretrained_dict = torch.load(file_weight, map_location=device)
    #     dd = {}
    #     for k, v in pretrained_dict.items():
    #         if model_dict[k].shape == v.shape:
    #             dd[k] = v
    #     model_dict.update(dd)
    #     model.load_state_dict(model_dict)
    #     flog.warning('手动加载完成:%s',file_weight)
    #     return start_epoch

    if file_weight and os.path.exists(file_weight):
        checkpoint = torch.load(file_weight, map_location=device)

        '''对多gpu的k进行修复'''
        pretrained_dict = checkpoint['model']
        # 特殊处理
        # if True:
        #     del pretrained_dict['module.head_yolov1.weight']
        #     del pretrained_dict['module.head_yolov1.bias']
        #     # del pretrained_dict['module.ClassHead.1.conv1x1.weight']
        #     # del pretrained_dict['module.ClassHead.1.conv1x1.bias']
        #     # del pretrained_dict['module.ClassHead.2.conv1x1.weight']
        #     # del pretrained_dict['module.ClassHead.2.conv1x1.bias']
        #     model.load_state_dict(pretrained_dict, strict=False)
        #     start_epoch = checkpoint['epoch'] + 1
        #     flog.error('已特殊加载 feadre 权重文件为 %s', file_weight)
        #     return start_epoch

        dd = {}
        ss = 'module.'
        if is_mgpu:
            for k, v in pretrained_dict.items():
                if ss not in k:
                    dd[ss + k] = v
                else:
                    dd = pretrained_dict
                    break
                    # dd[k] = v
        else:
            for k, v in pretrained_dict.items():
                dd[k.replace(ss, '')] = v
        '''重组权重'''
        # load_weights_dict = {k: v for k, v in weights_dict.items()
        #                      if model.state_dict()[k].numel() == v.numel()}

        keys_missing, keys_unexpected = model.load_state_dict(dd, strict=False)
        if len(keys_missing) > 0 or len(keys_unexpected):
            flog.error('missing_keys %s', keys_missing)
            flog.error('unexpected_keys %s', keys_unexpected)
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler and checkpoint['lr_scheduler']:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        flog.warning('已加载 feadre 权重文件为 %s', file_weight)
    else:
        # raise Exception(' 未加载 feadre权重文件 ')
        flog.error(' 未加载 feadre权重文件 %s', file_weight)
    return start_epoch


def save_weight(path_save, model, name, loss=None, optimizer=None, lr_scheduler=None, epoch=0):
    '''

    :param path_save:
    :param model:
    :param name:
    :param loss: loss值
    :param optimizer:
    :param lr_scheduler:
    :param epoch:
    :return:
    '''
    if path_save and os.path.exists(path_save):
        sava_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
            'epoch': epoch}
        file_weight = os.path.join(path_save, (name + '-{}_{}.pth').format(epoch + 1, loss))
        torch.save(sava_dict, file_weight)
        flog.info('保存成功 %s', file_weight)
