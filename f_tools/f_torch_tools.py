import os
import tempfile
from collections import OrderedDict

import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_gpu.f_gpu_api import fis_mgpu, is_main_process


def load_weight(file_weight, model, optimizer=None, lr_scheduler=None,
                device=torch.device('cpu'), is_mgpu=False, ffun=None):
    start_epoch = 0

    # 只匹配需要的
    # model_dict = model.state_dict() # 获取模型每层的参数阵
    # checkpoint = torch.load(file_weight, map_location=device)
    # pretrained_dict = checkpoint['model']
    # dd = {}
    # for k, v in pretrained_dict.items():
    #     if k in model_dict:
    #         shape_1 = model_dict[k].shape
    #         shape_2 = pretrained_dict[k].shape
    #         if shape_1 == shape_2:
    #             dd[k] = v
    #         else:
    #             print('shape mismatch in %s. shape_1=%s, while shape_2=%s.' % (k, shape_1, shape_2))
    # model_dict.update(dd)
    # model.load_state_dict(model_dict)
    # flog.warning('手动加载完成:%s',file_weight)
    # return start_epoch

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

        ''' debug '''
        if ffun is not None:
            pretrained_dict = ffun(pretrained_dict)

        dd = {}

        # 多GPU处理
        ss = 'module.'
        for k, v in pretrained_dict.items():
            if is_mgpu:
                if ss not in k:
                    dd[ss + k] = v
                else:
                    dd = pretrained_dict
                    break
                    # dd[k] = v
            else:
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
        if fis_mgpu():
            path = os.path.join(tempfile.gettempdir(), "_init_weights_tmp.pt")
            checkpoint_path = path
            if is_main_process():
                torch.save(model.state_dict(), checkpoint_path)

            # import torch.distributed as dist
            torch.distributed.barrier()  # 多GPU阻塞
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            flog.error('已默认加载临时文件 %s', path)
    return start_epoch


def save_weight(path_save, model, name, loss=None, optimizer=None, lr_scheduler=None, epoch=0, maps_val=None):
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
        if maps_val is not None:
            if loss is not None:
                l = round(loss, 2)
            else:
                l = ''
            file_weight = os.path.join(path_save, (name + '-{}_{}_{}_{}.pth')
                                       .format(epoch + 1,
                                               l,
                                               'p' + str(round(maps_val[0] * 100, 1)),
                                               'r' + str(round(maps_val[1] * 100, 1)),
                                               ))
        else:
            file_weight = os.path.join(path_save, (name + '-{}_{}.pth').format(epoch + 1, round(loss, 3)))
        torch.save(sava_dict, file_weight)
        flog.info('保存成功 %s', file_weight)
