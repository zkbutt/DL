'''解决linux导入出错'''
import os
import sys

from f_tools.fits.f_gpu.f_gpu_api import mgpu_init

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from torch.utils.tensorboard import SummaryWriter

from object_detection.z_retinaface.CONFIG_RETINAFACE import CFG
from object_detection.z_retinaface.process_fun import init_model, data_loader, train_eval

'''解决linux导入出错 完成'''
import torch
from f_tools.GLOBAL_LOG import flog
import numpy as np

'''
\home\feadre\.conda\pkgs\pytorch-1.6.0-py3.7_cuda10.2.89_cudnn7.6.5_0\lib\python3.7\site-packages\torch\distributed\launch.py
--nproc_per_node=2 /AI/temp/tmp_pycharm/DL/object_detection/z_retinaface/train_retinaface_DDP.py
python -m torch.distributed.launch --nproc_per_node=2 /AI/temp/tmp_pycharm/DL/object_detection/z_retinaface/train_retinaface_DDP.py

单GPU B48 640 F1 P400 time: 1.6707  0:07:39 (1.7069 s / it) mem: 3754 d121 # 锁定
双GPU B48 640 F1 P400 time: 1.8350  0:04:18 (1.9258 s / it) mem: 3769 d121 # 锁定
双GPU B8  640 F2 P400 time: 0.7893  0:10:41 (0.7975 s / it) mem: 4989 d121 


单GPU B64 640 F2 P400 time: 2.0831  mem: 4254 mv2 # 锁定
单GPU B60 640 F2 P400 time: 1.9477  mem: 4069 mv2 # 锁定
双GPU B60 640 F2 P400 time: 2.1121  0:03:59 (2.2374 s / it) mem: 3611 mv2 # 锁定
双GPU B60 640 F2 P400 time: 2.1121  0:03:59 (2.2374 s / it) mem: 3611 mv2 # 锁定
双GPU B22  640 F2 P400 time: 1.7534 0:09:00 (1.8503 s / it) mem: 6849 mv2
验 B22 147/147 [01:05<00:00,  2.24it/s]

训练 id缺失: 9227 3808 279 7512
验证 id缺失: 29  1828 2501 3086

train_retinaface_mobilenet_v2-88_2.6454436779022217.pth
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.214
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.384
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.217
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.107
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.452
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.559
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.051
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.170
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.236
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.120
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.498
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.614

'''

if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise EnvironmentError("未发现GPU")

    cfg = CFG
    cfg.DATA_NUM_WORKERS = 6
    torch.multiprocessing.set_sharing_strategy('file_system')  # 多进程开文件
    if cfg.DEBUG or cfg.IS_FMAP_EVAL:
        raise Exception('调试 和 IS_FMAP_EVAL 模式无法使用')

    args, device = mgpu_init()

    tb_writer = None
    if args.rank == 0:
        # 主进程任务
        flog.info(args)
        if not os.path.exists(CFG.PATH_SAVE_WEIGHT):
            try:
                os.makedirs(CFG.PATH_SAVE_WEIGHT)
            except Exception as e:
                flog.error(' %s %s', CFG.PATH_SAVE_WEIGHT, e)
        # tensorboard --logdir=runs --host=192.168.0.199
        tb_writer = SummaryWriter()

    '''------------------模型定义---------------------'''
    model, optimizer, lr_scheduler, start_epoch = init_model(cfg, device, id_gpu=args)

    '''---------------数据加载及处理--------------'''
    loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler = data_loader(model.cfg, is_mgpu=True)
    # loader_train, loader_val, train_sampler = data_loader4mgpu(CFG)

    flog.debug('---训练开始---epoch %s', start_epoch + 1)
    train_eval(start_epoch=start_epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
               loader_train=loader_train, loader_val_fmap=loader_val_fmap, loader_val_coco=loader_val_coco,
               device=device, train_sampler=train_sampler, eval_sampler=eval_sampler, tb_writer=tb_writer,
               )

    # torch.distributed.destroy_process_group()  # 释放进程

    flog.info('---%s--main执行完成--进程号：%s---- ' % (os.path.basename(__file__), torch.distributed.get_rank()))
