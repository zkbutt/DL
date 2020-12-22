'''解决linux导入出错'''
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from f_tools.fits.f_fit_fun import custom_set
from object_detection.z_yolov3.CONFIG_YOLO3 import CFG
from object_detection.z_yolov3.process_fun import init_model, data_loader, train_eval
from f_tools.fits.f_gpu.f_gpu_api import mgpu_init

from torch.utils.tensorboard import SummaryWriter

'''解决linux导入出错 完成'''
import torch
from f_tools.GLOBAL_LOG import flog

'''
\home\feadre\.conda\pkgs\pytorch-1.6.0-py3.7_cuda10.2.89_cudnn7.6.5_0\lib\python3.7\site-packages\torch\distributed\launch.py
pycharm用这个 --nproc_per_node=2 /AI/temp/tmp_pycharm/DL/object_detection/z_yolov3/train_yolov3_DDP.py
linux用这个   
python -m torch.distributed.launch --nproc_per_node=2 /AI/temp/tmp_pycharm/DL/object_detection/z_yolov3/train_yolov3_DDP.py

双GPU B128 416 F1 P400 time: 7.5721  data: 0.0006  0:08:31 (7.7556 s / it) mem: 6241 mv2 # 锁定
双GPU B128 416 F1 P400 time: 7.5721  data: 0.0006  0:08:31 (7.7556 s / it) mem: 6241 mv2 # 锁定 IS_MOSAIC
0:07:04 (6.4365 s / it)

双GPU B40  416 F1 P400 time: 2.4378  0:02:16 (2.5708 s / it) mem: 6076 mv2 # MOSAIC
双GPU B36  416 F1 P400 time: 2.1986  0:08:53 (2.2531 s / it) mem: 6076 mv2 
双GPU B16  640 F4 P400 time: 1.7682  0:04:00 (1.8049 s / it) mem: 5780 mv2 # MOSAIC
 
train_yolov3_mobilenet_v2-70_3.059.pth
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.110
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.350
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.025
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.002
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.095
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.124
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.157
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.208
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.208
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.028
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.179
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.222

'''

if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise EnvironmentError("未发现GPU")

    cfg = CFG
    if cfg.IS_MOSAIC:
        # cfg.IMAGE_SIZE = [640, 640]
        # cfg.BATCH_SIZE = 16
        pass

    custom_set(cfg)

    cfg.DATA_NUM_WORKERS = 6
    torch.multiprocessing.set_sharing_strategy('file_system')  # 多进程开文件
    # if cfg.DEBUG or cfg.IS_FMAP_EVAL:
    #     raise Exception('调试 和 IS_FMAP_EVAL 模式无法使用')

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
        print('"tensorboard --logdir=runs --host=192.168.0.199", view at http://192.168.0.199:6006/')
        tb_writer = SummaryWriter(cfg.PATH_PROJECT_ROOT + '/runs')

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
