'''解决linux导入出错'''
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from torch.utils.tensorboard import SummaryWriter
from object_detection.z_yolov1.CONFIG_YOLOV1 import CFG
from object_detection.z_yolov1.process_fun import init_model, data_loader, train_eval
from f_tools.fits.f_gpu.f_gpu_api import mgpu_init

'''解决linux导入出错 完成'''
import torch
from f_tools.GLOBAL_LOG import flog

'''
\home\feadre\.conda\pkgs\pytorch-1.6.0-py3.7_cuda10.2.89_cudnn7.6.5_0\lib\python3.7\site-packages\torch\distributed\launch.py
pycharm用这个 --nproc_per_node=2 /AI/temp/tmp_pycharm/DL/object_detection/z_yolov1/train_yolov1_DDP.py
linux用这个   
python -m torch.distributed.launch --nproc_per_node=2 /AI/temp/tmp_pycharm/DL/object_detection/z_yolov1/train_yolov1_DDP.py


单GPU B128 416 F1 P400 time: 3.5156  0:07:13 (4.2914 s / it) mem: 4285 mv2 # 锁定
双GPU B128 416 F1 P400 time: 4.6866  0:04:09 (4.9945 s / it) mem: 4287 mv2 # 锁定
双GPU B52  416 F1 P400 time: 3.0276  0:06:29 (3.1640 s / it) mem: 6133 mv2 
双GPU B52  416 F1 P400 time: 3.0864  0:06:32 (3.1897 s / it) mem: 6133 mv2 

voc
双GPU B120 416 F2 P400 time: 2.7259 data: 0.0005 0:03:24 (2.8785 s / it) mem: 4088 mv2 # 锁定
双GPU B52  416 F2 P400 time: 2.1853  0:06:29 (3.1640 s / it) mem: 6133 mv2 

data: 3.3477 正常

训练 id缺失: 9227 3808 279 7512
验证 id缺失: 29  1828 2501 3086

train_yolov1_mobilenet_v2-111_2.4843227863311768.pth
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.077
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.204
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.041
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.029
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.117
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.123
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.161
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.162
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.009
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.129
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.191

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
        # print('Start Tensorboard with "tensorboard --logdir=runs --host=192.168.0.199", view at http://localhost:6006/')
        # tb_writer = SummaryWriter()

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
