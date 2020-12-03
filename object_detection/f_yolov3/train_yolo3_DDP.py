'''解决linux导入出错'''
import os
import sys

from torch.utils.tensorboard import SummaryWriter

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
'''解决linux导入出错 完成'''
import torch
from f_tools.GLOBAL_LOG import flog
from f_tools.fits.fitting.f_fit_eval_base import mgpu_init
from object_detection.f_yolov3.CONFIG_YOLO3 import CFG
from object_detection.f_yolov3.process_fun import init_model, data_loader, train_eval
import numpy as np

'''
\home\feadre\.conda\pkgs\pytorch-1.6.0-py3.7_cuda10.2.89_cudnn7.6.5_0\lib\python3.7\site-packages\torch\distributed\launch.py
--nproc_per_node=2 /home/win10_sys/tmp/DL/object_detection/f_yolov3/train_yolo3_DDP.py
python -m torch.distributed.launch --nproc_per_node=2 /home/win10_sys/tmp/DL/object_detection/f_yolov3/train_yolo3_DDP.py


单GPU B16 416 F2 P50 time: 0.8156  0:15:02 (0.8430 s / it)
单GPU B8 416 F2 P50 time: 0.7084  0:12:47 (0.7175 s / it)


单GPU B15 416 F2 P50 time: 0.8396  0:16:18 (0.8573 s / it) mem: 6583
双GPU B15 416 F2 P50 time: 0.8950  0:15:49 (0.8869 s / it)
双GPU B15 416 F2 P50 time: 0.9017   mem: 6824
双GPU B8 MOSAIC 512 F2 P50 time: 1.1774  0:05:08 (1.1557 s / it) mem: 6212 CPU25 内存21
双GPU B8 MOSAIC 512 F2 P50 time: 1.0429  0:04:46 (1.0716 s / it) mem: 5761 CPU17 内存20

单GPU B8 MOSAIC 416 F1 P50 time: 1.8259  0:08:03 (1.8043 s / it) mem: 4275 d121 # 锁定
双GPU B32 MOSAIC 416 F1 P50 time: 2.9522   0:03:20 (3.0394 s / it) mem: 3467 d121 # 锁定
双GPU B48 MOSAIC 416 F1 P400 time: 4.9589   0:03:20 (3.0394 s / it) mem: 3467 d121 # 锁定
双GPU B16 MOSAIC 416 F1 P400 time: 1.9799   0:04:21 (1.9644 s / it) mem: 5763 d121


'''

if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise EnvironmentError("未发现GPU")

    if CFG.IS_MOSAIC:
        # CFG.IMAGE_SIZE = [512, 512]
        # CFG.ANC_SCALE = list(np.array(CFG.ANC_SCALE,dtype=np.float32) / 2)
        pass

    CFG.SAVE_FILE_NAME = os.path.basename(__file__)
    CFG.DATA_NUM_WORKERS = 6
    torch.multiprocessing.set_sharing_strategy('file_system')  # 多进程开文件
    if CFG.DEBUG:
        raise Exception('调试模式无法使用')

    args, device = mgpu_init()

    if args.rank == 0:
        # 主进程任务
        flog.info(args)
        if not os.path.exists(CFG.PATH_SAVE_WEIGHT):
            try:
                os.makedirs(CFG.PATH_SAVE_WEIGHT)
            except Exception as e:
                flog.error(' %s %s', CFG.PATH_SAVE_WEIGHT, e)
        tb_writer = SummaryWriter()
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

    '''------------------模型定义---------------------'''
    model, losser, optimizer, lr_scheduler, start_epoch, anc_obj = init_model(CFG, device, id_gpu=args.gpu)  # 初始化完成

    '''---------------数据加载及处理--------------'''
    loader_train, loader_val, train_sampler = data_loader(CFG, is_mgpu=True)
    # loader_train, loader_val, train_sampler = data_loader4mgpu(CFG)

    flog.debug('---训练开始---epoch %s', start_epoch + 1)
    train_eval(cfg=CFG, start_epoch=start_epoch, model=model, anc_obj=anc_obj,
               losser=losser, optimizer=optimizer, lr_scheduler=lr_scheduler,
               loader_train=loader_train, loader_val=loader_val, device=device,
               train_sampler=train_sampler
               )

    # torch.distributed.destroy_process_group()  # 释放进程

    flog.info('---%s--main执行完成--进程号：%s---- ' % (os.path.basename(__file__), torch.distributed.get_rank()))
