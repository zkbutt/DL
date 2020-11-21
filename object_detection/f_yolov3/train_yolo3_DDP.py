import argparse
import os
import torch
from torch import optim
from f_tools.GLOBAL_LOG import flog
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_lossfun import LossYOLOv3
from f_tools.fits.f_lr_fun import f_lr_cos
from f_tools.fits.fitting.f_fit_eval_base import mgpu_init
from f_tools.fun_od.f_anc import FAnchors
from object_detection.f_yolov1.utils.process_fun import data_loader4mgpu
from object_detection.f_yolov3.CONFIG_YOLO3 import CFG
from object_detection.f_yolov3.process_fun import init_model, data_loader, train_eval

'''
\home\feadre\.conda\pkgs\pytorch-1.6.0-py3.7_cuda10.2.89_cudnn7.6.5_0\lib\python3.7\site-packages\torch\distributed\launch.py
--nproc_per_node=2 /home/win10_sys/tmp/DL/object_detection/f_yolov3/train_yolo3_DDP.py

单GPU B16 F2 P50 time: 0.8156  0:15:02 (0.8430 s / it)
双GPU B15 F2 P50 time: 0.8950  0:15:49 (0.8869 s / it)
'''

if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise EnvironmentError("未发现GPU")

    CFG.SAVE_FILE_NAME = os.path.basename(__file__)
    CFG.DATA_NUM_WORKERS = 8
    torch.multiprocessing.set_sharing_strategy('file_system')  # 多进程开文件
    if CFG.DEBUG:
        raise Exception('调试模式无法使用')

    args, device = mgpu_init()

    if args.rank == 0:
        flog.info(args)

        '''tensorboard'''
        # from torch.utils.tensorboard import SummaryWriter
        # tb_writer = SummaryWriter()
        # if os.path.exists("./log") is False:
        #     os.makedirs("./log")
        # if os.path.exists("./log/weights") is False:
        #     os.makedirs("./log/weights")
        # 在每一批训练完成时在主进程rank=0中记录
        # tb_writer.add_scalar(tags[0], mean_loss, epoch)
        # tb_writer.add_scalar(tags[1], acc, epoch)
        # tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

    '''------------------模型定义---------------------'''
    model = init_model(CFG)  # 初始化完成
    if CFG.SYSNC_BN:
        # 不冻结权重的情况下可, 使用SyncBatchNorm后训练会更耗时
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    else:
        model.to(device)  # 这个不需要加等号
    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model.train()

    anc_obj = FAnchors(CFG.IMAGE_SIZE, CFG.ANC_SCALE, CFG.FEATURE_MAP_STEPS, CFG.ANCHORS_CLIP, device=device)
    # anchors = anchors.to(devi
    losser = LossYOLOv3(CFG.NUM_CLASSES, anc_obj.ancs)

    pg = model.parameters()
    lr0 = 1e-3
    lrf = lr0 / 100
    optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.937, weight_decay=0.0005, nesterov=True)
    # ---------------!!! 必须每一个设备的权重是一样的 !!!-------------------
    start_epoch = load_weight(CFG.FILE_FIT_WEIGHT, model,
                              optimizer=optimizer,
                              lr_scheduler=None,
                              device=device,
                              is_mgpu=True)
    lr_scheduler = f_lr_cos(optimizer, start_epoch, CFG.END_EPOCH, lrf_scale=0.01)

    # start_epoch = 0

    '''---------------数据加载及处理--------------'''
    loader_train, loader_val, train_sampler = data_loader(CFG, is_mgpu=True)
    # loader_train, loader_val, train_sampler = data_loader4mgpu(CFG)

    flog.debug('---训练开始---epoch %s', start_epoch + 1)
    train_eval(cfg=CFG, start_epoch=start_epoch, model=model, anc_obj=anc_obj,
               losser=losser, optimizer=optimizer, lr_scheduler=lr_scheduler,
               loader_train=loader_train, loader_val=loader_val, device=device,
               train_sampler=train_sampler
               )

    torch.distributed.destroy_process_group()  # 释放进程

    flog.info('---%s--main执行完成--进程号：%s---- ' % (os.path.basename(__file__), torch.distributed.get_rank()))
