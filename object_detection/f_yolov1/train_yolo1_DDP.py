import argparse

import os

import torch
from torch import optim

from f_tools.GLOBAL_LOG import flog
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_lossfun import LossYOLOv1
from object_detection.f_yolov1.CONFIG_YOLO1 import CFG
from object_detection.f_yolov1.utils.process_fun import init_model, train_eval, data_loader4mgpu

'''
\home\feadre\.conda\pkgs\pytorch-1.6.0-py3.7_cuda10.2.89_cudnn7.6.5_0\lib\python3.7\site-packages\torch\distributed\launch.py
--nproc_per_node=2 /home/win10_sys/tmp/DL/object_detection/f_yolov1/train_yolo1_DDP.py
必须每一个设备的权重是一样的
开启SYSNC_BN B4 F1 P50 time: 0.7237
关闭SYSNC_BN B4 F2 P50 time: 0.4948 用时15:42
关闭SYSNC_BN B5 F2 P50 time: 0.4677 用时13:22  爆内存

'''

if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise EnvironmentError("未发现GPU")

    CFG.SAVE_FILE_NAME = os.path.basename(__file__)
    CFG.DATA_NUM_WORKERS = 8
    torch.multiprocessing.set_sharing_strategy('file_system')  # 多进程开文件
    if CFG.DEBUG:
        raise Exception('调试模式无法使用')

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])  # 多机时：当前是第几的个机器  单机时：第几个进程
        args.world_size = int(os.environ['WORLD_SIZE'])  # 多机时：当前是有几台机器 单机时：总GPU个数
        args.gpu = int(os.environ['LOCAL_RANK'])  # 多机时：当前GPU序号
    else:
        raise Exception('环境变量有问题 %s' % os.environ)

    torch.cuda.set_device(args.local_rank)

    device = torch.device("cuda", args.local_rank)  # 获取显示device
    torch.distributed.init_process_group(backend="nccl",
                                         init_method="env://",
                                         world_size=args.world_size,
                                         rank=args.rank
                                         )
    torch.distributed.barrier()  # 等待所有GPU初始化 初始化完成 629M

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

    losser = LossYOLOv1(
        grid=CFG.GRID,
        l_noobj=CFG.L_NOOBJ,
        l_coord=CFG.L_COORD,
        num_cls=CFG.NUM_CLASSES
    )

    # 学习率确定每步走多远，多GPU建议增大学习率？？ lr0 = 1e-4 * args.world_size
    lr0 = 1e-4

    optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=0.9, weight_decay=0.0005)
    optimizer.zero_grad()

    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    # ---------------!!! 必须每一个设备的权重是一样的 !!!-------------------
    start_epoch = load_weight(CFG.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device, is_mgpu=True)
    # start_epoch = 0

    '''---------------数据加载及处理--------------'''

    loader_train, loader_val, train_sampler = data_loader4mgpu(CFG)

    flog.debug('---训练开始---epoch %s', start_epoch + 1)
    # v1不需要anchor
    train_eval(CFG, start_epoch, model, losser, optimizer, lr_scheduler,
               loader_train=loader_train, loader_val=loader_val,
               device=device,
               train_sampler=train_sampler,
               )

    torch.distributed.destroy_process_group()  # 释放进程

    flog.info('---%s--main执行完成--进程号：%s---- ' % (os.path.basename(__file__), torch.distributed.get_rank()))
