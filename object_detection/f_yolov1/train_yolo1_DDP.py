import argparse

import os

import torch
from torch import optim

from f_tools.GLOBAL_LOG import flog
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_lossfun import LossYOLOv1
from object_detection.f_yolov1.CONFIG_YOLO1 import CFG
from object_detection.f_yolov1.process_fun import init_model, data_loader, train_eval

'''
\home\feadre\.conda\pkgs\pytorch-1.6.0-py3.7_cuda10.2.89_cudnn7.6.5_0\lib\python3.7\site-packages\torch\distributed\launch.py
--nproc_per_node=2  /home/win10_sys/tmp/DL/object_detection/f_yolov1/train_yolo1_DDP.py
'''

if __name__ == '__main__':
    CFG.SAVE_FILE_NAME = os.path.basename(__file__)
    CFG.DATA_NUM_WORKERS = 8
    torch.multiprocessing.set_sharing_strategy('file_system')  # 多进程开文件
    if CFG.DEBUG:
        raise Exception('调试模式无法使用')

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)  # 设置单一显卡
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if (int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1) < 2:
        raise Exception('请查询GPU数据', os.environ["WORLD_SIZE"])
    print('当前 GPU', torch.cuda.get_device_name(args.local_rank),
          args.local_rank,  # 进程变量 GPU号 进程号
          '总GPU个数', os.environ["WORLD_SIZE"],  # 这个变量是共享的
          "ppid:%s ...pid: %s" % (os.getppid(), os.getpid())
          )

    '''------------------模型定义---------------------'''
    model = init_model(CFG)
    model.to(device)  # 这个不需要加等号
    model.train()

    losser = LossYOLOv1(
        grid=CFG.GRID,
        l_noobj=CFG.L_NOOBJ,
        l_coord=CFG.L_COORD,
        num_cls=CFG.NUM_CLASSES
    )

    # 最初学习率
    lr0 = 1e-4
    # optimizer = optim.Adam(model.parameters(), lr=lr0, weight_decay=5e-4)  # 权重衰减(如L2惩罚)(默认: 0)
    # optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.937, weight_decay=0.0005, nesterov=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=0.9, weight_decay=0.0005)

    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    start_epoch = load_weight(CFG.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device)
    # start_epoch = 0

    # '''---------------数据加载及处理--------------'''
    loader_train, loader_val = data_loader(CFG, device)

    flog.debug('---训练开始---epoch %s', start_epoch + 1)
    # v1不需要anchor
    train_eval(CFG, start_epoch, model, losser, optimizer, lr_scheduler,
               loader_train=loader_train, loader_val=loader_val, device=device,
               )

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
