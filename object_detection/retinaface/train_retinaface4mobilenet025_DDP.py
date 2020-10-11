import os
import sys

# 加入根目录避免找不到包
from object_detection.retinaface.train_retinaface import data_loader, model_init, trainning

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

import argparse

import torch

from f_tools.GLOBAL_LOG import flog

from object_detection.retinaface.CONFIG_RETINAFACE import PATH_SAVE_WEIGHT, PATH_DATA_ROOT, DEBUG, IMAGE_SIZE, \
    MOBILENET025

if __name__ == "__main__":
    '''
    CUDA_VISIBLE_DEVICES=2,1,0 python -m torch.distributed.launch --nproc_per_node=3  /home/fast120/ai_code/DL/tmp/pycharm_project_243/f_pytorch/t_multi_gpu_DDP.py
    python -m torch.distributed.launch --nproc_per_node=2  /home/win10_sys/tmp/pycharm_project_243/object_detection/retinaface/train_retinaface4mobilenet025_DDP.py

    '''
    '''------------------系统配置---------------------'''
    if DEBUG:
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

    # claxx = RESNET50  # 这里根据实际情况改
    claxx = MOBILENET025  # 这里根据实际情况改

    '''---------------数据加载及处理--------------'''
    loader_train, loader_val = data_loader(device)

    '''------------------模型定义---------------------'''
    model, anchors, losser, optimizer, lr_scheduler, start_epoch = model_init(claxx, device)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank,
        # this should be removed if we update BatchNorm stats
        broadcast_buffers=True,
    )
    '''------------------模型训练---------------------'''
    trainning(start_epoch, model, device, anchors, losser, optimizer, lr_scheduler, loader_train, loader_val)

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
