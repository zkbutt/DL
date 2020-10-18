import os

import argparse
import torch
from torch import optim

from f_tools.GLOBAL_LOG import flog
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_lossfun import LossOD_K
from f_tools.fun_od.f_anc import AnchorsFound
from object_detection.f_retinaface.CONFIG_F_RETINAFACE import CFG
from object_detection.f_retinaface.utils.process_fun import data_loader, init_model, train_eval

if __name__ == '__main__':
    '''
    time: 0.6086  data: 0.0841  剩余时间: 0  max mem: 3892
    Total time: 0:03:16 (0.7304 s / it)
    
    time: 1.3898  data: 0.7840  max mem: 3893
    time: 1.1856  data: 0.4661  max mem: 3893
    Total time: 0:05:48 (1.2965 s / it)
    '''
    '''------------------系统配置---------------------'''
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

    '''---------------数据加载及处理--------------'''
    loader_train, loader_val = data_loader(CFG, device)

    '''------------------模型定义---------------------'''
    model = init_model(CFG)
    model.to(device)  # 这个不需要加等号
    model.train()

    anchors = AnchorsFound(CFG.IMAGE_SIZE, CFG.ANCHORS_SIZE, CFG.FEATURE_MAP_STEPS, CFG.ANCHORS_CLIP).get_anchors()
    anchors = anchors.to(device)

    losser = LossOD_K(anchors, CFG.LOSS_WEIGHT, CFG.NEGATIVE_RATIO)
    # 权重衰减(如L2惩罚)(默认: 0)
    optimizer = optim.Adam(model.parameters(), 1e-3, weight_decay=5e-4)
    # 在发现loss不再降低或者acc不再提高之后，降低学习率
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    start_epoch = load_weight(CFG.FILE_FIT_WEIGHT, model, optimizer, lr_scheduler, device)
    # 权重初始模式
    # state_dict = torch.load(CFG.FILE_FIT_WEIGHT, map_location=device)
    # model_dict = model.state_dict()
    # keys_missing, keys_unexpected = model.load_state_dict(state_dict)
    # start_epoch = 0

    train_eval(CFG, start_epoch, model, anchors, losser, optimizer, lr_scheduler,
               loader_train=loader_train, loader_val=loader_val, device=device,
               )

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
