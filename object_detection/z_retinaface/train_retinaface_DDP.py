'''解决linux导入出错'''
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
'''解决linux导入出错 完成'''

from f_tools.fits.f_fit_fun import train_eval4od
from f_tools.fits.f_gpu.f_gpu_api import mgpu_init
from torch.utils.tensorboard import SummaryWriter
from object_detection.z_retinaface.CONFIG_RETINAFACE import CFG
import torch
from f_tools.GLOBAL_LOG import flog
from object_detection.z_retinaface.train_retinaface import init_model, fdatas_l2

'''
\home\feadre\.conda\pkgs\pytorch-1.6.0-py3.7_cuda10.2.89_cudnn7.6.5_0\lib\python3.7\site-packages\torch\distributed\launch.py
--nproc_per_node=2 /AI/temp/tmp_pycharm/DL/object_detection/z_retinaface/train_retinaface_DDP.py
python -m torch.distributed.launch --nproc_per_node=2 /AI/temp/tmp_pycharm/DL/object_detection/z_retinaface/train_retinaface_DDP.py


'''

if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise EnvironmentError("未发现GPU")
    cfg = CFG
    torch.multiprocessing.set_sharing_strategy('file_system')  # 多进程开文件
    if cfg.DEBUG or cfg.IS_FMAP_EVAL:
        raise Exception('调试 和 IS_FMAP_EVAL 模式无法使用')

    args, device = mgpu_init()

    '''---------------数据加载及处理--------------'''
    loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler = cfg.FUN_LOADER_DATA(cfg,
                                                                                                      is_mgpu=True, )
    '''------------------模型定义---------------------'''
    model, optimizer, lr_scheduler, start_epoch = init_model(cfg, device, id_gpu=args)

    '''---------------主进程任务启动--------------'''
    tb_writer = None
    if args.rank == 0:
        # 主进程任务
        flog.info(args)
        if not os.path.exists(CFG.PATH_SAVE_WEIGHT):
            try:
                os.makedirs(CFG.PATH_SAVE_WEIGHT)
            except Exception as e:
                flog.error(' %s %s', CFG.PATH_SAVE_WEIGHT, e)
        # tensorboard --logdir=runs_widerface --host=192.168.0.199
        # tensorboard --logdir=runs_voc --host=192.168.0.199
        # cfg.PATH_TENSORBOARD = 'raccoon200'
        print('"tensorboard --logdir=%s --host=192.168.0.199", view at http://192.168.0.199:6006/'
              % cfg.PATH_TENSORBOARD)
        tb_writer = SummaryWriter(os.path.join(cfg.PATH_PROJECT_ROOT, cfg.PATH_TENSORBOARD))

    print('cfg.BATCH_SIZE', cfg.BATCH_SIZE)
    '''---------------训练验证开始--------------'''
    train_eval4od(start_epoch=start_epoch, model=model, optimizer=optimizer,
                  fdatas_l2=fdatas_l2, lr_scheduler=lr_scheduler,
                  loader_train=loader_train, loader_val_fmap=loader_val_fmap, loader_val_coco=loader_val_coco,
                  device=device, train_sampler=train_sampler, eval_sampler=eval_sampler,
                  tb_writer=tb_writer,
                  )
    # torch.distributed.destroy_process_group()  # 释放进程
    flog.info('---%s--main执行完成--进程号：%s---- ' % (os.path.basename(__file__), torch.distributed.get_rank()))