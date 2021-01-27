'''解决linux导入出错'''
import os
import sys

from f_tools.datas.data_loader import DataLoader

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
'''解决linux导入出错 完成'''

from f_tools.fits.fitting.f_fit_fun import train_eval4od, show_train_info
from f_tools.fits.f_gpu.f_gpu_api import mgpu_init, mgpu_process0_init
from object_detection.z_retinaface.CONFIG_RETINAFACE import CFG
import torch
from f_tools.GLOBAL_LOG import flog
from object_detection.z_retinaface.train_retinaface import init_model, fdatas_l2, train_eval_set

'''
\home\feadre\.conda\pkgs\pytorch-1.6.0-py3.7_cuda10.2.89_cudnn7.6.5_0\lib\python3.7\site-packages\torch\distributed\launch.py
--nproc_per_node=2 /AI/temp/tmp_pycharm/DL/object_detection/z_retinaface/train_retinaface_DDP.py
python -m torch.distributed.launch --nproc_per_node=2 /AI/temp/tmp_pycharm/DL/object_detection/z_retinaface/train_retinaface_DDP.py

'''

if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise EnvironmentError("未发现GPU")
    cfg = CFG
    train_eval_set(cfg)
    # cfg.LR0 = 1e-3
    # cfg.BATCH_SIZE = 10  # 多GPU减小批次

    torch.multiprocessing.set_sharing_strategy('file_system')  # 多进程开文件
    if cfg.DEBUG or cfg.IS_FMAP_EVAL:
        raise Exception('调试 和 IS_FMAP_EVAL 模式无法使用')

    args, device = mgpu_init()

    '''---------------数据加载及处理--------------'''
    data_loader = DataLoader(cfg)
    _ret = data_loader.get_train_eval_datas(is_mgpu=True)
    loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler = _ret

    cfg.PATH_PROJECT_ROOT = cfg.PATH_HOST + '/AI/temp/tmp_pycharm/DL/object_detection/z_retinaface'  # 这个要改

    '''------------------模型定义---------------------'''
    model, optimizer, lr_scheduler, start_epoch = init_model(cfg, device, id_gpu=args)

    '''---------------主进程任务启动--------------'''
    tb_writer = None
    if args.rank == 0:
        tb_writer = mgpu_process0_init(args, cfg, loader_train, loader_val_coco, model, device)
        show_train_info(cfg, loader_train, loader_val_coco)

    '''---------------训练验证开始--------------'''
    train_eval4od(start_epoch=start_epoch, model=model, optimizer=optimizer,
                  fdatas_l2=fdatas_l2, lr_scheduler=lr_scheduler,
                  loader_train=loader_train, loader_val_fmap=loader_val_fmap, loader_val_coco=loader_val_coco,
                  device=device, train_sampler=train_sampler, eval_sampler=eval_sampler,
                  tb_writer=tb_writer, maps_def=cfg.MAPS_VAL
                  )
    # torch.distributed.destroy_process_group()  # 释放进程
    flog.info('---%s--main执行完成--进程号：%s---- ' % (os.path.basename(__file__), torch.distributed.get_rank()))
