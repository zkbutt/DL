import os
import sys

'''用户命令行启动'''
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from torch.utils.tensorboard import SummaryWriter

from object_detection.z_yolov1.CONFIG_YOLOV1 import CFG
from object_detection.z_yolov1.process_fun import data_loader, init_model, train_eval

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_fit_fun import init_od, base_set

'''

linux用这个   python /AI/temp/tmp_pycharm/DL/object_detection/z_yolov1/train_yolov1.py
'''
if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    # -----------通用系统配置----------------
    init_od()
    device, cfg, ids2classes = base_set(CFG)

    if cfg.IS_MOSAIC:
        cfg.IMAGE_SIZE = [640, 640]
        cfg.BATCH_SIZE = 20

    '''------------------模型定义---------------------'''
    model, optimizer, lr_scheduler, start_epoch = init_model(cfg, device, id_gpu=None)

    '''---------------数据加载及处理--------------'''
    loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler = data_loader(model.cfg, is_mgpu=False)

    # flog.debug('---训练开始---epoch %s', start_epoch + 1)
    # 有些参数可通过模型来携带  model.nc = nc

    # print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    # tb_writer = SummaryWriter()

    train_eval(start_epoch=start_epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
               loader_train=loader_train, loader_val_fmap=loader_val_fmap, loader_val_coco=loader_val_coco,
               device=device, train_sampler=None, eval_sampler=None, tb_writer=None,
               )

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
