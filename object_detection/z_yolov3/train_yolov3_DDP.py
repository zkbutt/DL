'''解决linux导入出错'''
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
'''解决linux导入出错 完成'''
from object_detection.z_yolov3.CONFIG_YOLO3 import CFG
from object_detection.z_yolov3.train_yolo3 import train_eval_set, init_model
from f_tools.fits.fitting.f_fit_class_base import Train_Mgpu
import torch
from f_tools.GLOBAL_LOG import flog

'''
pycharm启动
\home\feadre\.conda\pkgs\pytorch-1.6.0-py3.7_cuda10.2.89_cudnn7.6.5_0\lib\python3.7\site-packages\torch\distributed\launch.py
--nproc_per_node=2 /AI/temp/tmp_pycharm/DL/object_detection/z_yolov3/train_yolov3_DDP.py

linux用这个   
python -m torch.distributed.launch --nproc_per_node=2 /AI/temp/tmp_pycharm/DL/object_detection/z_yolov3/train_yolov3_DDP.py

tensorboard --host=192.168.0.199 --logdir=/AI/temp/tmp_pycharm/DL/object_detection/z_yolov3/runs_type3

'''

if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise EnvironmentError("未发现GPU")
    cfg = CFG
    path_project_root = '/AI/temp/tmp_pycharm/DL/object_detection/z_yolov3'  # 这个要加/
    train = Train_Mgpu(cfg, train_eval_set, init_model, path_project_root)
    train.f_run()

    # torch.distributed.destroy_process_group()  # 释放进程
    flog.info('---%s--main执行完成--进程号：%s---- ' % (os.path.basename(__file__), torch.distributed.get_rank()))
