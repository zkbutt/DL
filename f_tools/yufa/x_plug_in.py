import argparse
import random
import time

from tqdm import tqdm

from object_detection.yolov3_spp.utils import torch_utils


def t_argparse():
    # 工具 t_argparse 使用
    global i
    '''
       usage: t001.py [-h] [--name NAME] -f FAMILY t4 integers [integers ...]
    '''
    parser = argparse.ArgumentParser(description='命令行中传入一个数字')
    parser.add_argument('t4', type=str, help='传入数字')  # 必须填
    parser.add_argument('integers', type=str, nargs='+', help='传入数字')  # 至少传一个
    parser.add_argument('--name', type=str, help='传入姓名', )  # --表示可选参数, required必填
    parser.add_argument('-f', '--family', default='张三的家', type=str, help='传入姓名', required=True)  # --表示可选参数
    args = parser.parse_args()  # 解析出来是一个命名空间对象 直接.属性使用
    print(args)
    print(args.integers)
    print(args.name)
    print(args.family)
    r = range(3, 6)
    for i in r:
        print(i)


def t_tqdm():
    # 工具进度t_tqdm使用
    num_show = 100
    num_total = 1000
    with tqdm(total=num_total, desc=f'复制文件111111111111111111111111111111', postfix=dict, mininterval=0.3) as pbar:
        '''
        1.01s/it        1秒/次
        63.65it/s       次/秒
        '''
        for i in range(num_total):
            time.sleep(0.01)
            d = {
                'Conf Loss': random.randint(1, 99),
                'Regression Loss1': random.randint(1, 99),
                'Regression Loss2': random.randint(1, 99),
                'Regression Loss3': random.randint(1, 99),
                'LandMark Loss': random.randint(1, 99),
                'lr': random.randint(1, 99),
                's/step': random.randint(1, 99),
            }

            pbar.set_postfix(**d)
            pbar.update(num_total / num_total)


def t_random(a, b):
    torch.manual_seed(7) # cpu
    torch.cuda.manual_seed(7) #gpu
    np.random.seed(7) #numpy
    random.seed(7) #random and transforms
    # 工具随机t_random使用
    import numpy as np
    np.random.rand() * (b - a) + a

    np.random.shuffle(a)  # 随机洗牌np数组 按维度1

    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)

    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # 0.5~1.5之间


if __name__ == '__main__':
    # t_tqdm()

    t_argparse()

    # t_random(0.7, 2.1)
