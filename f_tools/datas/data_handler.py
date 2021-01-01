import os
import random
from shutil import copy

from tqdm import tqdm

from f_tools.GLOBAL_LOG import flog


def spilt_voc2txt(path_files, val_rate=0.3, overlay=False):
    '''
    划分文件为训练集和验证集,适用于所有数据在同一文件夹
    :param path_files: 指定 trainval 文件夹的路径
    :param val_rate: 验证集比例 或数量  为0不要验证集
    :return: 训练和验证的文件名
    '''
    if not os.path.exists(path_files):
        flog.debug('文件夹不存在 %s', path_files)
        exit(1)

    _file_name_train = 'train.txt'
    _file_name_val = 'val.txt'
    path_train = os.path.join(path_files, _file_name_train)
    path_val = os.path.join(path_files, _file_name_val)

    if os.path.exists(path_train):
        print('文件已存在 : ', path_train)
    else:
        path_xml = os.path.join(path_files, 'Annotations')

        # 文件名和目录都出来 只取文件名
        files_name = sorted([file.split('.')[0] for file in os.listdir(path_xml)])
        files_num = len(files_name)  # 文件数量
        flog.debug('总文件数 %s', files_num)
        # 随机选出val的index
        if val_rate > 1.:
            val_index = random.sample(range(0, files_num), k=int(val_rate))
        else:
            val_index = random.sample(range(0, files_num), k=int(files_num * val_rate))
        flog.debug('测试集数量 %s', len(val_index))

        train_files = []
        val_files = []
        for index, file_name in enumerate(files_name):
            if index in val_index:
                val_files.append(file_name)
            else:
                train_files.append(file_name)

        try:
            train_f = open(path_train, 'x')
            eval_f = open(path_val, 'x')
            train_f.write('\n'.join(train_files))  # 每个元素添加换行符
            eval_f.write('\n'.join(val_files))
        except FileExistsError as e:
            print(e)
            exit(1)
    return _file_name_train, _file_name_val,


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


def spilt_data2folder(path_root, val_rate=0.3):
    names_class = [cla for cla in os.listdir(path_root) if ".txt" not in cla]
    path_train_root = os.path.join(os.path.dirname(path_root), 'train')
    path_val_root = os.path.join(os.path.dirname(path_root), 'val')
    for n in names_class:
        path_train_n = os.path.join(path_train_root, n)
        path_val_n = os.path.join(path_val_root, n)

        if not os.path.exists(path_train_n):
            os.makedirs(path_train_n)
        else:
            flog.warning('文件夹已存在 %s', path_train_n)
            exit(-1)
        if not os.path.exists(path_val_n):
            os.makedirs(path_val_n)
        else:
            flog.warning('文件夹已存在 %s', path_val_n)
            exit(-1)

        path_cla_imgs = os.path.join(path_root, n)
        name_images = os.listdir(path_cla_imgs)
        num = len(name_images)
        eval_index = random.sample(name_images, k=int(num * val_rate))

        for i, n in tqdm(enumerate(name_images)):
            path_img = os.path.join(path_cla_imgs, n)
            if n in eval_index:  # 验值集
                path_img_new = os.path.join(path_val_n, n)
            else:
                path_img_new = os.path.join(path_train_n, n)
            copy(path_img, path_img_new)
            print("\r[{}] processing [{}/{}]".format(n, i + 1, num), end="")  # processing bar

    print("spilt_data2folder done!")


if __name__ == '__main__':
    path = r'M:\AI\datas\VOC2012\train'
    # path = r'M:\AI\datas\VOC2012\val'
    # path = r'M:\AI\datas\VOC2007\train'
    # path = r'M:\AI\datas\VOC2007\val'
    # path = r'M:\AI\datas\raccoon200\VOCdevkit\VOC2007'

    # spilt_data2folder(r'M:\datas\flower_data\flower_photos')
    # spilt_voc2txt(path, val_rate=256)
    spilt_voc2txt(path, val_rate=0)  # 为0不要验证集
