import os
import shutil

from tqdm import tqdm

from f_tools.GLOBAL_LOG import flog

'''
abc.txt --- abc.jpg   ltrb
    pictureframe 176 206 225 266
    heater 170 156 350 240
    pottedplant 272 190 316 259
    book 439 157 556 241
    book 437 246 518 351
    book 515 306 595 375
    book 407 386 531 476
    book 544 419 621 476
    book 609 297 636 392
    coffeetable 172 251 406 476
    coffeetable 2 236 102 395
    tvmonitor 2 10 173 238
    bookcase 395 2 639 470
    doll 482 83 515 107
    vase 276 233 304 259

'''
if __name__ == '__main__':
    # path_img = r'M:\AI\datas\widerface\val/images'
    # file_label = r'M:\AI\datas\widerface\val/label.txt'
    path_img = r'/home/win10_sys/AI/datas/widerface/WIDER_val/images'
    file_label = r'/home/win10_sys/AI/datas/widerface/val/label.txt'
    class_name = 'face'
    is_move = False
    path_dst = '/home/win10_sys/AI/datas/widerface/val'  # 目标自动创建 images gt_info
    path_imgages = os.path.join(path_dst, 'images')
    path_gt_info = os.path.join(path_dst, 'gt_info')

    if os.path.exists(path_dst):
        if not os.path.exists(path_imgages):
            os.mkdir(path_imgages)
        if not os.path.exists(path_gt_info):
            os.mkdir(path_gt_info)
    else:
        raise Exception('path_dst目录不存在: %s' % path_dst)

    if is_move:
        fun_copy = shutil.move
    else:
        fun_copy = shutil.copy

    if os.path.exists(file_label):
        f = open(file_label, 'r')
    else:
        raise Exception('标签文件不存在: %s' % file_label)
    lines = f.readlines()  # 读出标签文件的每一行 list
    name_jpg = None
    lines_write = []
    file_txt = None

    for line in tqdm(lines):
        line = line.rstrip()  # 去换行符
        if line.startswith('#'):  # 删除末尾空格
            if file_txt is not None:
                with open(file_txt, "w") as f:
                    f.writelines(lines_write)
                lines_write.clear()
            # 添加文件名
            file_src_img = os.path.normpath(os.path.join(path_img, line[2:]))
            str_split = file_src_img.split(os.sep)
            name_jpg = str_split[-1]  # 取文件名
            file_img = os.path.join(path_imgages, name_jpg)
            if os.path.exists(file_img):
                flog.warning('文件已复制 %s', file_img)
            fun_copy(file_src_img, file_img)

            _name_txt = name_jpg.split('.')[0] + '.txt'
            file_txt = os.path.join(path_gt_info, _name_txt)
        else:
            ds = line.split(' ')
            _t = []
            for d in ds:
                _t.append(int(d))
            # lrwh 转 lrtb
            _t[2] += _t[0]
            _t[3] += _t[1]

            _line = 'face ' + str(_t[0]) + ' ' + str(_t[1]) + ' ' + str(_t[2]) + ' ' + str(_t[3]) + '\n'
            lines_write.append(_line)
