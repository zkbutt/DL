import json
import os
import shutil

from tqdm import tqdm

from f_tools.GLOBAL_LOG import flog


def widerface_bboxs2csv(path_file, file_name, path_dst=None, is_move=False):
    '''
    这个解析方法是定制写
        # info = [[filename0, "xmin ymin xmax ymax label0"],
        #         filename1, "xmin ymin xmax ymax label1"]
    :param path_file: 原根目录
    :param file_name: 标注文件名
    :param path_dst: 将图片文件复制
    :param is_move: 是否
    :return:
        infos=[filename,l,t,r,b,英文标准]
        classes={"Parade": 1, "Handshaking": 2, "People_Marching":3...}

        最终
            58_Hockey_icehockey_puck_58_825.jpg,107,110,136,147,Hockey
            ...
            58_Hockey_icehockey_puck_58_825.jpg,90,384,141,435,Hockey
    '''
    path = os.path.join(path_file, file_name)
    if os.path.exists(path):
        f = open(path, 'r')
    else:
        raise Exception('标签文件不存在: %s' % path_file)

    classes = {}
    infos = []
    _ts = []
    file = ''
    label = ''
    if is_move:
        fun_copy = shutil.move
    else:
        fun_copy = shutil.copy

    lines = f.readlines()
    for line in tqdm(lines):
        line = line.rstrip()
        if line.startswith('#'):  # 删除末尾空格
            # 添加文件名
            path_src = os.path.normpath(os.path.join(path_file, 'images', line[2:]))
            fun_copy(path_src, path_dst)
            str_split = path_src.split(os.sep)
            file = str_split[-1]  # 取文件名
            # flog.debug('f %s', path_src)
            _t = str_split[-2].split('--')
            _class_name = '_'.join(_t[1:])  # 59--people--driving--car\59_peopledrivingcar_peopledrivingcar_59_592.jpg
            label = _class_name
            classes[_class_name] = int(_t[0]) + 1
            continue
        else:
            line = line.split(' ')
            _bbox = [int(float(x)) for x in line]  # 这里 lrtb
            # _t = list(itemgetter(0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17)(_bbox))
            _t = _bbox
            # lrwh 转 lrtb
            _t[2] += _t[0]
            _t[3] += _t[1]
            _ts.append(file)
            _ts.append(_t[:4])
            _ts.append(label)
        infos.append(_ts.copy())
        _ts.clear()
    return infos, classes


def widerface_keypoints2csv(path_file, file_name, path_dst=None, is_move=False):
    pass


if __name__ == '__main__':
    '''
    将文件进行重新标注 并 重构到一个文件夹中  选框为 ltrb
    '''
    # 标注文件
    path_file = r'D:\down\AI\datas\widerface\val'
    file_txt = 'label.txt'
    path_dst = os.path.join(path_file, 'images')  # 图片移动到此处

    infos, classes = widerface_bboxs2csv(path_file, file_txt, path_dst)

    # infos, classes = widerface_keypoints2csv(path_file, file_txt, path_dst) # 待完成

    f = open('./file/classes.json', 'w')
    json.dump(json.dumps(classes, ensure_ascii=False), f, ensure_ascii=False)

    # shutil.move(srcfile, dstfile)
    csv_labels = open("./file/csv_labels.csv", "w")
    for filename, bbox, label in infos:
        csv_labels.write('%s,%s,%s,%s,%s,%s\n' % (filename, bbox[0], bbox[1], bbox[2], bbox[3], label))
    csv_labels.close()
    flog.info('数据生成成功 %s', )
