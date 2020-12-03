import os
import shutil

import xmltodict
from tqdm import tqdm

from f_tools.GLOBAL_LOG import flog

if __name__ == '__main__':
    path_img = r'M:\AI\datas\VOC2012\test\JPEGImages'
    path_label = r'M:\AI\datas\VOC2012\test\Annotations'
    is_move = False
    is_copy = False

    # 在这个指定的目录下 目标自动创建 images gt_info
    path_dst = r'M:\AI\datas\VOC2012\f_map'
    path_imgages = os.path.join(path_dst, 'images')
    path_gt_info = os.path.join(path_dst, 'gt_info')

    if os.path.exists(path_dst):
        if not os.path.exists(path_imgages) and is_copy:
            os.mkdir(path_imgages)
        if not os.path.exists(path_gt_info) :
            os.mkdir(path_gt_info)
    else:
        raise Exception('path_dst目录不存在: %s' % path_dst)

    # for dirpath, dirnames, files in os.walk(path_label):
    names_xml = os.listdir(path_label)
    for name_xml in tqdm(names_xml):
        _name_txt = name_xml.split('.')[0] + '.txt'
        file_txt = os.path.join(path_gt_info, _name_txt)
        lines_write = []
        with open(os.path.join(path_label, name_xml)) as f:
            str_xml = f.read()
            doc = xmltodict.parse(str_xml)
            objs = doc['annotation']['object']
            if isinstance(objs, dict):
                xmin = str(float(objs['bndbox']['xmin']))
                ymin = str(float(objs['bndbox']['ymin']))
                xmax = str(float(objs['bndbox']['xmax']))
                ymax = str(float(objs['bndbox']['ymax']))
                _line = obj['name'] + ' ' + xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax + '\n'
                lines_write.append(_line)
            else:
                for obj in objs:
                    # 可能有多个目标
                    xmin = str(float(obj['bndbox']['xmin']))
                    ymin = str(float(obj['bndbox']['ymin']))
                    xmax = str(float(obj['bndbox']['xmax']))
                    ymax = str(float(obj['bndbox']['ymax']))
                    _line = obj['name'] + ' ' + xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax+'\n'
                    lines_write.append(_line)
        with open(file_txt, "w") as f:
            f.writelines(lines_write)
            lines_write.clear()
