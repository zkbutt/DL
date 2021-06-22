import json
import os
import shutil

import xmltodict
from tqdm import tqdm
import pandas as pd

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.f_coco.convert_data.csv2coco import to_coco, to_coco_4keypoint

'''
['2007_000027.jpg', '174.0', '101.0', '349.0', '351.0', 'person']
'''


def cre_csv(name_cls, path_root, res, train_type):
    file_csv = os.path.join(path_root, train_type + '.csv')
    with open(file_csv, "w") as f:
        for r in res:
            r = [str(i) for i in r]
            # ','.join(r) 只加前面
            s_ = ','.join(r) + ',' + name_cls + '\n'  # ---类别---
            f.write(s_)
    return file_csv


def cre_cls_json(classes_ids, name_cls, path_root):
    file_json = os.path.join(path_root, 'ids_classes.json')
    with open(file_json, 'w', encoding='utf-8') as f:
        json.dump(classes_ids, f, ensure_ascii=False, )
    ids_classes = {1: name_cls}
    file_json = os.path.join(path_root, 'classes_ids.json')
    with open(file_json, 'w', encoding='utf-8') as f:
        json.dump(ids_classes, f, ensure_ascii=False, )


def copy_img(file_path, is_copy, path_copy_dst, path_img_src, t_names):
    if is_copy:
        if not os.path.exists(path_copy_dst):
            os.makedirs(path_copy_dst)
        for name in t_names:
            _file_dst = os.path.join(path_copy_dst, name)
            if os.path.exists(_file_dst):
                _s = '文件有重名 %s' % _file_dst
                # raise Exception(_s)
                flog.error(_s)
            else:
                _file_src = os.path.join(path_img_src, file_path)
                shutil.copy(_file_src, _file_dst)


class Widerface2Csv:

    def __init__(self, path_root, file_name, mode) -> None:
        '''

        :param path_root:
        :param file_name:
        :param mode: bbox segm keypoints caption
        '''
        super().__init__()
        self.path_root = path_root
        self.path_file_label = os.path.join(path_root, file_name)
        self.mode = mode

    def to_csv(self, is_copy=False, path_dst=None, is_move=False):
        '''
           这个解析方法是定制写
               # info = [[filename0, "xmin ymin xmax ymax label0"],
               #         filename1, "xmin ymin xmax ymax label1"]
           :param is_copy: 是否复制图片到同一位置
           :param path_dst: 将图片文件复制到统一的位置 与is_move相对应
           :param is_move: 是否移动复制
           :return:
               infos=[filename,l,t,r,b,英文标准]
               classes={"Parade": 1, "Handshaking": 2, "People_Marching":3...}

               最终
                   58_Hockey_icehockey_puck_58_825.jpg,107,110,136,147,Hockey
                   ...
                   58_Hockey_icehockey_puck_58_825.jpg,90,384,141,435,Hockey
           '''
        if os.path.exists(self.path_file_label):
            f = open(self.path_file_label, 'r')
        else:
            raise Exception('标签文件不存在: %s' % self.path_file_label)

        if is_move:
            fun_copy = shutil.move
        else:
            fun_copy = shutil.copy
        lines = f.readlines()

        if self.mode == 'bbox':
            classes, infos = self.handler_b_k(fun_copy, is_copy, lines, path_dst)
        elif self.mode == 'keypoints':
            classes, infos = self.handler_b_k(fun_copy, is_copy, lines, path_dst)
        else:
            # caption
            classes, infos = self.handler_caption(fun_copy, is_copy, lines, path_dst)

        f.close()
        self.cre_file_json(classes, 'ids_widerface')

        # shutil.move(srcfile, dstfile)
        self.cre_file_csv(infos, self.mode)
        flog.info('数据生成成功 %s', )

    def handler_b_k(self, fun_copy, is_copy, lines, path_dst):
        '''
        生成 list(x,21)  1文件名 + 4boxxs+ (2+1)*5 +1 lable
        :param fun_copy:
        :param is_copy:
        :param lines:
        :param path_dst:
        :return:
        '''
        classes = {'human_face': 1}  # ---类别--- 写死
        infos = []
        _ts = []
        file = ''
        label = ''
        for line in tqdm(lines, desc='标注框个数'):
            line = line.rstrip()
            if line.startswith('#'):  # 删除末尾空格
                # 添加文件名
                path_src = os.path.normpath(os.path.join(self.path_root, 'images', line[2:]))
                str_split = path_src.split(os.sep)
                file = str_split[-1]  # 取文件名
                _file_dst = os.path.join(path_dst, file)

                if os.path.exists(_file_dst):
                    # flog.warning('文件已复制 %s', _file_dst)
                    continue
                if is_copy:
                    fun_copy(path_src, path_dst)

                # flog.debug('f %s', path_src

                # -----------这里处理类别------------
                # 59--people--driving--car/59_peopledrivingcar_peopledrivingcar_59_592.jpg
                # _t = str_split[-2].split('--')
                # _class_name = '_'.join(_t[1:])
                # label = _class_name
                # classes[_class_name] = int(_t[0]) + 1
                continue
            else:
                line = line.split(' ')  # 20个数字
                _t = []
                for i, x in enumerate(line):
                    if i < 4:
                        _t.append(int(x))
                    else:
                        _t.append(float(x))

                # lrwh 转 lrtb
                _t[2] += _t[0]
                _t[3] += _t[1]
                _ts.append(file)
                _ts.extend(_t[:4])

                if self.mode == 'keypoints':
                    if _t[4] == -1:
                        # 没有目标
                        _ts.extend([0.0] * 15)
                    else:
                        for i in range(6, 19, 3):
                            _t[i] = 2
                        _ts.extend(_t[4:19])
                _ts.extend(label)
            infos.append(_ts.copy())
            _ts.clear()
        return classes, infos

    def handler_caption(self, fun_copy, is_copy, lines, path_dst):
        classes = {}
        infos = []
        return classes, infos

    def cre_file_json(self, classes, name):
        file_json = '../_file/classes_' + name + '.json'
        with open(file_json, 'w', encoding='utf-8') as f:
            json.dump(classes, f, ensure_ascii=False, )

    def cre_file_csv(self, infos, name):
        path_csv = '../_file/csv_labels_' + name + '.csv'
        with open(path_csv, "w") as csv_labels:
            for info in infos:
                info = [str(i) for i in info]
                s_ = ','.join(info) + 'human_face' + '/n'  # ---类别---
                csv_labels.write(s_)
            csv_labels.close()


def handler_widerface():
    # 这个可以直接创建json
    data_type = 'train'  # val  test
    # type = 'val'
    path_root = os.path.join('M:/AI/datas/widerface', data_type)
    file_name = 'label.txt'
    mode = 'keypoints'  # bbox segm keypoints caption
    widerface_csv = Widerface2Csv(path_root, file_name, mode)
    # widerface_csv = Widerface2Csv(path_root, file_txt, 'keypoints')
    path_dst = os.path.join(path_root, 'images')  # 图片移动到此处
    widerface_csv.to_csv(is_copy=False, path_dst=path_dst, is_move=False)


def handler_face_5():
    # 这个可以直接创建json
    path_root = r'M:\AI\datas\face_5'
    path_copy_dst = os.path.join(path_root, 'images_13466')

    # file_path = os.path.join(path_root, 'train/trainImageList.txt')
    # train_type = 'train'  # train test val
    # num_file = 999999999

    file_path = os.path.join(path_root, 'train/testImageList.txt')
    train_type = 'test'  # train test val
    num_file = 200  # 这个是最大文件数

    mode = 'keypoints'  # bbox segm keypoints caption
    is_copy = False
    path_img_src = os.path.join(path_root, 'train', 'lfw_5590')  # is_copy = False 则无效

    # ['2007_000027.jpg', '174.0', '101.0', '349.0', '351.0', 'person']
    size_ann = 1 + 4 + 5 * 3 + 1  # 用于csv验证
    name_keypoints = [
        "left_eye", "right_eye", "nose",
        "left_mouth", "right_mouth",
    ]
    skeleton = [[3, 1], [3, 2], [3, 4], [3, 5]]  # 连接顺序

    if os.path.exists(file_path):
        f = open(file_path, 'r')
    else:
        raise Exception('标签文件不存在: %s' % file_path)
    lines = f.readlines()
    idx = 0
    res = []  # 标签数

    ''' 创建类型cls文件 '''
    name_cls = 'human_face'
    classes_ids = {name_cls: 1}  # 1开始
    cre_cls_json(classes_ids, name_cls, path_root)

    t_names = set()  # 用于验证重复
    for line in tqdm(lines[:num_file], desc='标注框个数'):
        msg = line.strip().split(' ')
        idx += 1
        # print('idx-', idx, ' : ', len(msg))

        file_path = msg[0]
        # print(file_path)
        file_name = file_path.split('\\')[1]
        bbox = [msg[1], msg[3], msg[2], msg[4]]  # list不支持数组索引
        keypoint = msg[5:15]
        keypoint_coco = []
        for i in range(int(len(keypoint) / 2)):
            # 转换并添加2
            x = keypoint[i * 2 + 0]
            y = keypoint[i * 2 + 1]
            if float(x) == 0 and float(y) == 0:
                flog.warning('xy均为0 %s', line)
                keypoint_coco.extend([x, y, '0'])
            else:
                keypoint_coco.extend([x, y, '2'])

        _t = []
        # 类别在后面写入时加的
        _t.append(file_name)
        _t.extend(bbox)
        _t.extend(keypoint_coco)
        t_names.add(file_name)
        # _t.extend(name_cls)  # 写死

        res.append(_t)

    copy_img(file_path, is_copy, path_copy_dst, path_img_src, t_names)

    ''' 创建csv 文件格式 [文件名,box,keypoint,类型] '''
    file_csv = cre_csv(name_cls, path_root, res, train_type)

    annotations = pd.read_csv(file_csv, header=None).values
    to_coco_4keypoint(annotations, classes_ids, path_copy_dst, path_root,
                      size_ann=size_ann,
                      name_keypoints=name_keypoints,
                      skeleton=skeleton,
                      is_copy=False, is_move=False,
                      file_coco=mode + '_' + train_type + '_' + str(len(res))  # 标签数
                      )


def handler_face_98():
    # 这个可以直接创建json
    path_root = r'M:\AI\datas\face_98'

    # 2500个
    # path_copy_dst = os.path.join(path_root, 'images_test_2118')
    # file_path = os.path.join(path_root, 'WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_test.txt')
    # train_type = 'test'  # train test val

    path_copy_dst = os.path.join(path_root, 'images_train_5316')
    file_path = os.path.join(path_root, 'WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_train.txt')
    train_type = 'train'  # train test val

    mode = 'keypoints'  # bbox segm keypoints caption
    is_copy = False
    path_img_src = os.path.join(path_root, 'WFLW_images')

    # ['2007_000027.jpg', '174.0', '101.0', '349.0', '351.0', 'person']
    size_ann = 1 + 4 + 98 * 3 + 1
    # 脸(逆时针)33 左眉(顺时针)9 右眉(顺时针)9 笔子(上下左右)9 左眼8 右眼8 嘴(先外圈再内圈)12+10
    name_keypoints = [
        # 脸33
        'face1', 'face2', 'face3', 'face4', 'face5',
        'face6', 'face7', 'face8', 'face9',
        'face10', 'face11', 'face12', 'face13',
        'face14', 'face15', 'face16', 'face17',
        'face18', 'face19', 'face20', 'face21',
        'face22', 'face23', 'face24', 'face25',
        'face26', 'face27', 'face28', 'face29',
        'face30', 'face31', 'face32', 'face33',
        # 左眉9
        'brow_l1', 'brow_l2', 'brow_l3', 'brow_l4', 'brow_l5',
        'brow_l6', 'brow_l7', 'brow_l8', 'brow_l9',
        # 右眉9
        'brow_r1', 'brow_r2', 'brow_r3', 'brow_r4', 'brow_r5',
        'brow_r6', 'brow_r7', 'brow_r8', 'brow_r9',
        # 笔子9
        'nose1', 'nose2', 'nose3', 'nose4', 'nose5',
        'nose6', 'nose7', 'nose8', 'nose9',
        # 左眼8
        'eye_l1', 'eye_l2', 'eye_l3', 'eye_l4',
        'eye_l5', 'eye_l6', 'eye_l7', 'eye_l8',
        # 右眼8
        'eye_r1', 'eye_r2', 'eye_r3', 'eye_r4',
        'eye_r5', 'eye_r6', 'eye_r7', 'eye_r8',
        # 嘴22
        'mouth1', 'mouth2', 'mouth3', 'mouth4',
        'mouth5', 'mouth6', 'mouth7', 'mouth8',
        'mouth9', 'mouth10', 'mouth11', 'mouth12',
        'mouth13', 'mouth14', 'mouth15', 'mouth16',
        'mouth17', 'mouth18', 'mouth19', 'mouth20',
        'mouth21', 'mouth22',
    ]
    skeleton = [
        # 脸33
        [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
        [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
        [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
        [16, 17], [17, 18], [18, 19], [19, 20], [20, 21],
        [21, 22], [22, 23], [23, 24], [24, 25], [25, 26],
        [26, 27], [27, 28], [28, 29], [29, 30], [30, 31],
        [31, 32], [32, 33],
        # 左眉9
        [34, 35], [35, 36], [36, 37], [37, 38],
        [38, 39], [39, 40], [40, 41], [41, 42],
        # 右眉9
        [43, 44], [44, 45], [45, 46], [46, 47],
        [47, 48], [48, 49], [49, 50], [50, 51],
        # 笔子9
        [52, 53], [53, 54], [54, 55], [55, 56],
        [56, 57], [57, 58], [58, 59], [59, 60],
        # 左眼8
        [61, 62], [62, 63], [63, 64], [64, 65],
        [65, 66], [66, 67], [67, 68],
        # 右眼8
        [69, 70], [70, 71], [71, 72], [72, 73],
        [73, 74], [74, 75], [75, 76],
        # 嘴22
        [76, 77], [78, 79], [79, 80], [80, 81],
        [81, 82], [82, 83], [83, 84], [84, 85],
        [85, 86], [86, 87], [87, 88], [88, 89],
        [89, 90], [90, 91], [91, 92], [92, 93],
        [93, 94], [94, 95], [95, 96], [96, 97],
        [97, 98],
    ]

    if os.path.exists(file_path):
        f = open(file_path, 'r')
    else:
        raise Exception('标签文件不存在: %s' % file_path)
    lines = f.readlines()
    idx = 0
    res = []

    ''' 创建类型cls文件 '''
    name_cls = 'human_face'
    classes_ids = {name_cls: 1}  # 1开始
    cre_cls_json(classes_ids, name_cls, path_root)

    t_names = set()  # 用于验证重复 copy
    for line in tqdm(lines, desc='标注框个数'):
        msg = line.strip().split(' ')
        idx += 1
        # print('idx-', idx, ' : ', len(msg))

        file_path = msg[206]
        # print(file_path)
        file_name = file_path.split('/')[1]
        bbox = msg[196:200]
        attributes = msg[200:206]  # 这个没用 姿势 表情 光照 化妆 阻挡 模糊
        keypoint = msg[0:196]
        keypoint_coco = []
        for i in range(int(len(keypoint) / 2)):
            # 转换并添加2
            x = keypoint[i * 2 + 0]
            y = keypoint[i * 2 + 1]
            keypoint_coco.extend([x, y, '2'])

        _t = []
        # 类别在后面写入时加的
        _t.append(file_name)
        _t.extend(bbox)
        _t.extend(keypoint_coco)
        t_names.add(file_name)
        # _t.extend(name_cls)  # 写死

        res.append(_t)

    copy_img(file_path, is_copy, path_copy_dst, path_img_src, t_names)

    ''' 创建csv 文件格式 [文件名,box,keypoint,类型] '''
    file_csv = cre_csv(name_cls, path_root, res, train_type)

    annotations = pd.read_csv(file_csv, header=None).values
    to_coco_4keypoint(annotations, classes_ids, path_copy_dst, path_root,
                      size_ann=size_ann,
                      name_keypoints=name_keypoints,
                      skeleton=skeleton,
                      is_copy=False, is_move=False,
                      file_coco=mode + '_' + train_type + '_' + str(len(res))  # 标签数
                      )


def hadler_voc():
    # 这个可以直接创建json
    mode = 'bbox'  # 'keypoints'  # 'bbox':
    # path_root = r'M:/AI/datas/VOC2012'
    path_root = r'M:/AI/datas/VOC2007'
    path_data = path_root + '/val'  # 这个是VOC文件名
    train_type = 'test'  # JSON名 name
    path_file_txt = 'train.txt'  # 文件名txt
    file_classes_ids = path_root + '/classes_ids.json'

    path_coco_save = path_root  # 这个是生成的根 目录必须存在

    path_img = path_data + '/JPEGImages'  # 真实图片路径
    path_txt = os.path.join(path_data, path_file_txt)
    path_xml = os.path.join(path_data, 'Annotations')
    with open(path_txt) as read:
        # 读每一行加上路径和扩展名---完整路径
        xml_list = [os.path.join(path_xml, line.strip() + ".xml") for line in read.readlines()]

    if len(xml_list) == 0:
        raise Exception('未读到数据')
    '''读文件获取类型名称'''
    # try:
    #     # {"类别1": 1, "类别2":2}
    #     path_classes = os.path.join(r'D:/tb/tb/ai_code/DL/f_tools/datas/f_coco/_file', 'classes_ids_voc.json')
    #     json_file = open(path_classes, 'r')
    #     class_dict = json.load(json_file)
    # except Exception as e:
    #     flog.error(e)
    #     exit(-1)

    rets = []
    # for file_xml in tqdm(xml_list[:1000]):  # 这里定义测试数量
    for file_xml in tqdm(xml_list, desc='组装CSV标签'):
        with open(file_xml) as file:
            str_xml = file.read()
        doc = xmltodict.parse(str_xml)
        filename = doc['annotation']['filename']

        ret = []
        objs = doc['annotation']['object']
        if isinstance(objs, dict):
            xmin = str(float(objs['bndbox']['xmin']))
            ymin = str(float(objs['bndbox']['ymin']))
            xmax = str(float(objs['bndbox']['xmax']))
            ymax = str(float(objs['bndbox']['ymax']))
            ret.append(filename)
            ret.extend([xmin, ymin, xmax, ymax])
            ret.extend(objs['name'])
            rets.append(ret)
        else:
            for obj in objs:
                # 可能有多个目标
                xmin = str(float(obj['bndbox']['xmin']))
                ymin = str(float(obj['bndbox']['ymin']))
                xmax = str(float(obj['bndbox']['xmax']))
                ymax = str(float(obj['bndbox']['ymax']))
                ret.append(filename)
                ret.extend([xmin, ymin, xmax, ymax])
                ret.extend(obj['name'])
                rets.append(ret.copy())
                ret.clear()

    # print(rets)
    # ['2007_000027.jpg', '174.0', '101.0', '349.0', '351.0', 'person']
    infos = rets
    file_csv = '../_file/csv_labels_' + 'voc_' + path_file_txt.split('.')[0] + '.csv'
    with open(file_csv, "w") as csv_labels:
        for info in infos:
            s_ = ','.join(info) + '\n'  # ---类别---
            csv_labels.write(s_)
        csv_labels.close()
    flog.debug('file_csv : %s', file_csv)

    '''这里是转换'''
    with open(file_classes_ids, 'r', encoding='utf-8') as f:
        classes_ids = json.load(f)  # 文件转dict 或list

    classes_name = []
    file = os.path.join(path_root, 'classes_name.txt')
    if not os.path.exists(file):
        for k, v in classes_ids.items():
            classes_name.append(k)
        with open(file, 'w') as f:
            f.write(' '.join(classes_name))

    to_coco(file_csv, classes_ids, path_img, path_coco_save, mode, is_copy=False, is_move=False,
            file_coco=train_type + '_' + str(len(xml_list)))


if __name__ == '__main__':
    '''
    将文件进行重新标注 并 重构到一个文件夹中  选框为 ltrb 可选复制或移动
    搜 # ---类别--- 写死
    bbox segm keypoints
    
    1、文件名,bbox,类别名
        0_Parade_marchingband_1_465.jpg,345,211,349,215,human_face 
    2、图片移到一个文件夹
    
    '''
    # handler_widerface()
    # hadler_voc()  # 这个可以直接创建json
    # handler_face_98()
    handler_face_5()
