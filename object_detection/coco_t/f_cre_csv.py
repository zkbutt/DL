import json
import os
import shutil

from tqdm import tqdm

from f_tools.GLOBAL_LOG import flog


class Widerface2Csv:

    def __init__(self, path_root, file_txt, mode) -> None:
        '''

        :param path_root:
        :param file_txt:
        :param mode: 'bboxs'  'keypoints'   'caption'
        '''
        super().__init__()
        self.path_root = path_root
        self.path_file_txt = os.path.join(path_root, file_txt)
        self.mode = mode

    def to_csv(self, is_copy=False, path_dst=None, is_move=False):
        '''
           这个解析方法是定制写
               # info = [[filename0, "xmin ymin xmax ymax label0"],
               #         filename1, "xmin ymin xmax ymax label1"]
           :param is_copy: 是否复制图片到同一位置
           :param path_dst: 将图片文件复制到统一的位置
           :param is_move: 是否
           :return:
               infos=[filename,l,t,r,b,英文标准]
               classes={"Parade": 1, "Handshaking": 2, "People_Marching":3...}

               最终
                   58_Hockey_icehockey_puck_58_825.jpg,107,110,136,147,Hockey
                   ...
                   58_Hockey_icehockey_puck_58_825.jpg,90,384,141,435,Hockey
           '''
        if os.path.exists(self.path_file_txt):
            f = open(self.path_file_txt, 'r')
        else:
            raise Exception('标签文件不存在: %s' % self.path_file_txt)

        if is_move:
            fun_copy = shutil.move
        else:
            fun_copy = shutil.copy
        lines = f.readlines()

        if self.mode == 'bboxs':
            classes, infos = self.handler_b_k(fun_copy, is_copy, lines, path_dst)
        elif self.mode == 'keypoints':
            classes, infos = self.handler_b_k(fun_copy, is_copy, lines, path_dst)
        else:
            # caption
            classes, infos = self.handler_caption(fun_copy, is_copy, lines, path_dst)

        f.close()
        self.cre_file_json(classes, self.mode)

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
        classes = {}
        infos = []
        _ts = []
        file = ''
        label = ''
        for line in tqdm(lines):
            line = line.rstrip()
            if line.startswith('#'):  # 删除末尾空格
                # 添加文件名
                path_src = os.path.normpath(os.path.join(self.path_root, 'images', line[2:]))
                if is_copy:
                    fun_copy(path_src, path_dst)
                str_split = path_src.split(os.sep)
                file = str_split[-1]  # 取文件名
                # flog.debug('f %s', path_src)
                _t = str_split[-2].split('--')
                # 59--people--driving--car\59_peopledrivingcar_peopledrivingcar_59_592.jpg
                _class_name = '_'.join(_t[1:])
                label = _class_name
                classes[_class_name] = int(_t[0]) + 1
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
                _ts.append(label)
            infos.append(_ts.copy())
            _ts.clear()
        return classes, infos

    def handler_caption(self, fun_copy, is_copy, lines, path_dst):
        classes = {}
        infos = []
        return classes, infos

    def cre_file_json(self, classes, name):
        f = open('./_file/classes_' + name + '.json', 'w')
        json.dump(json.dumps(classes, ensure_ascii=False), f, ensure_ascii=False)

    def cre_file_csv(self, infos, name):
        csv_labels = open('./_file/csv_labels_' + name + '.csv', "w")
        for info in infos:
            info = [str(i) for i in info]
            s_ = ','.join(info) + '\n'
            csv_labels.write(s_)
        csv_labels.close()


if __name__ == '__main__':
    '''
    将文件进行重新标注 并 重构到一个文件夹中  选框为 ltrb
    '''
    # 标注文件
    path_root = r'M:\datas\widerface\train'
    file_txt = '_label.txt'

    # widerface_csv = Widerface2Csv(path_root, file_txt, 'bboxs')
    widerface_csv = Widerface2Csv(path_root, file_txt, 'keypoints')
    path_dst = os.path.join(path_root, 'images')  # 图片移动到此处
    widerface_csv.to_csv(True, path_dst)
