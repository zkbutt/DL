import glob
import xml.etree.ElementTree as ET

import numpy as np
from tqdm import tqdm

from f_tools.f_general import show_time
from f_tools.fun_od.kmeans_anc.kmeans import kmeans, avg_iou


def load_dataset(path):
    '''
    将所有框按ltrb 归一化成框组合
    :param path:
    :return:
    '''
    dataset = []
    for xml_file in tqdm(glob.glob("{}/*xml".format(path))):  # 获取路径下文件名
        tree = ET.parse(xml_file)

        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))

        for obj in tree.iter("object"):
            xmin = float(obj.findtext("bndbox/xmin")) / width
            ymin = float(obj.findtext("bndbox/ymin")) / height
            xmax = float(obj.findtext("bndbox/xmax")) / width
            ymax = float(obj.findtext("bndbox/ymax")) / height

            dataset.append([xmax - xmin, ymax - ymin])  # w,h

    return np.array(dataset)


def t_anc_size(data, clusters):
    out = kmeans(data, k=clusters)  # 输出5,2
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    # 计算尺寸大小排序后的索引
    a = out[:, 0] + out[:, 1]
    indexs = np.argsort(a)  # 默认升序
    # indexs = np.argsort(-a)  # 降序
    print("Boxes:\n {}".format(out[indexs]))
    print("size:\n {}".format((out * size)[indexs]))
    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))


if __name__ == '__main__':
    '''
    VOC2012 17125 6 Accuracy: 63.18%
    VOC2012 17125 9 Accuracy: 67.88%
    VOC2012 17125 18 Accuracy: 75.18%
    输出anc的归一化比例
    '''
    ANNOTATIONS_PATH = r"M:\AI\datas\VOC2012\trainval\Annotations"
    CLUSTERS = 9
    size = [416, 416]
    data = load_dataset(ANNOTATIONS_PATH)

    show_time(t_anc_size, data, CLUSTERS)