import json

from f_tools.GLOBAL_LOG import flog


def base1():
    global data
    path_json4write = './file/path_json4write.json'
    path_json4dump = './file/path_json4dump.json'
    path_json4dumps_dump = './file/path_json4dumps_dump.json'  # 这个是字符串
    d1 = {'供应商': '重庆飞卓', 'aaa': 123, 'bbb': 456, 333: '123'}
    d1 = [{'供应商': '重庆飞卓', 'aaa': 123, 'bbb': 456, 333: '123'}, {'供应商': '重庆飞卓', 'aaa': 123, 'bbb': 456, 333: '123'}]
    with open(path_json4write, 'w') as f:
        data = json.dumps(d1)
        f.write(data)
    with open(path_json4dump, 'w', encoding='utf-8') as f:
        json.dump(d1, f, ensure_ascii=False, )
    # 这个是保字符串
    with open(path_json4dumps_dump, 'w') as f:
        data = json.dumps(d1)
        json.dump(data, f)
    with open(path_json4dump, 'r', encoding='utf-8') as f:
        data_read = json.load(f)  # 文件转dict 或list
    print(data_read)
    print(type(data_read))
    with open(path_json4dumps_dump, 'r') as f:
        data_read = json.load(f)  # 读进来是字符串
    print(data_read)
    print(type(data_read))


if __name__ == '__main__':
    # base1()
    # path = r'M:\AI\datas\widerface\coco\annotations\person_keypoints_train2017.json'  # 这个是字符串
    path = r'p.json'  # 这个是字符串
    ids = [9227, 7512, 3808, 279]

    with open(path, 'r') as f:
        data_dict = json.load(f)  # 读进来是字符串
        print()
        for img in data_dict['images'][:]:
            if img['id'] in ids:
                data_dict['images'].remove(img)
                flog.debug('del %s', img)
        for ann in data_dict['annotations'][:]:
            if ann['image_id'] in ids:
                data_dict['annotations'].remove(ann)
                flog.debug('del %s', ann)

    with open('p.json', 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, )
