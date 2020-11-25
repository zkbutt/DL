import datetime
import os
import random
import string
import threading


def generate_random_password(password_length=10):
    """
    生成指定长度的密码字符串，当密码长度超过3时，密码中至少包含：
    1个大写字母+1个小写字母+1个特殊字符
    :param password_length:密码字符串的长度
    :return:密码字符串
    """
    special_chars = '~%#%^&*'
    password_chars = string.ascii_letters + string.digits + special_chars

    char_list = [
        random.choice(string.ascii_lowercase),
        random.choice(string.ascii_uppercase),
        random.choice(special_chars),
    ]
    if password_length > 3:
        # random.choice 方法返回一个列表，元组或字符串的随机项
        # (x for x in range(N))返回一个Generator对象
        # [x for x in range(N)] 返回List对象
        char_list.extend([random.choice(password_chars) for _ in range(password_length - 3)])
    # 使用random.shuffle来将list中元素打乱
    random.shuffle(char_list)
    return ''.join(char_list[0:password_length])


def random_letters(n):
    '''

    :param n:指定数量
    :return: [] 大小写字母数组
    '''
    letters_list = []
    while len(letters_list) < n:
        # a_str = string.ascii_uppercase # 所有大写字母
        a_str = string.ascii_letters  # 所有字母
        # a_str = string.ascii_lowercase # #所有小写字母
        random_letter = random.choice(a_str)
        if (random_letter not in letters_list):
            letters_list.append(random_letter)
    return letters_list


LOCK = threading.Lock()
FILE_PATH = './_file/number.txt'


def create_number(s, hlen=5):
    '''

    :param s: 前缀
    :param hlen:  流水位数
    :return:
    '''
    LOCK.acquire()

    # 拼音首字母前缀
    # py = xpinyin.Pinyin()
    # s = py.get_initials(s, '')

    # now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    now_time = datetime.datetime.now().strftime('%Y%m%d')
    num = ''
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))  # 找这个文件的路径
        f1 = open(os.path.join(script_dir, FILE_PATH), 'r+', encoding='utf-8')
        dx = f1.read().split('&')
        if dx[0] == now_time:
            num = str(int(dx[1]) + 1)
            f1.seek(0)
            f1.truncate()
            f1.write(now_time + '&' + num)
        else:
            num = '1'
            f1.seek(0)
            f1.truncate()
            f1.write(now_time + '&' + num)
        f1.close()
    except:
        # f1 = open(FILE_PATH, 'w', encoding='utf-8')
        with open(FILE_PATH, mode='w', encoding='utf-8') as ff:
            print(FILE_PATH + '出错拉  创建成功')
    LOCK.release()
    return str(s) + now_time + num.zfill(hlen)


if __name__ == '__main__':
    random_password = generate_random_password(password_length=10)
    print(random_password)
    print(random_letters(19))
    print(create_number('abc', 3))

    material_code = random.sample([1, 2, 3, 5, 5, 6], 3)  # 这个是不重复选取
