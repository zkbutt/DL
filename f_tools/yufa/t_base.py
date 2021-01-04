class T001:
    pass


if __name__ == '__main__':
    t001 = T001()
    print(t001)
    t001.abc = 123
    print(t001.abc)

    t002 = object()  # 这个要报错
    t002 = {}
    t002.abc = 123
    print(t002)
