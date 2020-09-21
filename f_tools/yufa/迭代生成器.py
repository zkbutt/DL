# from collections import Iterator, Iterable, Generator


def 方法生成器(d):
    for e in d:
        yield 'Province:\t' + str(e)


class FIterator(object):
    def __init__(self, a):
        self.a = a
        self.len = len(self.a)
        self.cur_pos = -1

    def __iter__(self):
        return self

    def __next__(self):  # Python3中只能使用__next__()而Python2中只能命名为next()
        self.cur_pos += 1

        if self.cur_pos < self.len:
            return self.a[self.cur_pos]
        else:
            raise StopIteration()  # 表示至此停止迭代


if __name__ == '__main__':
    d = [1, 2, 'ab']

    _iter = 方法生成器(d)

    _iter = FIterator(d)
    print(_iter.__sizeof__())
    print(_iter.len)

    for i in _iter:
        print(i)

    for i in range(10):
        print(_iter.__next__())
