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
            raise StopIteration  # 表示至此停止迭代


class MySquares():

    def __init__(self) -> None:
        self.num = 0

    def __iter__(self):  # 返回自身的迭代器
        return self

    def __next__(self):
        # ret = torch.rand((300, 300))
        ret = self.num
        # 这里是预加载
        if self.num == 5:
            raise StopIteration()
        # self.next_input, self.next_target = next(self.loader)  # 加载到显存
        self.num += 1
        return ret


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
