import re


# 将匹配的数字乘于 2
def double(matched):
    value = int(matched.group('value'))
    return str(value * 2)
s = 'A2312G4HFD567'
search = re.search('(?P<f_name>\d+)', s)  # 只匹配一次
print(search.group('f_name'))
print(re.sub('(?P<value>\d+)', double, s))  # 不停匹配
print(search)
print(search.groups())
