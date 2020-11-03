box = '11.01,11.02,11.03,11.05'
split = box.split(',')
# print(map(int, split))
print(list(map(int, map(float, split))))
