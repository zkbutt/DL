import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, datasets

'''
3892 48s
2743 54s

# ---半精度训练1---
scaler = GradScaler()
# ---半精度训练2---
with autocast():
    outputs = net(images.to(device))
    loss = loss_function(outputs, labels.to(device))

# ---半精度训练3---
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
'''

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0,1,2]))
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
# 设置主显卡
cuda_z = "cuda:2"
torch.cuda.set_device(cuda_z)
for i in range(torch.cuda.device_count()):  # 可用GPU
    print(torch.cuda.get_device_name(i))
device = torch.device(cuda_z if torch.cuda.is_available() else "cpu", )  # 默认GPU
print(device)

print('默认GPU', torch.cuda.get_device_name(torch.cuda.current_device()))  # 永远是cuda0

batch_size = 66

'''
1070    66  9线程 锁页4.1   21.751310329999797  63度
k80-1   66  9线程 锁页3.6   49.68736823100062   84度
k80-2   66  9线程 锁页3.6   51.620361883999976  65度
k80-x   66  9线程 锁页4.2   35.91468242600058   91 63度
'''

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

data_root = r'/home/bak3t/bak299g/AI/datas/flower_data'  # get data root path
train_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'train'),
                                     transform=data_transform["train"])
# train_num = len(train_dataset)
validate_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'val'),
                                        transform=data_transform["val"])
val_num = len(validate_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           pin_memory=True,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=9)

validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              pin_memory=True,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=9)

net = torchvision.models.resnet50(pretrained=False)

# load pretrain weights

model_weight_path = r"/home/bak3t/bak299g/AI/weights/pytorch/resnet50-19c8e357.pth"
# missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
# missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)

# for param in net.parameters():
#     param.requires_grad = False
# change fc layer structure
inchannel = net.fc.in_features
net.fc = nn.Linear(inchannel, 5)

net.to(device)
net = torch.nn.DataParallel(net, device_ids=[2, 1])  # ---多gpu
# net = torch.nn.DataParallel(net, device_ids=[0, 1])  # ---多gpu
# net = torch.nn.DataParallel(net)  # ---多gpu

# loss_function = nn.CrossEntropyLoss().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

best_acc = 0.0
save_path = './resNet34.pth'

# ---半精度训练1---
scaler = GradScaler()

for epoch in range(10):
    # train
    net.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()

        # ---半精度训练2---
        with autocast():
            # print(device)
            outputs = net(images.float().to(device))
            loss = loss_function(outputs, labels.to(device))

        # ---半精度训练3---
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # logits = net(images.to(device))
        # loss = loss_function(logits, labels.to(device))
        # loss.backward()
        # optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print(time.perf_counter() - t1)

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))
print('Finished Training')
