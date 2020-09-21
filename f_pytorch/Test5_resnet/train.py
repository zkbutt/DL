import time

import torch
import torch.nn as nn
import torchvision
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim

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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


data_root = r'M:\datas\flower_data'  # get data root path
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

batch_size = 24
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

net = torchvision.models.resnet50(pretrained=False)


# load pretrain weights
model_weight_path = r"M:\AI\weights\pytorch\resnet50-19c8e357.pth"
missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)

# for param in net.parameters():
#     param.requires_grad = False
# change fc layer structure
inchannel = net.fc.in_features
net.fc = nn.Linear(inchannel, 5)
net.to(device)

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
            outputs = net(images.to(device))
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


