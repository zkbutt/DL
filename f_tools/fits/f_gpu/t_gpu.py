import torch


def func(model):
    print("1:{}".format(torch.cuda.memory_allocated(0)))
    a = f1(model)
    print("2:{}".format(torch.cuda.memory_allocated(0)))
    b = f2(a)
    print("3:{}".format(torch.cuda.memory_allocated(0)))
    c = f3(b)
    print("4:{}".format(torch.cuda.memory_allocated(0)))
    d = f4(c)
    print("5:{}".format(torch.cuda.memory_allocated(0)))


def train_epoch(model, data):
    model.train()

    optim = torch.optimizer()

    for batch_data in data:
        print("1:{}".format(torch.cuda.memory_allocated(0)))
        output = model(batch_data)
        print("2:{}".format(torch.cuda.memory_allocated(0)))
        loss = loss(output, data.target)
        print("3:{}".format(torch.cuda.memory_allocated(0)))
        optim.zero_grad()
        print("4:{}".format(torch.cuda.memory_allocated(0)))
        loss.backward()
        print("5:{}".format(torch.cuda.memory_allocated(0)))
        func(model)
        print("6:{}".format(torch.cuda.memory_allocated(0)))


def train(model, epochs, data):
    for e in range(epochs):
        print("1:{}".format(torch.cuda.memory_allocated(0)))
        train_epoch(model, data)
        print("2:{}".format(torch.cuda.memory_allocated(0)))
        eval(model, data)
        print("3:{}".format(torch.cuda.memory_allocated(0)))


def pro_weight(p, x, w, alpha=1.0, cnn=True, stride=1):
    if cnn:
        _, _, H, W = x.shape
        F, _, HH, WW = w.shape
        S = stride  # stride
        Ho = int(1 + (H - HH) / S)
        Wo = int(1 + (W - WW) / S)
        for i in range(Ho):
            for j in range(Wo):
                # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
                r = x[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1)
                # r = r[:, range(r.shape[1] - 1, -1, -1)]
                k = torch.mm(p, torch.t(r))
                p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
        w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(p.data)).view_as(w)
    else:
        r = x
        k = torch.mm(p, torch.t(r))
        p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
        w.grad.data = torch.mm(w.grad.data, torch.t(p.data))

