import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #
# Create tensors of shape (10, 3) and (10, 2).
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# Build a fully connected layer.
linear = nn.Linear(3, 2)  # x*weight^T + bias <--> y
print('w: ', linear.weight)  # (out_features, in_features)
print('b: ', linear.bias)  # out_features

# Build loss function and optimizer.
criterion = nn.MSELoss(reduction='mean')  # mean square error
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward pass.
pred = linear(x)

# Compute loss.
loss = criterion(pred, y)
print('loss: ', loss.item())

# Backward pass.
loss.backward()

# Print out the gradients.
print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

# 1-step gradient descent(one forward and backward).
optimizer.step()

# You can also perform gradient descent at the low level.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# Print out the loss after 1-step gradient descent.
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())
