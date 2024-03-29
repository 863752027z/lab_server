import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import  Variable
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

x = np.random.randn(1000, 1)*4
w = np.array([0.5,])
bias = -1.68

y_true = np.dot(x, w) + bias  #真实数据
y = y_true + np.random.randn(x.shape[0])#加噪声的数据

class LinearRression(nn.Module):
    def __init__(self, input_size, out_size):
        super(LinearRression, self).__init__()
        self.x2o = nn.Linear(input_size, out_size)
    #初始化
    def forward(self, x):
        return self.x2o(x)
    #前向传递
batch_size = 10
model = LinearRression(1, 1)#回归模型
criterion = nn.MSELoss()  #损失函数
#调用cuda
model.cuda()
criterion.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
losses = []

epoches = 1000000000
for i in range(epoches):
    loss = 0
    optimizer.zero_grad()#清空上一步的梯度
    idx = np.random.randint(x.shape[0], size=batch_size)
    batch_cpu = Variable(torch.from_numpy(x[idx])).float()
    batch = batch_cpu.cuda()#很重要

    target_cpu = Variable(torch.from_numpy(y[idx])).float()
    target = target_cpu.cuda()#很重要
    output = model.forward(batch)
    loss += criterion(output, target)
    loss.backward()
    optimizer.step()

    if (i +1)%10 == 0:
        print('Loss at epoch[%s]: %.3f' % (i, loss.item()))
'''
    print(loss.data)
    losses.append(loss.data)

plt.plot(losses, '-or')
plt.xlabel("Epoch")
plt.xlabel("Loss")

plt.show()

'''