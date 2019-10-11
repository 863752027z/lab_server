import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

print('torch:' + torch.__version__)
#定义一个网络

class Net(nn.Module):
    def __init__(self):
        #nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()
        #卷积层1表示输入的图片为单通道，6表示输出的通道数，3表示卷积核为3*3
        self.conv1 = nn.Conv2d(1, 6, 3)
        #线性层，输入1350个特征，输出10个特征
        self.fc1 = nn.Linear(1350, 10)
    def forward(self, x):
        print(x.size()) #结果[1,1,32,32]
        #卷积->激活->池化
        x = self.conv1(x) #根据卷积的尺寸计算公式，计算结果是30
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))  # 我们使用池化层，计算结果是15
        x = F.relu(x)
        print(x.size()) #结果[1,6,15,15]
        #reshape-1表示自适应
        x = x.view(x.size()[0], -1)
        print(x.size()) #这里就是fc1层的输入1350
        x = self.fc1(x)
        return x
while True:
    net = Net()
    print(net)

    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    y = torch.arange(0, 10).view(1, 10).float()

    print('更新参数')
    criterion = nn.MSELoss()
    loss = criterion(out, y)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    #更新参数
    optimizer.step()