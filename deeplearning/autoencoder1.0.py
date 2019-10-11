import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms, datasets


def draw(lose_list):
    x = range(0, len(loss_list))
    y = loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x, y, 'r-')
    plt.xlabel('batch_num')
    plt.ylabel('loss')
    plt.show()


def printGPU():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(0))


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),

        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 6, stride=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        #print('x:', x.shape)
        x = self.decoder(x)
        return x


printGPU()
num_epochs = 100
batch_size = 1
learning_rate = 1e-3
loss_list = []
file_path = 'F:/ZLW_generate/s15/'

data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
dataset = datasets.ImageFolder(file_path, transform=data_transform)
train_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device('cuda:0')
model = autoencoder().to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for idx, (data, label) in enumerate(train_loader):
        img = data
        img = img.to(device)
        # =========forward===========
        output = model(img)
        print('img:', img.shape)
        # print(temp_img.shape)

        if epoch > 8:
            temp_img = output[0][0]
            input = img[0][0].cpu().detach().numpy()
            temp_img = temp_img.cpu().detach().numpy()
            temp_img = np.concatenate([input, temp_img])
            cv2.imshow('img', temp_img)
            cv2.waitKey(300)

        loss = criterion(output, img)
        # =========backward=========
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ==============log=========
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))
        loss_list.append(loss.item())

draw(loss_list)
