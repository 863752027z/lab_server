import cv2
import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms, datasets
from collections import OrderedDict


os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device('cuda:0')


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        encoder_layer = OrderedDict([
            ('Con1', nn.Conv2d(3, 32, 4, stride=2, padding=1)),
            ('BatchNorm1', nn.BatchNorm2d(32)),
            ('LeakyReLU1', nn.LeakyReLU(0.2, True)),

            ('Con2', nn.Conv2d(32, 64, 4, stride=2, padding=1)),
            ('BatchNorm2', nn.BatchNorm2d(64)),
            ('LeakyReLU2', nn.LeakyReLU(0.2, True)),

            ('Con3', nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            ('BatchNorm3', nn.BatchNorm2d(128)),
            ('LeakyReLU3', nn.LeakyReLU(0.2, True)),

            ('Con4', nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            ('BatchNorm4', nn.BatchNorm2d(256)),
            ('LeakyReLU4', nn.LeakyReLU(0.2, True)),

            ('Con5', nn.Conv2d(256, 256, 4, stride=2, padding=1)),
            ('BatchNorm5', nn.BatchNorm2d(256)),
            ('LeakyReLU5', nn.LeakyReLU(0.2, True)),

            ('Con6', nn.Conv2d(256, 256, 4, stride=2, padding=1)),
            ('BatchNorm6', nn.BatchNorm2d(256)),
            ('LeakyReLU6', nn.LeakyReLU(0.2, True)),

            ('Con7', nn.Conv2d(256, 256, 4, stride=2, padding=1)),
            ('BatchNorm7', nn.BatchNorm2d(256)),
            ('LeakyReLU7', nn.LeakyReLU(0.2, True)),

            ('Con8', nn.Conv2d(256, 256, 4, stride=2, padding=1)),
        ])
        self.Encoder = nn.Sequential(encoder_layer)

    def forward(self, x):
        x = self.Encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        decoder_layer = OrderedDict([
            ('Upsample1', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            ('Con1', nn.Conv2d(512, 32, 3, stride=1, padding=1)),
            ('BatchNorm1', nn.BatchNorm2d(32)),
            ('ReLU1', nn.ReLU()),

            ('Upsample2', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            ('Con2', nn.Conv2d(32, 64, 3, stride=1, padding=1)),
            ('BatchNorm2', nn.BatchNorm2d(64)),
            ('ReLU2', nn.ReLU()),

            ('Upsample3', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            ('Con3', nn.Conv2d(64, 128, 3, stride=1, padding=1)),
            ('BatchNorm3', nn.BatchNorm2d(128)),
            ('ReLU3', nn.ReLU()),

            ('Upsample4', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            ('Con4', nn.Conv2d(128, 256, 3, stride=1, padding=1)),
            ('BatchNorm4', nn.BatchNorm2d(256)),
            ('ReLU4', nn.ReLU()),

            ('Upsample5', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            ('Con5', nn.Conv2d(256, 256, 3, stride=1, padding=1)),
            ('BatchNorm5', nn.BatchNorm2d(256)),
            ('ReLU5', nn.ReLU()),

            ('Upsample6', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            ('Con5', nn.Conv2d(256, 256, 3, stride=1, padding=1)),
            ('BatchNorm5', nn.BatchNorm2d(256)),
            ('ReLU6', nn.ReLU()),

            ('Upsample7', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            ('Con7', nn.Conv2d(256, 256, 3, stride=1, padding=1)),
            ('BatchNorm7', nn.BatchNorm2d(256)),
            ('ReLU7', nn.ReLU()),

            ('Upsample8', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            ('Con8', nn.Conv2d(256, 3, 3, stride=1, padding=1)),
            ('Tanh', nn.Tanh())
        ])
        self.Decoder = nn.Sequential(decoder_layer)

    def forward(self, x):
        x = self.Decoder(x)
        return x


class LstmCell(nn.Module):
    def __init__(self):
        super(LstmCell, self).__init__()
        self.LstmCell = nn.LSTMCell(input_size=256, hidden_size=256)

    def forward(self, xt, h, c):
        x = [h, c]
        h, c = self.LstmCell(xt, x)
        return h, c


def printGPU():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(0))


def draw(loss_list):
    x = range(0, len(loss_list))
    y = loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x, y, 'r-')
    plt.xlabel('batch_num')
    plt.ylabel('loss')
    plt.show()


def save_data_to_excel(data, path):
    print(datetime.datetime.now())
    print('generating:', path)
    print(data.shape)

    data_df = pd.DataFrame(data)
    writer = pd.ExcelWriter(path)
    data_df.to_excel(writer, 'page_1', float_format='%.5f') # float_format 控制精度
    writer.save()

    print('done')


def read_data_from_excel(path):
    df = pd.read_excel(path, 'page_1')
    data = np.array(df)
    data = np.delete(data, 0, axis=1)
    return data


def get_path(base_path):
    path_list = []
    for root, dirs, files in os.walk(base_path):
        for i in range(len(dirs)):
            temp_path = base_path + '/' + dirs[i]
            path_list.append(temp_path)
        break
    return path_list


def trainLoader(file_path, batch_size, shuffle, num_workers):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])

    data_set = datasets.ImageFolder(file_path,
                                    transform=data_transform)

    train_loader = Data.DataLoader(dataset=data_set,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   num_workers=num_workers)
    return train_loader


def testLoader(file_path, batch_size, shuffle, num_workers):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])

    data_set = datasets.ImageFolder(file_path,
                                    transform=data_transform)

    test_loader = Data.DataLoader(dataset=data_set,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   num_workers=num_workers)
    return test_loader


def train(loader_list, learning_rate, num_epochs, seq):
    cell_model = LstmCell().to(device)
    encoder_model = Encoder().to(device)
    decoder_model = Decoder().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD([
        {'params': encoder_model.parameters()},
        {'params': cell_model.parameters()},
        {'params': decoder_model.parameters()}
    ], lr=learning_rate, momentum=0.9)

    loss_list = []
    for epoch in range(num_epochs):
        for i in range(len(loader_list)):
            train_loader = loader_list[i]
            for idx, (data, label) in enumerate(train_loader):
                if data.shape[0] < seq:
                    break
                h = torch.zeros(seq-1, 256).to(device)
                c = torch.zeros(seq-1, 256).to(device)
                data = data.to(device)  #4*3*256*256
                # =========forward===========
                encoder_output = encoder_model(data)
                encoder_output = encoder_output.view((encoder_output.shape[0], encoder_output.shape[1]))    #4*256
                temp_target = encoder_output[0].view(1, encoder_output.shape[1], 1, 1)     #1*256*1*1
                temp_source = encoder_output[1:].view(3, encoder_output.shape[1])    #3*256

                h, c = cell_model(temp_source, h, c)
                cell_output = h[-1].view(1, 256, 1, 1)

                decoder_input = torch.cat((temp_target, cell_output), 1)    #1*512*1*1
                decoder_output = decoder_model(decoder_input)   #1*3*256*256

                target_img = data[0:1, :, :, :]     #1*3*256*256
                loss = criterion(decoder_output, target_img)
                # =========backward=========
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # ============log===========
                print('epoch [{}/{}], batch [{}], loader [{}] loss:{:.4f}'
                      .format(epoch + 1, num_epochs, idx, i, loss.item()))
                if epoch % 2 == 0:
                    loss_list.append(loss.item())
    return loss_list, encoder_model, cell_model


def test(encoder_moudle, cell_moudle, loader, seq):
    with torch.no_grad():
        for idx, (data, label) in enumerate(loader):
            data = data.to(device)
            if idx <= seq - 2:
                if idx == 0:
                    Quad = data
                else:
                    Quad = torch.cat((Quad, data), 0)
            if idx == seq - 1:
                Quad = torch.cat((Quad, data), 0)
                h = torch.zeros(seq - 1, 256).to(device)
                c = torch.zeros(seq - 1, 256).to(device)
                encoder_out = encoder_moudle(Quad)
                temp_target = encoder_out[0:1, :, :, :]
                temp_source = encoder_out[1:, :, :, :].view(seq - 1, 256)
                h, c = cell_moudle(temp_source, h, c)
                cell_out = h[-1].view(1, 256, 1, 1)
                feature = torch.cat((temp_target, cell_out), 1).to(device)
            if idx >= seq:
                h = torch.zeros(seq - 1, 256).to(device)
                c = torch.zeros(seq - 1, 256).to(device)
                Quad = Quad[1:, :, :, :]
                Quad = torch.cat((Quad, data), 0)
                encoder_out = encoder_moudle(Quad)
                temp_target = encoder_out[0:1, :, :, :]
                temp_source = encoder_out[1:, :, :, :].view(seq - 1, 256)

                h, c = cell_moudle(temp_source, h, c)
                cell_out = h[-1].view(1, 256, 1, 1)

                curr_feature = torch.cat((temp_target, cell_out), 1)
                feature = torch.cat((feature, curr_feature), 0)
        feature = feature.cpu().detach().view(feature.shape[0], feature.shape[1]).numpy()
        return feature


def gen_train_feature(encoder_moudle, cell_moudle, path_list, save_path, seq):
    for i in range(len(path_list)):
        curr_path = save_path + '/' + path_list[i][29:] + '.xlsx'
        temp_loader = testLoader(path_list[i], batch_size=1, shuffle=False, num_workers=8)
        feature = test(encoder_moudle, cell_moudle, temp_loader, seq)
        print('generating ' + curr_path)
        save_data_to_excel(feature, curr_path)


printGPU()
base_path = '/home/zlw/dataset/SAMM/train'
encoder_moudle_path = '/home/zlw/dataset/SAMM/moudle/encoder_moudle_40.pkl'
cell_moudle_path = '/home/zlw/dataset/SAMM/moudle/cell_moudle_40.pkl'
save_path = '/home/zlw/dataset/SAMM/train_feature'

learning_rate = 1e-4
batch_size = 4
num_workers = 8
num_epochs = 40

path_list = get_path(base_path)
loader_list = []
for i in range(len(path_list)):
    temp_loader = trainLoader(path_list[i], batch_size, False, num_workers)
    loader_list.append(temp_loader)

loss_list, encoder_moudle, cell_moudle = train(loader_list, learning_rate, num_epochs, batch_size)
torch.save(encoder_moudle, encoder_moudle_path)
torch.save(cell_moudle, cell_moudle_path)
print(str(datetime.datetime.now()) + ' moudle save successfully\n')



draw(loss_list)
