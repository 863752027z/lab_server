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
            ('BatchNorm5', nn.BatchNorm2d(512)),
            ('LeakyReLU5', nn.LeakyReLU(0.2, True)),

            ('Con6', nn.Conv2d(512, 256, 4, stride=2, padding=1)),
            ('BatchNorm6', nn.BatchNorm2d(512)),
            ('LeakyReLU6', nn.LeakyReLU(0.2, True)),

            ('Con7', nn.Conv2d(256, 256, 4, stride=2, padding=1)),
            ('BatchNorm7', nn.BatchNorm2d(512)),
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


def train(train_loader, learning_rate, num_epochs):
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
        for idx, (data, label) in enumerate(train_loader):
            h = torch.zeros(4, 256).to(device)
            c = torch.zeros(4, 256).to(device)
            data = data.to(device)
            # =========forward===========
            encoder_output = encoder_model(data)
            encoder_output = encoder_output.view((encoder_output.shape[0], encoder_output.shape[1]))    #4*256
            target = encoder_output[0].view(1, encoder_output.shape[1], 1, 1)     #1*256*1*1
            source = encoder_output[1:].view(3, encoder_output.shape[1])    #3*256

            h, c = cell_model(source, h, c)
            cell_output = h[-1].view(1, 256, 1, 1)

            decoder_input = torch.cat((target, cell_output), 1)
            decoder_output = decoder_model(decoder_input)

            print('decoder_output', decoder_output.shape)
            loss = criterion(decoder_output, data[0])
            # =========backward=========
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ============log===========
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, num_epochs, loss.item()))
            loss_list.append(loss.item())
            ''''
            if epoch == 95:
                save_path = '/home/zlw/dataset/out1/'
                curr_count = decoder_output.shape[0]  # might be 8
                for i in range(curr_count):
                    curr_idx = i + idx * batch_size
                    temp_input = data[i]
                    temp_output = decoder_output[i]
                    temp_input = temp_input.cpu().detach().numpy()
                    temp_output = temp_output.cpu().detach().numpy()
                    temp_input = temp_input.transpose(1, 2, 0)
                    temp_output = temp_output.transpose(1, 2, 0)
                    temp_input = cv2.cvtColor(temp_input, cv2.COLOR_RGB2BGR)
                    temp_output = cv2.cvtColor(temp_output, cv2.COLOR_RGB2BGR)
                    temp_img = np.concatenate([temp_input, temp_output]) * 255
                    cv2.imwrite(save_path + str(curr_idx) + '.jpg', temp_img)
                    print(save_path + str(curr_idx) + '.jpg')
            '''
    return loss_list, encoder_model, cell_model


def test(test_loader, encoder_model, cell_model):
    for idx, (data, label) in enumerate(test_loader):
        h = torch.zeros(4, 512).to(device)
        c = torch.zeros(4, 512).to(device)
        data = data.to(device)
        with torch.no_grad():
            encoder_output = encoder_model(data)
            encoder_output = encoder_output.view((encoder_output.shape[0], encoder_output.shape[1]))
            h, c = cell_model(encoder_output, h, c)
        print(h.shape)
        if idx == 0:
            temp_feature = h
        else:
            temp_feature = torch.cat((temp_feature, h), 0)
    return temp_feature.cpu().detach().numpy()


printGPU()
log_file = open('/home/zlw/PycharmProjects/pycharm.txt', 'a+')
num_epochs = 1
batch_size = 4
learning_rate = 1e-3
train_path = '/home/zlw/dataset/SAMM_FACE_CUT/SAMM'
test_path = '/home/zlw/dataset/SAMM_FACE_CUT/MicroExpress'
save_path1 = '/home/zlw/dataset/SAMM_NOExpress.xlsx'
save_path2 = '/home/zlw/dataset/SAMM_Express.xlsx'
encoder_modle_path = '/home/zlw/PycharmProjects/encoder_modle.pkl'
cell_model_path = '/home/zlw/PycharmProjects/cell_modle.pkl'

train_loader = trainLoader(train_path, batch_size, False, 8)
lose_list, encoder_model, cell_model = train(train_loader, learning_rate, num_epochs)

torch.save(encoder_model, encoder_modle_path)
torch.save(cell_model, cell_model_path)
print('save successfully!')
log_file.write(str(datetime.datetime.now()) + '\n' + 'save successfully!\n')

load_encoder = torch.load(encoder_modle_path)
load_cell = torch.load(cell_model_path)
print('load successfully!')
log_file.write(str(datetime.datetime.now()) + '\n' + 'load successfully!\n')

test_loader = testLoader(test_path, batch_size, False, 8)
feature = test(test_loader, load_encoder, load_cell)
print('save features')
log_file.write(str(datetime.datetime.now()) + '\n' + 'save features\n')

save_data_to_excel(feature, save_path1)
log_file.write(str(datetime.datetime.now()) + '\n' + 'done!\n')

log_file.close()
