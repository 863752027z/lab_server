import os
import cv2
import torch
import datetime
import torch.utils.data as Data
from torchvision import transforms, datasets
import torch.nn as nn
import numpy as np
import pandas as pd


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


def testLoader(file_path, batch_size, shuffle, num_workers):
    data_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    data_set = datasets.ImageFolder(file_path,
                                    transform=data_transform)
    test_loader = Data.DataLoader(dataset=data_set,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   num_workers=num_workers)
    return test_loader


def test(test_loader, seq, encoder_moudle, cell_moudle):
    with torch.no_grad():
        for idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            if idx <= seq-2:
                if idx == 0:
                    Quad = data
                else:
                    Quad = torch.cat((Quad, data), 0)
            if idx == seq-1:
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
                h = torch.zeros(seq-1, 256).to(device)
                c = torch.zeros(seq-1, 256).to(device)
                Quad = Quad[1:, :, :, :]
                Quad = torch.cat((Quad, data), 0)
                encoder_out = encoder_moudle(Quad)
                temp_target = encoder_out[0:1, :, :, :]
                temp_source = encoder_out[1:, :, :, :].view(seq-1, 256)

                h, c = cell_moudle(temp_source, h, c)
                cell_out = h[-1].view(1, 256, 1, 1)

                curr_feature = torch.cat((temp_target, cell_out), 1)
                feature = torch.cat((feature, curr_feature), 0)
        feature = feature.cpu().detach().view(feature.shape[0], feature.shape[1]).numpy()
        return feature


base_path1 = 'F:/SAMM_FACE_CUT_1/SAMM/test/micro'
base_path2 = 'F:/SAMM_FACE_CUT_1/SAMM/test/no_micro'
base_path3 = 'F:/SAMM_FACE_CUT_1/SAMM/feature/micro'
base_path4 = 'F:/SAMM_FACE_CUT_1/SAMM/feature/no_micro'
moudle_path = 'F:/moudle/'
micro_feature = 'F:/SAMM_FACE_CUT_1/SAMM/feature/micro/'
nomicro_feature = 'F:/SAMM_FACE_CUT_1/SAMM/feature/no_micro/'
cell_moudle = torch.load(moudle_path + 'cell_modle256.pkl').to(device)
encoder_moudle = torch.load(moudle_path + 'encoder_modle256.pkl').to(device)
test_list1 = get_path(base_path1)
test_list2 = get_path(base_path2)
print(test_list1)
print(test_list2)
batch_size = 1
num_workers = 0
seq = 4


for i in range(len(test_list1)):
    print('处理' + test_list1[i])
    save_path = base_path3 + '/' + test_list1[i][35:] + '.xlsx'
    print(save_path)
    temp_loader = testLoader(test_list1[i], batch_size, False, num_workers)
    feature = test(temp_loader, seq, encoder_moudle, cell_moudle)
    '''
    save_data_to_excel(feature, save_path)
    save_path = ''
    '''


for i in range(len(test_list2)):
    print('处理' + test_list2[i])
    save_path = base_path4 + '/' + test_list2[i][38:] + '.xlsx'
    print(save_path)
    temp_loader = testLoader(test_list2[i], batch_size, False, num_workers)
    feature = test(temp_loader, seq, encoder_moudle, cell_moudle)
    '''
    save_data_to_excel(feature, save_path)
    save_path = ''
    '''