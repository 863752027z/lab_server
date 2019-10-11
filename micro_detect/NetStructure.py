import torch.nn as nn
from collections import OrderedDict


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