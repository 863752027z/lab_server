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
        transforms.ToTensor()])

    data_set = datasets.ImageFolder(file_path,
                                    transform=data_transform)

    test_loader = Data.DataLoader(dataset=data_set,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   num_workers=num_workers)
    return test_loader


def save_data_to_excel(data, path):
    print(datetime.datetime.now())
    print('generating:', path)
    print(data.shape)

    data_df = pd.DataFrame(data)
    writer = pd.ExcelWriter(path)
    data_df.to_excel(writer, 'page_1', float_format='%.5f') # float_format 控制精度
    writer.save()

    print('done')

def show(loader):
    for i, (data, label) in enumerate (loader):
        for j in range(data.shape[0]):
            print(i, j)
            img = data[j].numpy()
            img = img.transpose((1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('img', img)
            cv2.waitKey(1)


def gen_train_feature(encoder_moudle, cell_moudle, path_list, save_path, batch_size, num_workers):
    for i in range(len(path_list)):
        i = i+3
        curr_path = save_path + '/' + path_list[i][29:] + '.xlsx'
        temp_loader = testLoader(path_list[i], batch_size, False, num_workers)
        show(temp_loader)
        print('generating ' + curr_path)


save_path = '/home/zlw/dataset/SAMM/train_feature'
base_path = '/home/zlw/dataset/SAMM/train'
path_list = get_path(base_path)
encoder_moudle = None
cell_moudle = None
batch_size = 4
num_workers = 8
gen_train_feature(encoder_moudle, cell_moudle, path_list, save_path, batch_size, num_workers)