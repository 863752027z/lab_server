import os
import torch.utils.data as Data
from torchvision import transforms, datasets


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


batch_size = 4
num_workers = 8
base_path = '/home/zlw/dataset/SAMM/train'
path_list = get_path(base_path)
for i in range(len(path_list)):
    temp_train = testLoader(path_list[i], batch_size, False, num_workers)
    print(temp_train)
