import pandas as pd
import numpy as np
import datetime


def read_data_from_excel(path):
    print(datetime.datetime.now())
    print('read:', path)
    df = pd.read_excel(path, 'page_1')
    data = np.array(df)
    data = np.delete(data, 0, axis=1)
    print(datetime.datetime.now())
    print('done')
    return data

save_path = '/home/zlw/dataset/features1.xlsx'
features = read_data_from_excel(save_path)
print(features.shape)
