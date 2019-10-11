import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

print(torch.cuda.is_available())
print('hello pytorch' + torch.__version__)
a = torch.rand(2, 2, requires_grad=True)
print(a)
b = torch.rand(2, 2, requires_grad=True)
print(b)

print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(0))

'''
while True:
    c = torch.mm(a, b)
    print(c)
'''