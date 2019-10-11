import numpy as np
import torch

a = np.array([1, 2, 3])
b = a[0:0]
c = torch.zeros(0, 512)
print(c)
print(c.shape[0])