import torch
from torch import nn
import numpy as np


train_path = "/data/jyzheng/SeisCLIP/datasets/mech/train.npy"
data = np.fromfile(train_path, dtype=np.float64)
print(data.shape)



