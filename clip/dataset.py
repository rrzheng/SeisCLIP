from torch.utils.data import Dataset
import numpy as np


class STEADDataset(Dataset):
    def __init__(self, spec_path, info_path):
        self.spec_array = np.load(spec_path)
        self.info_array = np.load(info_path)

    def __len__(self):
        return len(self.spec_array)

    def __getitem__(self, index):
        spec, info = self.spec_array[index], self.info_array[index]
        return spec, info

