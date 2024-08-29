from typing import Tuple

import h5py
import numpy as np
import pandas as pd

from utils import get_bytes_range
import re
from stead_dataloader import read_stead_data, stead_loader
from memmap_dataset import STEADDataset
import torch
from glob import glob

# def read_chunk_from_memmap(path: str, index: int) -> Tuple[np.ndarray, np.ndarray]:
#     _chunk_size = 18008
#     _spec_size = 18000
#     dtype = np.float32
#     item_size = dtype(0).itemsize
#     bytes_start = index * item_size * _chunk_size
#     num_bytes = item_size * _chunk_size
#     buffer = get_bytes_range(path, bytes_start, num_bytes)
#     array = np.frombuffer(buffer, dtype=dtype)
#     spec = array[:_spec_size]
#     spec = spec.reshape(_spec_size // 3, 3)
#     info = array[_spec_size:]
#     return info.copy(), spec.copy()


# def norm_text(info_array):
#     conda_text = info_array[-1]
#     # 定义多个分隔符
#     separators = ['[[', "."]
#
#     # 使用多个分隔符对字符串进行分割并保留所有子字符串
#     pattern = "|".join(map(re.escape, separators))
#     result = re.split(pattern, conda_text)
#
#     info_array[-1] = int(result[1])
#     y = np.array(info_array, np.float32)
#
#     y[0] = y[0] / 6000
#     y[2] = y[2] / 60
#     y[3] = y[3] / 6000
#     y[5] = y[5] / 300
#     y[6] = y[6] / 360
#     y[7] = y[7] / 6000
#     mask = np.isnan(y)
#     y[mask] = 0
#     return y


# def read_from_hdf5(csv_path: str, hdf5_path: str, evi: str) -> Tuple[np.ndarray, np.ndarray]:
#     df = pd.read_csv(csv_path, low_memory=False)
#     df = df[df['trace_name'] == evi]
#     # print(df)
#     info_array = [
#         df['p_arrival_sample'].values[0],
#         df['p_weight'].values[0],
#         df['p_travel_sec'].values[0],
#         df['s_arrival_sample'].values[0],
#         df['s_weight'].values[0],
#         df['source_distance_km'].values[0],
#         df['back_azimuth_deg'].values[0],
#         df['coda_end_sample'].values[0]
#     ]
#     info = norm_text(info_array)
#
#     with h5py.File(hdf5_path, 'r') as dtfl:
#         spec = dtfl.get('data/' + evi)
#         spec = np.array(spec)
#
#     return info.copy(), spec.copy()


csv_path = "/data/jyzheng/datasets/stead/chunk2.csv"
hdf5_path = "/data/jyzheng/datasets/stead/chunk2.hdf5"
# evi_list= ['B084.PB_20120926034523_EV', 'ARG.HL_20040829211506_EV', 'AMD.SN_20080525222927_EV',
#             'B084.PB_20161007013105_EV', 'B082.PB_20170728111054_EV', 'B082.PB_20130602084143_EV',
#             'B086.PB_20150715060109_EV', 'B084.PB_20170118161412_EV', 'B084.PB_20080811131038_EV',
#             'B084.PB_20110409053005_EV']
#
# for idx, evi in enumerate(evi_list):
#     info, spec = read_from_hdf5(csv_path, hdf5_path, evi)
#     # print(info, spec)
#
#     part_path = "/data/jyzheng/SeisCLIP/datasets/test/0.npy"
#     info1, spec1 = read_chunk_from_memmap(part_path, idx)
#     print(np.array_equal(info, info1))
#     print(np.array_equal(spec, spec1))


csv_stead = read_stead_data(csv_path)
train_dataset = stead_loader(csv_stead, hdf5_path, window_length=20)

train_paths = glob("/data/jyzheng/SeisCLIP/datasets/test/train/part-00*.npy")
print(train_paths)
train_dataset1 = STEADDataset(*train_paths, window_length=20)

# info, spec = train_dataset[0]
# info1, spec1 = train_dataset1[0]
# print(len(train_dataset1))
# print(torch.allclose(info, info1))
# print(torch.allclose(spec, spec1, atol=1e-6))

print(len(train_dataset1))
# assert len(train_dataset) == len(train_dataset1)
for idx in range(len(train_dataset1)):
    info, spec = train_dataset[idx]
    info1, spec1 = train_dataset1[idx]
    # print(info)
    # print(info1)
    assert torch.allclose(info, info1)
    assert torch.allclose(spec, spec1, atol=1e-6)


