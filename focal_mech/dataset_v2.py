import os
import sys
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import z_norm, cal_norm_spectrogram


class MechDataset(Dataset):
    def __init__(self,
                 path: str,
                 chunk_size: int = 1152130,
                 max_seq_len: int = 64,
                 spec_size: int = 18000,
                 stat_size: int = 2,
                 memmap_dtype=np.float64,
                 sample_rate=100,
                 window_length=20,
                 nfft=100):
        self._path = path
        self._chunk_size = chunk_size
        self._max_seq_len = max_seq_len
        self._spec_size = spec_size
        self._stat_size = stat_size
        assert chunk_size == (1 + max_seq_len * (spec_size + stat_size) + 1)
        self.dtype = memmap_dtype
        self._sample_rate = sample_rate
        self._window_length = window_length
        self._nfft = nfft
        file_len = self._get_file_length(path)
        self._data = np.fromfile(self._path, dtype=memmap_dtype).reshape(file_len, chunk_size)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def _get_file_length(self, path: str):
        item_size = self.dtype(0).itemsize
        file_size = os.stat(path).st_size
        return file_size // (item_size * self._chunk_size)

    def _extract_example_from_memmap(self, index: int) -> Tuple[int, np.ndarray, np.ndarray, int]:
        array = self._data[index]
        seq_len = array[0]
        spec = array[1:self._max_seq_len * self._spec_size + 1]
        spec = spec.reshape(self._max_seq_len, self._spec_size // 3, 3)
        idx = self._max_seq_len * self._spec_size + 1
        stat_info = array[idx:idx + (self._max_seq_len * self._stat_size)]
        stat_info = stat_info.reshape(self._max_seq_len, self._stat_size)
        label = array[-1]
        return int(seq_len), spec.copy(), stat_info.copy(), int(label)

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        seq_len, spec, stat_info, label = self._extract_example_from_memmap(index)
        spec = [z_norm(spec[idx]) for idx in range(spec.shape[0])]
        spec = [cal_norm_spectrogram(s, self._sample_rate, self._window_length, self._nfft) for s in spec]

        mask = [True] * seq_len + [False] * (self._max_seq_len - seq_len)
        mask = np.array(mask, dtype=np.bool_)

        return mask, np.array(spec, dtype=np.float32), np.array(stat_info, dtype=np.float32), label
