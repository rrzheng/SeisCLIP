from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
import os

from utils import get_bytes_range
from scipy.signal import stft


class STEADDataset(Dataset):
    def __init__(self,
                 *paths: str,
                 chunk_size: int = 18008,
                 spec_size: int = 18000,
                 info_size: int = 8,
                 memmap_dtype=np.float32,
                 sample_rate=100,
                 window_length=20,
                 nfft=100):
        self._paths = paths
        self._chunk_size = chunk_size
        self._spec_size = spec_size
        self._info_size = info_size
        assert chunk_size == (spec_size + info_size)
        self.dtype = memmap_dtype
        self._sample_rate = sample_rate
        self._window_length = window_length
        self._nfft = nfft
        self._mmap_offsets: List[Tuple[int, int]] = self._offsets()

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def _offsets(self) -> List[Tuple[int, int]]:
        import concurrent.futures

        mmap_offsets = []
        path_to_length: Dict[str, int] = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            path_futures = []
            for i, path in enumerate(self._paths):
                path_futures.append(executor.submit(self._get_file_length, path))

            for future in concurrent.futures.as_completed(path_futures):
                path, length = future.result()
                path_to_length[path] = length

        start_offset = 0
        for path in self._paths:
            length = path_to_length[path]
            end_offset = start_offset + length
            mmap_offsets.append((start_offset, end_offset))
            start_offset += length
        return mmap_offsets

    def _get_file_length(self, path: str):
        item_size = self.dtype(0).itemsize
        file_size = os.stat(path).st_size
        return path, file_size // (item_size * self._chunk_size)

    def _read_chunk_from_memmap(self, path: str, index: int) -> Tuple[np.ndarray, np.ndarray]:
        item_size = self.dtype(0).itemsize
        bytes_start = index * item_size * self._chunk_size
        num_bytes = item_size * self._chunk_size
        buffer = get_bytes_range(path, bytes_start, num_bytes)
        array = np.frombuffer(buffer, dtype=self.dtype)
        spec = array[:self._spec_size]
        spec = spec.reshape(self._spec_size // 3, 3)
        info = array[self._spec_size:]
        return info.copy(), spec.copy()

    def __len__(self):
        return self._mmap_offsets[-1][1]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_index = index if index >= 0 else len(self) + index

        # 标识所属文件的索引
        memmap_index: Optional[int] = None
        # 标识在该文件中的样本索引
        memmap_local_index: Optional[int] = None

        for i, (offset_start, offset_end) in enumerate(self._mmap_offsets):
            if offset_start <= pos_index < offset_end:
                memmap_index = i
                memmap_local_index = pos_index - offset_start

        if memmap_index is None or memmap_local_index is None:
            raise IndexError(f"{index} is out of bounds for dataset of size {len(self)}")

        info, spec = self._read_chunk_from_memmap(self._paths[memmap_index], memmap_local_index)
        spec = self._z_norm(spec)
        spec = self._cal_norm_spectrogram(spec)
        return torch.tensor(info, dtype=torch.float64), torch.tensor(spec, dtype=torch.float64)

    def _z_norm(self, x: np.ndarray) -> np.ndarray:
        """z-score 归一化"""
        for i in range(3):
            x_std = x[:, i].std() + 1e-3
            x[:, i] = (x[:, i] - x[:, i].mean()) / x_std
        return x

    def _cal_norm_spectrogram(self, x: np.ndarray) -> np.ndarray:
        spec = np.zeros([3, int(x.shape[0] / self._window_length * 2), int(self._nfft / 2)])
        for i in range(3):
            _, _, spectrogram = stft(x[:, i],
                                     fs=self._sample_rate,
                                     window='hann',
                                     nperseg=self._window_length,
                                     noverlap=int(self._window_length / 2),
                                     nfft=self._nfft,
                                     boundary='zeros')
            spectrogram = spectrogram[1:, 1:]
            spec[i, :] = np.abs(spectrogram).transpose(1, 0)
        return spec
