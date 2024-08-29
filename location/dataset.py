import os
import sys
from typing import List, Tuple, Optional, Dict

import numpy as np
from torch.utils.data import Dataset

sys.path.append("../")
sys.path.append("../pretrain/")

from pretrain.utils import get_bytes_range
from pretrain.utils import z_norm, cal_norm_spectrogram


class LocationDataset(Dataset):
    def __init__(self,
                 *paths: str,
                 chunk_size: int = 576073,
                 max_seq_len: int = 32,
                 spec_size: int = 18000,
                 stat_size: int = 2,
                 memmap_dtype=np.float32,
                 sample_rate=100,
                 window_length=20,
                 nfft=100):
        self._paths = paths
        self._chunk_size = chunk_size
        self._max_seq_len = max_seq_len
        self._spec_size = spec_size
        self._stat_size = stat_size
        assert chunk_size == (1 + max_seq_len * (spec_size + stat_size) + 4 + 4)
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

    def _read_chunk_from_memmap(self, path: str, index: int) \
            -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        item_size = self.dtype(0).itemsize
        bytes_start = index * item_size * self._chunk_size
        num_bytes = item_size * self._chunk_size
        buffer = get_bytes_range(path, bytes_start, num_bytes)
        array = np.frombuffer(buffer, dtype=self.dtype)
        seq_len = array[0]
        spec = array[1:self._max_seq_len * self._spec_size + 1]
        spec = spec.reshape(self._max_seq_len, self._spec_size // 3, 3)
        idx = self._max_seq_len * self._spec_size + 1
        stat_info = array[idx:idx + (self._max_seq_len * self._stat_size)]
        stat_info = stat_info.reshape(self._max_seq_len, self._stat_size)
        mmc = array[-8:-4]
        y = array[-4:]
        return (int(seq_len), spec.copy(), stat_info.copy(),
                mmc.copy(), y.copy())

    def __len__(self):
        return self._mmap_offsets[-1][1]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        seq_len, spec, stat_info, mmc, target = self._read_chunk_from_memmap(self._paths[memmap_index],
                                                                             memmap_local_index)
        spec = [z_norm(spec[idx]) for idx in range(spec.shape[0])]
        spec = [cal_norm_spectrogram(s, self._sample_rate, self._window_length, self._nfft) for s in spec]

        mask = [True] * seq_len + [False] * (self._max_seq_len - seq_len)
        mask = np.array(mask, dtype=np.bool_)

        return mask, np.array(spec, dtype=np.float32), np.array(stat_info, dtype=np.float32), mmc, target
