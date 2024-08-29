import pandas as pd
import logging
import numpy as np
from scipy.signal import stft
import re
import h5py
import os

log = logging.getLogger(__name__)


def z_norm(x):
    """z-score 归一化"""
    for i in range(3):
        x_std = x[:, i].std() + 1e-3
        x[:, i] = (x[:, i] - x[:, i].mean()) / x_std
    return x


def cal_norm_spectrogram(x, sample_rate=100, window_length=20, nfft=100):
    spec = np.zeros([3, int(x.shape[0] / window_length * 2), int(nfft / 2)])
    for i in range(3):
        _, _, spectrogram = stft(x[:, i],
                                 fs=sample_rate,
                                 window='hann',
                                 nperseg=window_length,
                                 noverlap=int(window_length / 2),
                                 nfft=nfft,
                                 boundary='zeros')
        spectrogram = spectrogram[1:, 1:]
        spec[i, :] = np.abs(spectrogram).transpose(1, 0)
    return spec


def norm_text(selected_signal):
    string = selected_signal['coda_end_sample']
    # 定义多个分隔符
    separators = ['[[', "."]

    # 使用多个分隔符对字符串进行分割并保留所有子字符串
    pattern = "|".join(map(re.escape, separators))
    result = re.split(pattern, string)

    selected_signal['coda_end_sample'] = int(result[1])
    y = np.array(selected_signal.values, dtype='float')
    # normalize P_sample,p_travel,s_sample,source_distance,azimuth, coda and sample
    y[0] = y[0] / 6000
    y[2] = y[2] / 60
    y[3] = y[3] / 6000
    y[5] = y[5] / 300
    y[6] = y[6] / 360
    y[7] = y[7] / 6000
    y = np.nan_to_num(y, nan=0.0)
    return y


def process_stead(csv_path, hdf5_path, save_dir):
    meta_df = pd.read_csv(csv_path, low_memory=False)
    log.info(f'total events in csv file: {len(meta_df)}')
    meta_df = meta_df[(meta_df.trace_category == 'earthquake_local')]
    log.info(f'total events selected: {len(meta_df)}')
    col_names = ['p_arrival_sample', 'p_weight', 'p_travel_sec', 's_arrival_sample',
                 's_weight', 'source_distance_km', 'back_azimuth_deg', 'coda_end_sample']

    h5_data = h5py.File(hdf5_path, 'r')

    spec_array = []
    info_array = []
    for idx in range(len(meta_df)):
        row = meta_df.iloc[idx]
        trace_name = row.iloc[-1]
        data = h5_data.get('data/' + trace_name)
        data = np.array(data)
        data = z_norm(data)
        spec = cal_norm_spectrogram(data)
        info = norm_text(row[col_names])
        spec_array.append(spec)
        info_array.append(info)

    spec_array = np.array(spec_array)
    info_array = np.array(info_array)

    np.save(os.path.join(save_dir, 'spec.npy'), spec_array)
    np.save(os.path.join(save_dir, 'info.npy'), info_array)


if __name__ == '__main__':
    # csv_path = '/data/jyzheng/datasets/stead/chunk2.csv'
    # hdf5_path = '/data/jyzheng/datasets/stead/chunk2.hdf5'
    # save_dir = '/data/jyzheng/SeisCLIP/datasets/stead/'
    # process_stead(csv_path, hdf5_path, save_dir)

    spec_path = "/data/jyzheng/SeisCLIP/datasets/stead/spec.npy"
    data = np.load(spec_path)
    print(data.shape)
    print(data[0])
