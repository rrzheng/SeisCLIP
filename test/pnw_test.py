import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import pandas as pd
import h5py
import random


def parse_waveform(trace_name, dtfl, start_index=4000):
    bucket, array = trace_name.split('$')
    x, y, z = iter([int(i) for i in array.split(',:')])
    data = dtfl.get(f"/data/{bucket}")[x, :y, :z]
    data = np.array(data)
    spec = data[:, start_index:(start_index + 6000)]
    if spec.shape[1] != 6000:
        print("spec: ", start_index, data.shape, spec.shape)
    spec = np.transpose(spec)
    return spec


def plot(data, label, idx):
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
            # spectrogram = (spectrogram - spectrogram.mean())/spectrogram.std()+1e-3
            spec[i, :] = np.abs(spectrogram).transpose(1, 0)
        return spec

    stft_data = z_norm(data)
    stft_data = cal_norm_spectrogram(stft_data)

    # 创建图形和子图
    figsize = np.array([20, 10]) * 0.4
    dpi = 500
    fig, axs = plt.subplots(3, 2, figsize=figsize, dpi=dpi, constrained_layout=True)

    # 绘制左侧的时间域波形
    tx = np.arange(data[:, 0].size) / 100
    for tdata, ax in zip(data.T, axs[:, 0]):
        ax.plot(tx, tdata, 'k')
        ax.set_ylim(-20, 20)
        ax.set_xlim(0, 60)
        ax.set_ylabel('Amp')
    axs[-1, 0].set_xlabel('t')

    # 绘制右侧的频谱图，并添加颜色条
    tx, ty = stft_data[0].shape
    tx = np.arange(tx) / 10
    ty = np.arange(ty)
    for i, (img_data, ax) in enumerate(zip(stft_data, axs[:, 1])):
        im = ax.pcolor(tx, ty, img_data.T, cmap='jet', vmin=0, vmax=4)
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 40)
        cbar = fig.colorbar(im, ax=ax, orientation='vertical')
        ax.set_ylabel('Frq')
    axs[-1, 1].set_xlabel('t')

    # 显示图形
    # plt.show()

    plt.savefig(f"{label}_{idx}.png")


comcat_meta_path = "/data/jyzheng/datasets/pnw/comcat_metadata.csv"
comcat_hdf5_path = "/data/jyzheng/datasets/pnw/comcat_waveforms.hdf5"
comcat_meta_df = pd.read_csv(comcat_meta_path)

exotic_meta_path = "/data/jyzheng/datasets/pnw/exotic_metadata.csv"
exotic_hdf5_path = "/data/jyzheng/datasets/pnw/exotic_waveforms.hdf5"
exotic_meta_df = pd.read_csv(exotic_meta_path)

labels = ["earthquake", "explosion", "surface event"]
label_dict = {label: float(idx) for idx, label in enumerate(labels)}

input_path = "/data/jyzheng/SeisCLIP/Data/classification_train.csv"
df = pd.read_csv(input_path)
chose_label = "surface event"
df = df[df["source_type"] == chose_label]
print(df["source_type"].value_counts())

# with h5py.File(comcat_hdf5_path, "r") as chr:
#     idx = random.randint(0, len(df))
#     row = df.iloc[idx]
#     p_arri = row["trace_P_arrival_sample"]
#     trace_name = row["trace_name"]

#
#     # if st == "explosion":
#     p_arri = int(p_arri)
#     start_index = (p_arri // 100 - 15) * 100
#     if start_index > 9000:
#         print("start_index exceed 9000", start_index)
#         start_index = 9000
#     data = parse_waveform(trace_name, chr, start_index)
#     plot(data, chose_label, idx)


with h5py.File(exotic_hdf5_path, "r") as ehr:
    idx = random.randint(0, len(df))
    print("idx: ", idx)
    row = df.iloc[idx]
    p_arri = row["trace_P_arrival_sample"]
    trace_name = row["trace_name"]

    if np.isnan(p_arri):
        p_arri = 5000
    else:
        p_arri = int(p_arri)

    gap_second = random.randint(5, 10)
    start_index = (p_arri // 100 - gap_second) * 100
    if start_index > 9000:
        print("start_index exceed 9000", start_index)
        start_index = 9000
    data = parse_waveform(trace_name, ehr, start_index)
    plot(data, chose_label, idx)




