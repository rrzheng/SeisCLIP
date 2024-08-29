import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import pandas as pd

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


def get_spec_station_info(jma_id: str):
    spec_path = f"/data/jyzheng/datasets/mech/npy_data/data/{jma_id}.npy"
    spec_data = np.load(spec_path)
    # station_info_path = f"/data/jyzheng/datasets/mech/npy_data/sta_info/{jma_id}.npy"
    # station_info = np.load(station_info_path)

    for i in range(spec_data.shape[0]):
        if np.max(spec_data[i]) < 100 and np.min(spec_data[i]) > -100:
            continue
        else:
            spec = np.transpose(spec_data[i, :, :6000])
            return spec
    return 0


def preprocess(input_path, ):
    df = pd.read_csv(input_path)
    print(df["fault_type"].value_counts())

    # label 词典
    labels = ['strike-slip', 'normal-fault', 'reverse-fault']
    chosen_label = labels[0]
    df = df[df["fault_type"] == chosen_label]

    idx = random.randint(0, len(df))
    row = df.iloc[idx]
    print("idx: ", idx)
    x = get_spec_station_info(row["jmaID"])
    plot(x, chosen_label, idx)


modes = ['train', 'val', 'test']
input_path = f"../Data/mech_train.csv"
preprocess(input_path)
