import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

file_name = "/data/jyzheng/datasets/stead/chunk2.hdf5"
csv_file = "/data/jyzheng/datasets/stead/chunk2.csv"

# reading the csv file into a dataframe:
df = pd.read_csv(csv_file)
print(f'total events in csv file: {len(df)}')
# filterering the dataframe
df = df[(df.trace_category == 'earthquake_local') & (df.source_distance_km <= 20) & (df.source_magnitude > 3)]
print(f'total events selected: {len(df)}')

# making a list of trace names for the selected data
ev_list = df['trace_name'].to_list()


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


# retrieving selected waveforms from the hdf5 file:
dtfl = h5py.File(file_name, 'r')

# tidx = np.random.randint(268)
tidx = 147
evi = ev_list[tidx]
print(tidx)
x = dtfl.get('data/' + str(evi))
# waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
data = np.array(x)

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
plt.show()

# plt.savefig('stft_%d.png' % tidx)

# info = [x.attrs['p_arrival_sample'],
#         x.attrs['p_weight'],
#         x.attrs['p_travel_sec'],
#         x.attrs['s_arrival_sample'],
#         x.attrs['s_weight'],
#         x.attrs['source_distance_km'],
#         x.attrs['back_azimuth_deg'],
#         x.attrs['coda_end_sample'][0][0]
#         ]
# print(info)
