import numpy as np
from scipy.signal import stft


def z_norm(x: np.ndarray) -> np.ndarray:
    """z-score 归一化"""
    for i in range(3):
        x_std = x[:, i].std() + 1e-3
        x[:, i] = (x[:, i] - x[:, i].mean()) / x_std
    return x


def cal_norm_spectrogram(x: np.ndarray,
                         sample_rate=100,
                         window_length=20,
                         nfft=100) -> np.ndarray:
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
