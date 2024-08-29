import pandas as pd
from dataset import MechDataset
import numpy as np
from utils import z_norm, cal_norm_spectrogram

# dataset 信息
train_paths = ["/data/jyzheng/SeisCLIP/datasets/mech/train.npy"]
dataset = MechDataset(*train_paths)

# 原始数据
meta_path = f"../Data/mech_train.csv"
df = pd.read_csv(meta_path)
assert df.shape[0] == len(dataset)

for row_idx in range(df.shape[0]):
    mask, spec, stat_info, label = dataset[row_idx]

    jma_id = df.at[row_idx, "jmaID"]
    raw_spec = np.load(f"/data/jyzheng/datasets/mech/npy_data/data/{jma_id}.npy")
    raw_stat = np.load(f"/data/jyzheng/datasets/mech/npy_data/sta_info/{jma_id}.npy")

    j = 0
    for i in range(raw_spec.shape[0]):
        if np.max(raw_spec[i]) < 100 and np.min(raw_spec[i]) > -100:
            continue
        else:
            sp = np.transpose(raw_spec[i, :, :6000])
            sp = z_norm(sp)
            sp = cal_norm_spectrogram(sp)
            assert np.allclose(spec[j], sp)

            st = raw_stat[i, :2]
            assert np.allclose(stat_info[j], st)
            j += 1
            if j >= spec.shape[0]:
                break

    # for k in range(j, spec.shape[0]):
    #     assert np.all(spec[k] == 0.0)
    #     assert np.all(stat_info[k] == 0.0)









