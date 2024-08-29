import pandas as pd
from dataset import LocationDataset
import numpy as np
import sys
from utils import LAT_MIN, LAT_MAX, LONG_MIN, LONG_MAX, DEP_MIN, DEP_MAX, MAG_MIN, MAG_MAX
from utils import inverse_min_max_scale

sys.path.append("../")
sys.path.append("../pretrain/")

from pretrain.utils import z_norm, cal_norm_spectrogram

mode = "test"

# dataset 信息
train_paths = [f"/data/jyzheng/SeisCLIP/datasets/location_v4/{mode}.npy"]
dataset = LocationDataset(*train_paths)
print("len of dataset: ", len(dataset))

# 原始数据
meta_path = f"../Data/location_{mode}.csv"
df = pd.read_csv(meta_path)

ds_idx = 0
for row_idx in range(df.shape[0]):
    mask, spec, stat_info, mmc, target = dataset[ds_idx]
    assert np.min(stat_info) >= 0.0 and np.max(stat_info) <= 1.0
    assert np.min(target) >= 0 and np.max(target) <= 1.0
    # print("mmc: ", mmc)
    # print("target: ", target)
    # print(df.at[row_idx, "long"][:-1])

    jma_id = df.at[row_idx, "jmaID"]
    raw_spec = np.load(f"/data/jyzheng/datasets/location/spec/{jma_id}.npy")
    raw_stat = np.load(f"/data/jyzheng/datasets/mech/label_station/sta_info/{jma_id}.npy")

    if raw_spec.shape[0] == 0:
        continue

    j = 0
    for i in range(raw_spec.shape[0]):
        if np.all(np.abs(raw_spec[i]) < 1e-1):
            continue
        else:
            sp = np.transpose(raw_spec[i, :, :6000])
            sp = z_norm(sp)
            sp = cal_norm_spectrogram(sp)
            try:
                assert np.allclose(spec[j], sp, atol=1e-6)
            except AssertionError as e:
                print(jma_id, spec[j], sp)
                exit()

            st = raw_stat[i, :2]
            lat = inverse_min_max_scale(stat_info[j][0], mmc[0], mmc[1])
            lon = inverse_min_max_scale(stat_info[j][1], mmc[2], mmc[3])
            ds_st = np.array([lon, lat])
            # print(ds_st, st)
            assert np.allclose(ds_st, st)
            j += 1
            if j >= spec.shape[0]:
                break

    # 判断raw_spec 是否全为0，若为此则跳过后面的判断
    if j > 0:
        ds_idx += 1
    elif j == 0:
        continue

    dt_lat = inverse_min_max_scale(target[0], mmc[0], mmc[1])
    dt_lon = inverse_min_max_scale(target[1], mmc[2], mmc[3])
    dt_dep = inverse_min_max_scale(target[2], DEP_MIN, DEP_MAX)
    dt_mag = inverse_min_max_scale(target[3], MAG_MIN, MAG_MAX)
    dt_target = np.array([dt_lat, dt_lon, dt_dep, dt_mag])

    lat = float(df.at[row_idx, "lat"][:-1])
    long = float(df.at[row_idx, "long"][:-1])
    dep = float(df.at[row_idx, "dep"][:-2])
    mag = float(df.at[row_idx, "m"])
    true_target = np.array([lat, long, dep, mag])

    try:
        assert np.allclose(dt_target, true_target)
    except AssertionError as e:
        print("target not equal: ", jma_id, dt_target, true_target)
        print(e)
    # break

