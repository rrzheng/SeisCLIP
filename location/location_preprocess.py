import pandas as pd
import numpy as np
from typing import Tuple
from utils import DEP_MIN, DEP_MAX, MAG_MIN, MAG_MAX
from utils import min_max_scale, MinMaxCoordinate


def get_spec_station_info(jma_id: str) -> Tuple[int, MinMaxCoordinate, np.ndarray]:
    """
    解析波形数据、station 数据
    返回结果：序列长度, concat(波形, station 位置)
    """
    spec_path = f"/data/jyzheng/datasets/location/spec/{jma_id}.npy"
    spec_data = np.load(spec_path)
    station_info_path = f"/data/jyzheng/datasets/mech/label_station/sta_info/{jma_id}.npy"
    station_info = np.load(station_info_path)
    spec_arr = []
    sta_lat_arr = []
    sta_lon_arr = []
    max_seq_len = 32

    # 剔除无数据的波形样本
    if spec_data.shape[0] == 0:
        return 0, MinMaxCoordinate(0, 0, 0, 0), np.empty(1 + 32 * (18000 + 2))

    for i in range(spec_data.shape[0]):
        # 剔除全为0 的波形样本
        if np.all(np.abs(spec_data[i]) < 1e-1):
            continue
        else:
            spec = np.transpose(spec_data[i, :, :6000])
            spec = np.reshape(spec, spec.shape[0] * spec.shape[1])
            spec_arr.append(spec)
            sta_lat_arr.append(station_info[i, 1])
            sta_lon_arr.append(station_info[i, 0])
            if len(spec_arr) == max_seq_len:
                break

    # 剔除无波形的样本
    seq_len = len(spec_arr)
    if seq_len == 0:
        return 0, MinMaxCoordinate(0, 0, 0, 0), np.empty(1 + 32 * (18000 + 2))

    # 经纬度相对归一化
    lat_min = 0.87 * min(sta_lat_arr)
    lat_max = 1.033 * max(sta_lat_arr)
    lon_min = 0.98 * min(sta_lon_arr)
    lon_max = 1.027 * max(sta_lon_arr)
    sta_arr = []
    for i in range(seq_len):
        sta_lat = min_max_scale(sta_lat_arr[i], lat_min, lat_max)
        sta_long = min_max_scale(sta_lon_arr[i], lon_min, lon_max)
        sta_arr.append([sta_lat, sta_long])

    spec_arr = np.array(spec_arr)
    spec_arr = np.reshape(spec_arr, spec_arr.shape[0] * spec_arr.shape[1])
    sta_arr = np.array(sta_arr)
    sta_arr = np.reshape(sta_arr, sta_arr.shape[0] * sta_arr.shape[1])

    # padding
    if seq_len < max_seq_len:
        spec_pad = np.zeros((max_seq_len - seq_len) * 18000, dtype=np.float32)
        sta_pad = np.zeros((max_seq_len - seq_len) * 2, dtype=np.float32)
        spec_arr = np.concatenate((spec_arr, spec_pad))
        sta_arr = np.concatenate((sta_arr, sta_pad))
    x = np.concatenate(([seq_len], spec_arr, sta_arr), dtype=np.float32)

    return seq_len, MinMaxCoordinate(lat_min, lat_max, lon_min, lon_max), x


def preprocess(input_path, output_path):
    """
    处理样本
    返回结果格式：有效输入特征序列长度N_s, 波形(64, 6000, 3), station 数据(64, 2), 目标值(4,)
    """

    print("preprocessing file: ", input_path)
    df = pd.read_csv(input_path)
    result = []
    for _, row in df.iterrows():
        seq_len, mmc, x = get_spec_station_info(row["jmaID"])
        if seq_len == 0:
            continue
        else:
            lat = min_max_scale(float(row["lat"][:-1]), mmc.lat_min, mmc.lat_max)
            lon = min_max_scale(float(row["long"][:-1]), mmc.lon_min, mmc.lon_max)
            dep = min_max_scale(float(row["dep"][:-2]), DEP_MIN, DEP_MAX)
            mag = min_max_scale(float(row["m"]), MAG_MIN, MAG_MAX)
            result.append(np.concatenate((x, [mmc.lat_min, mmc.lat_max, mmc.lon_min, mmc.lon_max],
                                          [lat, lon, dep, mag])))

    print("num of examples: ", len(result))
    result = np.array(result, dtype=np.float32)
    result = np.reshape(result, result.shape[0] * result.shape[1])
    print("result shape: ", result.shape)
    result.tofile(output_path)


def main():
    modes = ['train', 'val', 'test']
    for mode in modes:
        input_path = f"../Data/location_{mode}.csv"
        output_path = f"/data/jyzheng/SeisCLIP/datasets/location_v4/{mode}.npy"
        preprocess(input_path, output_path)


if __name__ == "__main__":
    main()
