import numpy as np
import pandas as pd


def get_spec_station_info(jma_id: str):
    """
    解析波形数据、station 数据
    返回结果：序列长度、波形、station数据
    """
    spec_path = f"/data/jyzheng/datasets/mech/npy_data/data/{jma_id}.npy"
    spec_data = np.load(spec_path)
    station_info_path = f"/data/jyzheng/datasets/mech/npy_data/sta_info/{jma_id}.npy"
    station_info = np.load(station_info_path)
    spec_arr = []
    sta_arr = []
    max_seq_len = 64
    for i in range(spec_data.shape[0]):
        if np.max(spec_data[i]) < 100 and np.min(spec_data[i]) > -100:
            continue
        else:
            spec = np.transpose(spec_data[i, :, :6000])
            spec = np.reshape(spec, spec.shape[0]*spec.shape[1])
            spec_arr.append(spec)
            sta_arr.append(station_info[i, :2])
            if len(spec_arr) == max_seq_len:
                break

    seq_len = len(spec_arr)
    spec_arr = np.array(spec_arr)
    spec_arr = np.reshape(spec_arr, spec_arr.shape[0]*spec_arr.shape[1])
    sta_arr = np.array(sta_arr)
    sta_arr = np.reshape(sta_arr, sta_arr.shape[0]*sta_arr.shape[1])
    # padding
    if seq_len < max_seq_len:
        spec_pad = np.zeros((max_seq_len - seq_len)*18000, dtype=np.float64)
        sta_pad = np.zeros((max_seq_len - seq_len)*2, dtype=np.float64)
        spec_arr = np.concatenate((spec_arr, spec_pad))
        sta_arr = np.concatenate((sta_arr, sta_pad))
    x = np.concatenate(([seq_len], spec_arr, sta_arr), dtype=np.float64)

    return x


def preprocess(input_path, output_path):
    """
    处理样本
    返回结果格式：有效输入特征序列长度N_s, 波形(64, 6000, 3), station 数据(64, 2), 类别
    """
    df = pd.read_csv(input_path)
    print(df["fault_type"].value_counts())

    # label 词典
    labels = ['strike-slip', 'normal-fault', 'reverse-fault']
    label_dict = {label: float(idx) for idx, label in enumerate(labels)}

    result = []
    for _, row in df.iterrows():
        x = get_spec_station_info(row["jmaID"])
        y = label_dict[row["fault_type"]]
        result.append(np.concatenate((x, [y])))

    result = np.array(result, dtype=np.float64)
    result = np.reshape(result, result.shape[0] * result.shape[1])
    print(input_path, ", result shape: ", result.shape)
    result.tofile(output_path)


def main():
    modes = ['train', 'val', 'test']
    for mode in modes:
        input_path = f"../Data/mech_{mode}.csv"
        output_path = f"/data/jyzheng/SeisCLIP/datasets/mech/{mode}.npy"
        preprocess(input_path, output_path)


if __name__ == "__main__":
    main()
