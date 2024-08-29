import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_waveform(trace_name, dtfl, label, start_index=4000):
    bucket, array = trace_name.split('$')
    x, y, z = iter([int(i) for i in array.split(',:')])
    data = dtfl.get(f"/data/{bucket}")[x, :y, :z]
    data = np.array(data)
    spec = data[:, start_index:(start_index + 6000)]
    if spec.shape[1] != 6000:
        print("spec: ", start_index, data.shape, spec.shape)
    spec = np.transpose(spec)
    spec = np.reshape(spec, spec.shape[0] * spec.shape[1])
    spec = np.concatenate((spec, [label]), dtype=np.float64)
    return spec


def main(input_path, output_path):
    comcat_meta_path = "/data/jyzheng/datasets/pnw/comcat_metadata.csv"
    comcat_hdf5_path = "/data/jyzheng/datasets/pnw/comcat_waveforms.hdf5"
    comcat_meta_df = pd.read_csv(comcat_meta_path)

    exotic_meta_path = "/data/jyzheng/datasets/pnw/exotic_metadata.csv"
    exotic_hdf5_path = "/data/jyzheng/datasets/pnw/exotic_waveforms.hdf5"
    exotic_meta_df = pd.read_csv(exotic_meta_path)

    labels = ["earthquake", "explosion", "surface event"]
    label_dict = {label: float(idx) for idx, label in enumerate(labels)}

    df = pd.read_csv(input_path)
    # print(df["source_type"].value_counts())

    result_data = []
    with h5py.File(comcat_hdf5_path, "r") as chr, \
            h5py.File(exotic_hdf5_path, "r") as ehr:
        for _, row in tqdm(df.iterrows()):
            st = row["source_type"]
            p_arri = row["trace_P_arrival_sample"]
            trace_name = row["trace_name"]
            import random
            gap_second = random.randint(5, 10)

            if st == "earthquake" or st == "explosion":
                p_arri = int(p_arri)
                start_index = (p_arri // 100 - 15) * 100
                if start_index > 9000:
                    print("start_index exceed 9000", start_index)
                    start_index = 9000
                data = parse_waveform(trace_name, chr, label_dict[st], start_index)
                result_data.append(data)

            elif st == "surface event":
                if np.isnan(p_arri):
                    p_arri = 5000
                else:
                    p_arri = int(p_arri)
                start_index = (p_arri // 100 - gap_second) * 100
                if start_index > 9000:
                    print("start_index exceed 9000", start_index)
                    start_index = 9000
                data = parse_waveform(trace_name, ehr, label_dict[st], start_index)
                result_data.append(data)

    result_data = np.array(result_data, dtype=np.float64)
    result_data = np.reshape(result_data, result_data.shape[0] * result_data.shape[1])
    print(result_data.shape)
    result_data.tofile(output_path)


if __name__ == "__main__":
    for mode in ["train", "val", "test"]:
        input_path = f"/data/jyzheng/SeisCLIP/Data/classification_{mode}.csv"
        output_path = f"/data/jyzheng/SeisCLIP/datasets/pnw_v6/{mode}.npy"
        main(input_path, output_path)
