import os

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

csv_paths = [
    "/data/jyzheng/datasets/stead/chunk2.csv",
    "/data/jyzheng/datasets/stead/chunk3.csv",
    "/data/jyzheng/datasets/stead/chunk4.csv",
    "/data/jyzheng/datasets/stead/chunk5.csv",
    "/data/jyzheng/datasets/stead/chunk6.csv"
]

hdf5_paths = [
    "/data/jyzheng/datasets/stead/chunk2.hdf5",
    "/data/jyzheng/datasets/stead/chunk3.hdf5",
    "/data/jyzheng/datasets/stead/chunk4.hdf5",
    "/data/jyzheng/datasets/stead/chunk5.hdf5",
    "/data/jyzheng/datasets/stead/chunk6.hdf5"
]

partition_size = 40000


def generate_files(dtfl, evi_list, output_name, dtype=np.float32):
    arr = []
    for evi in evi_list:
        x = dtfl.get('data/' + str(evi))
        spec = np.array(x)
        spec = spec.reshape(spec.shape[0] * spec.shape[1])

        info = np.array([x.attrs['p_arrival_sample'],
                         x.attrs['p_weight'],
                         x.attrs['p_travel_sec'],
                         x.attrs['s_arrival_sample'],
                         x.attrs['s_weight'],
                         x.attrs['source_distance_km'],
                         x.attrs['back_azimuth_deg'],
                         x.attrs['coda_end_sample'][0][0]
                         ])
        info[info == ''] = 0.0
        info = np.nan_to_num(info, nan=0.0)
        info[0] = float(info[0]) / 6000
        info[2] = float(info[2]) / 60
        info[3] = float(info[3]) / 6000
        info[5] = float(info[5]) / 300
        info[6] = float(info[6]) / 360
        info[7] = float(info[7]) / 6000
        info = np.array(info, dtype=dtype)

        sample = np.concatenate((spec, info))
        arr.append(sample)

    arr = np.array(arr, dtype=dtype)
    arr = arr.reshape(arr.shape[0] * arr.shape[1])
    arr.tofile(output_name)


def process_file(csv_path, hdf5_path, idx, output_dir):
    print(f"Processing files {csv_path} and {hdf5_path}")
    meta_df = pd.read_csv(csv_path, low_memory=False)
    meta_df = meta_df[(meta_df.trace_category == 'earthquake_local')]
    evi_arr = meta_df["trace_name"].to_numpy()

    train_evi, val_evi = train_test_split(evi_arr, test_size=0.2, random_state=42)
    train_evi = np.array_split(train_evi, train_evi.shape[0] / partition_size)
    val_evi = np.array_split(val_evi, val_evi.shape[0] / partition_size)

    # train_evi = [evi_arr[:10]]

    with h5py.File(hdf5_path, "r") as dtfl:
        for sub_idx, evi_arr in enumerate(train_evi):
            output_name = os.path.join(output_dir, "train", f'part-{idx:02d}{sub_idx:02d}.npy')
            generate_files(dtfl, evi_arr, output_name)

        for sub_idx, evi_arr in enumerate(val_evi):
            output_name = os.path.join(output_dir, "val", f'part-{idx:02d}{sub_idx:02d}.npy')
            generate_files(dtfl, evi_arr, output_name)

    print(f"Processed file {csv_path} and {hdf5_path}.")


def main(output_dir):
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, csv_path, hdf5_path, idx, output_dir)
            for idx, (csv_path, hdf5_path) in enumerate(zip(csv_paths, hdf5_paths))
        ]
        for future in futures:
            future.result()

    # process_file(csv_paths[0], hdf5_paths[0], 0, output_dir)


if __name__ == '__main__':
    output_dir = "/data/jyzheng/SeisCLIP/datasets/stead_v2/"
    main(output_dir)
