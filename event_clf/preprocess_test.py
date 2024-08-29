from dataset import PNWDataset
import pandas as pd
import h5py
import numpy as np


# pnw_path = ["/data/jyzheng/SeisCLIP/datasets/pnw/evt_label.npy"]
# dataset = PNWDataset(*pnw_path)
#
# labels = ["earthquake", "explosion", "surface event"]
# label_dict = {label: idx for idx, label in enumerate(labels)}
#
# meta_path = "/data/jyzheng/SeisCLIP/Data/classification_train.csv"
# df = pd.read_csv(meta_path)
# comcat_hdf5_path = "/data/jyzheng/datasets/pnw/comcat_waveforms.hdf5"
# exotic_hdf5_path = "/data/jyzheng/datasets/pnw/exotic_waveforms.hdf5"
#
#
def parse_waveform(trace_name, dtfl):
    bucket, array = trace_name.split('$')
    x, y, z = iter([int(i) for i in array.split(',:')])
    data = dtfl.get(f"/data/{bucket}")[x, :y, :z]
    data = np.array(data)
    data = data[:, 3000:9000]
    data = np.transpose(data)
    return data


#
#
# idx = 0
# with h5py.File(comcat_hdf5_path, "r") as hr:
#     for _, row in df.iterrows():
#         st = row["source_type"]
#         if st == "earthquake" or st == "explosion":
#             data = parse_waveform(row["trace_name"], hr)
#             spec, info = dataset[idx]
#             assert np.allclose(data[:18000], spec.numpy())
#             assert label_dict[st] == info.numpy()[0]
#             idx += 1
#             if idx == 3:
#                 break
#
#
# with h5py.File(exotic_hdf5_path, "r") as hr:
#     for _, row in df.iterrows():
#         st = row["source_type"]
#         if st == "surface event":
#             data = parse_waveform(row["trace_name"], hr)
#             spec, info = dataset[idx]
#             assert np.allclose(data[:18000], spec.numpy())
#             assert label_dict[st] == info.numpy()[0]
#             idx += 1
#             if idx == 6:
#                 break

# test_df = pd.read_csv('./data/metadata_test.csv')
# test_df = test_df[test_df["source_type"] != "surface event"]
# row_index = 1
# row = test_df.loc[row_index]
# trace_name = row["trace_name"]
# print(test_df["source_type"].value_counts())
# print(test_df["trace_P_arrival_sample"], test_df["trace_S_arrival_sample"])


# from datetime import datetime
#
#
# test_df = pd.read_csv("/data/jyzheng/datasets/stead/chunk4.csv")
# time_format = '%Y-%m-%d %H:%M:%S.%f'
# cnt = 0
# for index, row in test_df.iterrows():
#     time1 = datetime.strptime(row["trace_start_time"], time_format)
#     time2 = datetime.strptime(row["source_origin_time"], time_format)
#     time_difference = time2 - time1
#     seconds_difference = time_difference.total_seconds()
#     print(row["trace_start_time"], row["source_origin_time"], seconds_difference)
#     cnt +=1
#     if cnt == 10:
#         break

# wf = np.load("./data/waveform_test.npz")["waveform"]
# spec = wf[row_index,]
# print(spec.shape, spec.dtype)
# # print(spec.min(), spec.max())
# print(spec)
# val = spec[4, -1]
# print("val: ", val)


# from math import isclose
#
# with h5py.File('/data/jyzheng/datasets/pnw/comcat_waveforms.hdf5', 'r') as hr:
#     data = parse_waveform(trace_name, dtfl=hr)
#     print(data.shape, data.dtype)
#     # print(data.min(), data.max())
#     print(data)
#     for index, e in np.ndenumerate(data):
#         if isclose(e, val, abs_tol=1e-2):
#             print(e, index)


# evt_id = "pnsn1421343"
# station_code = "VFP"
# cols = ["trace_name", "trace_P_arrival_sample", "source_type"]

df = pd.read_csv("../Data/classification_train.csv")
# df = df[(df["event_id"] == evt_id) &
#           (df["station_code"] == station_code)]
# print(df[cols])
p_arri = df["trace_P_arrival_sample"]
print(p_arri.min(), p_arri.max())

comcat_meta_df = pd.read_csv('/data/jyzheng/datasets/pnw/comcat_metadata.csv')
# c_df = comcat_meta_df[(comcat_meta_df["event_id"] == evt_id) &
#                       (comcat_meta_df["station_code"] == station_code)]
# print(c_df[cols])
# p_arri = comcat_meta_df["trace_P_arrival_sample"]
# print(p_arri.min(), p_arri.max())

exotic_meta_df = pd.read_csv('/data/jyzheng/datasets/pnw/exotic_metadata.csv')
# e_df = exotic_meta_df[(exotic_meta_df["event_id"] == evt_id) &
#                       (exotic_meta_df["station_code"] == station_code)]
# print(e_df[cols])

# s = meta_df["trace_P_arrival_sample"]
# print(s.min(), s.max())
# s = test_df["trace_S_arrival_sample"]
# print(s.min(), s.max())
# meta_df = meta_df[meta_df['event_id'] == evt_id]
# print(meta_df[["event_id", "source_type", "station_code"]])
