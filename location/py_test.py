import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# train_df = pd.read_csv("../Data/location_train.csv")
# test_df = pd.read_csv("../Data/location_test.csv")
# val_df = pd.read_csv("../Data/location_val.csv")
#
#
# def get_spec_station_info(jma_id: str):
#     spec_path = f"/data/jyzheng/datasets/location/{jma_id}.npy"
#     spec_data = np.load(spec_path)
#     station_info_path = f"/data/jyzheng/datasets/location/{jma_id}.npy"
#     station_info = np.load(station_info_path)
#
#     return np.min(spec_data), np.max(spec_data)
    # for i in range(spec_data.shape[0]):
    #     print(np.min(spec_data[i]), np.max(spec_data[i]))

    # spec = spec_data[i, :, :6000]
    # print(spec.shape)
    # plt.plot(spec[0])
    # plt.show()


# min_gap_arr = []
# max_gap_arr = []
# for df in [train_df, test_df, val_df]:
#     for row_idx in range(df.shape[0]):
#         row = df.iloc[row_idx]
#         epi_lat = float(row["lat"][:-1])
#         epi_long = float(row["long"][:-1])
#
#         jma_id = row["jmaID"]
#         raw_spec = np.load(f"/data/jyzheng/datasets/location/spec/{jma_id}.npy")
#         raw_stat = np.load(f"/data/jyzheng/datasets/mech/label_station/sta_info/{jma_id}.npy")
#
#         if raw_spec.shape[0] == 0:
#             continue
#
#         j = 0
#         lat_arr = []
#         long_arr = []
#         for i in range(raw_spec.shape[0]):
#             if np.all(np.abs(raw_spec[i]) < 1e-1):
#                 continue
#             else:
#                 sta_lat = raw_stat[i, 1]
#                 sta_long = raw_stat[i, 0]
#                 lat_arr.append(sta_lat)
#                 long_arr.append(sta_long)
#                 j += 1
#                 if j >= 32:
#                     break
#         if len(lat_arr) == 0:
#             continue

#         min_gap_arr.append(epi_lat / min(lat_arr))
#         max_gap_arr.append(epi_lat / max(lat_arr))
#
# print("min: ", min(min_gap_arr), max(min_gap_arr))
# print("max: ", min(max_gap_arr), max(max_gap_arr))

        # if epi_lat < 0.87*min(lat_arr) or epi_lat > 1.033*max(lat_arr):
        #     print(epi_lat, np.min(lat_arr), np.max(lat_arr))
        #     break
        # if epi_long < 0.98*min(long_arr) or epi_long > 1.027*max(long_arr):
        #     print(epi_long, np.min(long_arr), np.max(long_arr))
        #     break




# for df in [train_df, test_df, val_df]:
#     lat = df["lat"].apply(lambda l: float(l[:-1]))
#     print("lat: ", lat.min(), lat.max())
#
#     long = df["long"].apply(lambda l: float(l[:-1]))
#     print("long: ", long.min(), long.max())
#
#     dep = df["dep"].apply(lambda l: float(l[:-2]))
#     print("dep: ", dep.min(), dep.max())
#
#     mag = df["m"].apply(lambda l: float(l))
#     print("mag: ", mag.min(), mag.max())



import seisbench
import seisbench.models as sbm
import torch

# model = sbm.PhaseNet()
# print(model)


# pretrained_weights = sbm.PhaseNet.list_pretrained(details=True)
# for key, value in pretrained_weights.items():
#     print(f"{key}:\n{value}\n-----------------------\n")


model = sbm.PhaseNet.from_pretrained("original")
print(model.weights_docstring)

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt

client = Client("GFZ")

t = UTCDateTime("2007/01/02 05:48:50")
stream = client.get_waveforms(network="CX", station="PB01", location="*", channel="HH?", starttime=t-100, endtime=t+100)

print(stream[0].data.shape)

annotations = model.annotate(stream)
print(annotations)




