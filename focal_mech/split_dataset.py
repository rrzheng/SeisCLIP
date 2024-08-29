import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# 断层类型
def determine_fault_type(rake):
    """
    Normal Fault: rake < 0
    Reverse Fault: rake > 0
    Strike-Slip Fault: rake ≈ 0 或 rake ≈ ±180
    """
    if -35 <= rake <= 35 or -180 <= rake <= -135 or 135 <= rake <= 180:
        return 'strike-slip'
    elif -135 < rake < -35:
        return 'normal-fault'
    elif 35 < rake < 135:
        return 'reverse-fault'
    else:
        return 'Unclassified'


def save_as_csv(id_arr, type_arr, path):
    df = pd.DataFrame({
        "jmaID": id_arr,
        "fault_type": type_arr})
    print(df["fault_type"].value_counts())
    df.to_csv(path, index=False)


def main():
    df = pd.read_csv("../Data/select_mechanism_2011_2016.csv")
    cols = ["strike", "dip", "slip", "jmaID"]
    df = df[cols]
    df["fault_type"] = df.apply(lambda row: determine_fault_type(row["slip"]), axis=1)
    print(df["fault_type"].value_counts())

    id_arr = []
    type_arr = []
    for _, row in df.iterrows():
        id_arr.append(row["jmaID"])
        type_arr.append(row["fault_type"])

    id_train, id_test, y_train, y_test = train_test_split(id_arr, type_arr, test_size=0.2, random_state=0)
    print()
    id_val, id_test, y_val, y_test = train_test_split(id_test, y_test, test_size=0.7, random_state=0)

    save_as_csv(id_train, y_train, path="../Data/mech_train.csv")
    save_as_csv(id_val, y_val, path="../Data/mech_val.csv")
    save_as_csv(id_test, y_test, path="../Data/mech_test.csv")


if __name__ == "__main__":
    main()
