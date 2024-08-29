from typing import NamedTuple

LAT_MIN, LAT_MAX = 30.012, 44.873
LONG_MIN, LONG_MAX = 130.003, 149.619
DEP_MIN, DEP_MAX = 0.0, 99.6
MAG_MIN, MAG_MAX = 2.0, 7.4


class MinMaxCoordinate(NamedTuple):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


def min_max_scale(val, min_val, max_val) -> float:
    return (val - min_val) / (max_val - min_val)


def inverse_min_max_scale(scale_val, min_val, max_val):
    return scale_val * (max_val - min_val) + min_val
