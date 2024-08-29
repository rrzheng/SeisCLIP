import argparse

import torch
from geopy.distance import great_circle
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

from dataset import LocationDataset
from location.utils import DEP_MIN, DEP_MAX, MAG_MIN, MAG_MAX
from model import build_model
from utils import inverse_min_max_scale


def parse_args():
    parser = argparse.ArgumentParser("Evaluate model configuration")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')

    args = parser.parse_args()
    return args


def main(args):
    test_path = [args.input_path]
    test_dataset = LocationDataset(*test_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model()
    param = torch.load(args.model_path, map_location=device)
    model.load_state_dict(param)
    model.to(device)

    all_pred_lats = []
    all_pred_lons = []
    all_pred_deps = []
    all_pred_mags = []
    all_true_lats = []
    all_true_lons = []
    all_true_deps = []
    all_true_mags = []

    with torch.no_grad():
        model.eval()
        for mask, spec, stat_info, mmc, targets in test_loader:
            mask = mask.to(device)
            spec = spec.to(device)
            stat_info = stat_info.to(device)
            outputs = model(spec, stat_info, mask)

            for i in range(len(outputs)):
                all_pred_lats.append(inverse_min_max_scale(outputs[i, 0].item(), mmc[i, 0].item(), mmc[i, 1].item()))
                all_pred_lons.append(inverse_min_max_scale(outputs[i, 1].item(), mmc[i, 2].item(), mmc[i, 3].item()))
                all_pred_deps.append(inverse_min_max_scale(outputs[i, 2].item(), DEP_MIN, DEP_MAX))
                all_pred_mags.append(inverse_min_max_scale(outputs[i, 3].item(), MAG_MIN, MAG_MAX))

                all_true_lats.append(inverse_min_max_scale(targets[i, 0].item(), mmc[i, 0].item(), mmc[i, 1].item()))
                all_true_lons.append(inverse_min_max_scale(targets[i, 1].item(), mmc[i, 2].item(), mmc[i, 3].item()))
                all_true_deps.append(inverse_min_max_scale(targets[i, 2].item(), DEP_MIN, DEP_MAX))
                all_true_mags.append(inverse_min_max_scale(targets[i, 3].item(), MAG_MIN, MAG_MAX))

    # 计算震源距离
    examples_size = len(all_pred_mags)
    pred_dists = []
    for idx in range(examples_size):
        pred_point = (all_pred_lats[idx], all_pred_lons[idx])
        true_point = (all_true_lats[idx], all_true_lons[idx])
        dist = great_circle(pred_point, true_point).kilometers
        pred_dists.append(dist)

    lat_mae = mean_absolute_error(all_pred_lats, all_true_lats)
    long_mae = mean_absolute_error(all_pred_lons, all_true_lons)
    dist_mae = mean_absolute_error(pred_dists, [0] * examples_size)
    dep_mae = mean_absolute_error(all_pred_deps, all_true_deps)
    mag_mae = mean_absolute_error(all_pred_mags, all_true_mags)
    print(f"Latitude MAE: {lat_mae:.4f}")
    print(f"Longitude MAE: {long_mae:.4f}")
    print(f"Epi dist MAE: {dist_mae:.4f}")
    print(f"Dep MAG: {dep_mae:.4f}")
    print(f"Mag MAG: {mag_mae:.4f}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
