import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import PNWDataset
from model import build_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

def parse_args():
    parser = argparse.ArgumentParser("Evaluate model configuration")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    args = parser.parse_args()
    return args


def main(args):
    test_path = [args.input_path]
    test_dataset = PNWDataset(*test_path)
    test_loader = DataLoader(test_dataset, batch_size=256)

    model = build_model()
    param = torch.load(args.model_path)
    model.load_state_dict(param)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    cr = classification_report(all_labels,
                               all_preds,
                               target_names=["earthquake", "explosion", "surface event"],
                               digits=4)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(cr)


if __name__ == '__main__':
    args = parse_args()
    main(args)
