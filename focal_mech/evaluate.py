import argparse

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from dataset import MechDataset
from model_v2 import build_model


def parse_args():
    parser = argparse.ArgumentParser("Evaluate model configuration")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    args = parser.parse_args()
    return args


def main(args):
    test_path = [args.input_path]
    test_dataset = MechDataset(*test_path)
    test_loader = DataLoader(test_dataset, batch_size=8)

    model = build_model()
    param = torch.load(args.model_path)
    model.load_state_dict(param)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        model.eval()
        for mask, spec, stat_info, labels in test_loader:
            mask = mask.to(device)
            spec = spec.to(device)
            stat_info = stat_info.to(device)
            labels = labels.to(device)
            outputs = model(spec, stat_info, mask)
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
                               target_names=['strike-slip', 'normal-fault', 'reverse-fault'],
                               digits=4)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(cr)


if __name__ == '__main__':
    args = parse_args()
    main(args)
