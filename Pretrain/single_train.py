import argparse
import random
import sys

import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import numpy as np
import torch

from model_clip import AUDIO_CLIP
from stead_dataloader import read_stead_data, stead_loader
from utils import accuracy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument("--input_dir", type=str, default='../datasets/stead',
                        help='input STEAD dataset dir')
    parser.add_argument('--save_interval', type=int, default=5, help='interval for model saving')
    parser.add_argument('--model_path', type=str, default='./pretrained_models/',
                        help='path to save the model')
    parser.add_argument("--log_dir", type=str, default='./logs/', help='tensorboard log dir')
    parser.add_argument("--num_workers", type=int, default=64, help='number of workers')

    args = parser.parse_args()
    return args


def train(args):
    # Random seed
    set_seed(args.seed)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_stead = read_stead_data(os.path.join(args.input_dir, "merged.csv"))
    train_stead, val_stead = train_test_split(csv_stead, test_size=0.2, random_state=42)
    train_dataset = stead_loader(train_stead,
                                 os.path.join(args.input_dir, "merged.hdf5"),
                                 window_length=20)
    val_dataset = stead_loader(val_stead,
                               os.path.join(args.input_dir, "merged.hdf5"),
                               window_length=20)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

    model = AUDIO_CLIP(
        embed_dim=384, text_input=8, text_width=512, text_layers=2,
        spec_tdim=600, spec_model_size='small224',
        device_name=device, imagenet_pretrain=True).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # define scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    best_signal_accuracy = 0.0
    best_info_accuracy = 0.0

    writer = SummaryWriter(args.log_dir)

    with tqdm(total=args.num_epochs, file=sys.stdout) as pbar:
        for epoch in range(args.num_epochs):
            train_loss = 0.0
            val_loss = 0.0
            val_signal_accuracy = 0.0
            val_text_accuracy = 0.0
            train_signal_accuracy = 0.0
            train_text_accuracy = 0.0

            # Training
            model.train()
            for batch_idx, data in enumerate(train_dataloader):
                info = data[0].to(device)
                spec = data[1].to(device)
                labels = torch.tensor(np.arange(len(spec))).to(device)
                logits, loss = model(info, spec)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                global_step = epoch * len(train_dataloader) + batch_idx
                writer.add_scalar('train/loss', loss.item(), global_step=global_step)
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=global_step)

                accuracy_1, accuracy_5 = accuracy(logits, labels, topk=(1, 5))

                train_signal_accuracy += accuracy_1
                train_text_accuracy += accuracy_5

                if (batch_idx + 1) % 10 == 0:
                        print(
                            f'Train - Epoch [{epoch + 1}/{args.num_epochs}], Iteration [{batch_idx + 1}/{len(train_dataloader)}], '
                            f'Loss: {train_loss / 100:.4f}, '
                            f'T_1 Accuracy: {train_signal_accuracy.cpu().numpy()[0] / 100:.4f}, '
                            f'T_5 Accuracy: {train_text_accuracy.cpu().numpy()[0] / 100:.4f}')
                        train_signal_accuracy = 0.0
                        train_text_accuracy = 0.0
                        train_loss = 0.0

            # Evaluation
            model.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                    info = batch[0].to(device)
                    spec = batch[1].to(device)
                    labels = torch.tensor(np.arange(len(spec))).to(device)
                    logits, loss = model(info, spec)

                    val_loss += loss.item()

                    accuracy_singal, accuracy_info = accuracy(logits, labels, topk=(1, 5))

                    val_signal_accuracy += accuracy_singal
                    val_text_accuracy += accuracy_info

            train_loss /= len(train_dataloader)
            val_loss /= len(val_dataloader)
            val_signal_accuracy /= len(val_dataloader)
            val_text_accuracy /= len(val_dataloader)
            pbar.write(f'Epoch [{epoch + 1}], Train Loss: {train_loss:.4f}, '
                       f'Val Loss: {val_loss:.4f}, Val_1 Accuracy: {val_signal_accuracy.cpu().numpy()[0]:.4f}, '
                       f'Val_5 Accuracy: {val_text_accuracy.cpu().numpy()[0]:.4f}\n')
            pbar.update(1)

            writer.add_scalar('val/loss', val_loss, epoch + 1)

            # Update lr
            scheduler.step(val_loss)

            # Save model
            if (epoch + 1) % args.save_interval == 0:
                    save_model_path = os.path.join(args.model_path, str(epoch) + '.pt')
                    torch.save(model.module.state_dict(), save_model_path)
            if (val_signal_accuracy > best_signal_accuracy) or (val_text_accuracy > best_info_accuracy):
                    if val_signal_accuracy > best_signal_accuracy:
                        best_signal_accuracy = val_signal_accuracy
                    if val_text_accuracy > best_info_accuracy:
                        best_info_accuracy = val_text_accuracy
                    save_model_name = os.path.join(args.model_path, 'model_' + str(epoch) + '.' + str(
                        best_signal_accuracy.cpu().numpy()[0] + best_info_accuracy.cpu().numpy()[0]) + '.pt')
                    torch.save(model.state_dict(), save_model_name)

    writer.close()


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
