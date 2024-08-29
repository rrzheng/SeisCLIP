import argparse
import os
import random
import sys
from glob import glob

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from memmap_dataset import STEADDataset
from model_clip import AUDIO_CLIP
from utils import accuracy
from stead_dataloader import read_stead_data, stead_loader
from sklearn.model_selection import train_test_split
from utils import all_reduce_mean


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument("--local-rank", "--local_rank", type=int)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument("--input_dir", type=str, default='stead',
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

    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    device = torch.device('cuda', local_rank)

    # csv_stead = read_stead_data(os.path.join(args.input_dir, "chunk2.csv"))
    # train_stead, val_stead = train_test_split(csv_stead, test_size=0.2, random_state=42)
    # train_dataset = stead_loader(train_stead,
    #                              os.path.join(args.input_dir, "chunk2.hdf5"),
    #                              window_length=20)
    # val_dataset = stead_loader(val_stead,
    #                            os.path.join(args.input_dir, "chunk2.hdf5"),
    #                            window_length=20)

    train_paths = glob(os.path.join(args.input_dir, "train", "*.npy"))
    val_paths = glob(os.path.join(args.input_dir, "val", "*.npy"))
    if local_rank == 0:
        print("train_paths: ", train_paths)
        print("val_paths: ", val_paths)

    train_dataset = STEADDataset(*train_paths, window_length=20)
    val_dataset = STEADDataset(*val_paths, window_length=20)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  pin_memory=True,
                                  sampler=train_sampler,
                                  num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                pin_memory=True,
                                sampler=val_sampler,
                                num_workers=args.num_workers)

    model = AUDIO_CLIP(
        embed_dim=384, text_input=8, text_width=512, text_layers=2,
        spec_tdim=600, spec_model_size='small224',
        device_name=device, imagenet_pretrain=False)
    model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=True)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # define scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    best_val_acc = 0.0

    writer = SummaryWriter(args.log_dir)

    with tqdm(total=args.num_epochs, file=sys.stdout) as pbar:
        for epoch in range(args.num_epochs):
            train_loss, train_acc = 0.0, 0.0

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

                if local_rank == 0:
                    global_step = epoch * len(train_dataloader) + batch_idx
                    writer.add_scalar('train/loss', loss.item(), global_step=global_step)
                    writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=global_step)

                    train_loss += loss.item()
                    accuracy_1, _ = accuracy(logits, labels, topk=(1, 5))
                    train_acc += accuracy_1
                    num_steps = 10
                    if (batch_idx + 1) % num_steps == 0:
                            print(
                                f'Train - Epoch [{epoch + 1}/{args.num_epochs}], '
                                f'Iteration [{batch_idx + 1}/{len(train_dataloader)}], '
                                f'Loss: {train_loss / num_steps:.4f}, '
                                f'T_1 Accuracy: {train_acc.cpu().numpy()[0] / num_steps:.4f}.')
                            train_loss = 0.0
                            train_acc = 0.0

            # Evaluation
            model.eval()
            val_loss, val_acc = [], []
            with torch.no_grad():
                for batch in val_dataloader:
                    info = batch[0].to(device)
                    spec = batch[1].to(device)
                    labels = torch.tensor(np.arange(len(spec))).to(device)
                    logits, loss = model(info, spec)

                    acc1, _ = accuracy(logits, labels, topk=(1, 5))
                    val_acc.append(all_reduce_mean(acc1))
                    val_loss.append(all_reduce_mean(loss.item()))

            val_loss = np.mean(val_loss)
            val_acc = np.mean(val_acc)
            if local_rank == 0:
                pbar.write(f'Epoch [{epoch + 1}], '
                           f'Val_1 Accuracy: {val_acc:.4f}\n')
                pbar.update(1)
                writer.add_scalar('val/acc', val_acc, epoch + 1)

            # Update lr
            scheduler.step(val_loss)

            # Save model
            if (epoch + 1) % args.save_interval == 0:
                if dist.get_rank() == 0:
                    save_model_path = os.path.join(args.model_path, f"{epoch}.pt")
                    torch.save(model.module.state_dict(), save_model_path)
            if dist.get_rank() == 0:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_path = os.path.join(args.model_path, f"best_model.pt")
                    torch.save(model.module.state_dict(), best_model_path)

    dist.destroy_process_group()
    writer.close()


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
