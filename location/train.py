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

from dataset import LocationDataset
from model import build_model_from_pretrain


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def parse_args():
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--input_dir", type=str, default='location',
                        help='input focal mechanism dataset dir')
    parser.add_argument("--pretrained_model", type=str, default='pretrained_model.pth')
    parser.add_argument("--mode", type=str, default='finetune')
    parser.add_argument('--save_interval', type=int, default=5, help='interval for model saving')
    parser.add_argument('--model_path', type=str, default='./location/',
                        help='path to save the model')
    parser.add_argument("--log_dir", type=str, default='./logs/', help='tensorboard log dir')
    parser.add_argument("--num_workers", type=int, default=32, help='number of workers')

    args = parser.parse_args()
    return args


def evaluate(data_loader, model, mse_criterion, mae_criterion, device):
    total_mse_loss = 0.0
    total_mae_loss = 0.0
    total = 0

    with torch.no_grad():
        model.eval()
        for mask, spec, stat_info, mmc, targets in data_loader:
            mask = mask.to(device)
            spec = spec.to(device)
            stat_info = stat_info.to(device)
            targets = targets.to(device)
            logit = model(spec, stat_info, mask)

            mse_loss = mse_criterion(logit, targets)
            mae_loss = mae_criterion(logit, targets)

            total_mse_loss += mse_loss.item() * spec.size(0)
            total_mae_loss += mae_loss.item() * spec.size(0)
            total += targets.size(0)

    # 收集所有GPU的结果
    total_mse_loss_all = torch.tensor([total_mse_loss], device=device)
    total_mae_loss_all = torch.tensor([total_mae_loss], device=device)
    total_all = torch.tensor([total], device=device)

    dist.all_reduce(total_mse_loss_all, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_mae_loss_all, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_all, op=dist.ReduceOp.SUM)

    # 计算平均损失
    avg_mse_loss = total_mse_loss_all.item() / total_all.item()
    avg_mae_loss = total_mae_loss_all.item() / total_all.item()

    return avg_mse_loss, avg_mae_loss


def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    dist.init_process_group(backend='nccl')

    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    random_seed(rank=local_rank)

    train_path = [os.path.join(args.input_dir, 'train.npy')]
    val_path = [os.path.join(args.input_dir, 'val.npy')]

    train_dataset = LocationDataset(*train_path)
    val_dataset = LocationDataset(*val_path)
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              pin_memory=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            sampler=val_sampler,
                            pin_memory=True,
                            num_workers=args.num_workers)

    model = build_model_from_pretrain(args.pretrained_model, device, mode=args.mode)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=True)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)
    mse_criterion = torch.nn.MSELoss()
    mae_criterion = torch.nn.L1Loss()

    writer = SummaryWriter(args.log_dir)

    best_val_loss = float('inf')
    with tqdm(total=args.num_epochs, file=sys.stdout) as pbar:
        for epoch in range(1, args.num_epochs + 1):
            train_loss = 0.0

            # Training
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                mask = batch[0].to(device)
                spec = batch[1].to(device)
                stat_info = batch[2].to(device)
                label = batch[4].to(device)
                logit = model(spec, stat_info, mask)
                loss = mse_criterion(logit, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if local_rank == 0:
                    global_step = (epoch - 1) * len(train_loader) + batch_idx
                    writer.add_scalar('train/loss', loss.item(), global_step=global_step)
                    writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=global_step)

                    train_loss += loss.item()
                    num_steps = 10
                    if (batch_idx + 1) % num_steps == 0:
                        print(
                            f'Train - Epoch [{epoch}/{args.num_epochs}], '
                            f'Iteration [{batch_idx + 1}/{len(train_loader)}], '
                            f'Loss: {train_loss / num_steps:.4f}')
                        train_loss = 0.0

            # validation
            val_mse_loss, val_mae_loss = evaluate(val_loader, model, mse_criterion, mae_criterion, device)
            if local_rank == 0:
                pbar.write(f'Epoch [{epoch}], '
                           f'Valid MSE loss: {val_mse_loss:.4f}, '
                           f'Valid MAE loss: {val_mae_loss:.4f}\n')
                pbar.update(1)
                writer.add_scalar('val/mse_loss', val_mae_loss, epoch)
                writer.add_scalar('val/mae_loss', val_mae_loss, epoch)

            # Update lr
            scheduler.step(val_mse_loss)

            # Save model
            if dist.get_rank() == 0:
                if epoch % args.save_interval == 0:
                    save_model_path = os.path.join(args.model_path, f"{epoch}.pt")
                    torch.save(model.module.state_dict(), save_model_path)
                if val_mae_loss < best_val_loss:
                    best_val_loss = val_mae_loss
                    best_model_path = os.path.join(args.model_path, f"best_model.pt")
                    torch.save(model.module.state_dict(), best_model_path)

    dist.destroy_process_group()
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
