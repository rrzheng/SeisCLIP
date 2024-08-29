import time
import random
import numpy as np
import torch
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau



from sklearn.model_selection import train_test_split

import argparse
from tqdm import tqdm
import sys
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # parser.add_argument('--batch_size', type=int, default=192, help='batch size')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--save_interval', type=int, default=5, help='interval for model saving')
    parser.add_argument('--model_save_path', type=str, default='./work_dir/model_SeisClip_600/',
                        help='path to save the model')

    args = parser.parse_args()
    return args


def train(args):
    # Random seed
    set_seed(args.seed)
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_stead = read_stead_data()
    train_stead, val_stead = train_test_split(csv_stead, test_size=0.2, random_state=42)

    # Create Dataloder
    train_dataset = stead_loader(train_stead, window_length=20)
    val_dataset = stead_loader(val_stead, window_length=20)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=8)

    model = AUDIO_CLIP(
        embed_dim=384, text_input=8, text_width=512, text_layers=2,
        spec_tdim=600, spec_model_size='small224',
        device_name=device, imagenet_pretrain=True).to(device)

    # model.cuda(args.local_rank)
    # model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=True)
    # if dist.get_rank()==0:
    #     print('after DistributedDataParallel model')

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # define scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    best_signal_accuracy = 0.0
    best_info_accuracy = 0.0
    # set iterations
    # total_iterations = args.num_epochs * 2500000 / args.batch_size
    i = 0

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
            for batch in train_dataloader:
                info = batch[1].to(device)
                spec = batch[2].to(device)
                labels = torch.tensor(np.arange(len(spec))).to(device)
                _, logits, loss = model(info, spec)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                accuracy_1, accuracy_5 = accuracy(logits, labels, topk=(1, 5))

                train_signal_accuracy += accuracy_1
                train_text_accuracy += accuracy_5

                # Calculate remain time for training
                # remaining_iterations = total_iterations - completed_iterations
                # average_iteration_time = total_time / completed_iterations
                # average_iteration_time_minutes = average_iteration_time / 60
                # days, hours = divmod(remaining_iterations * average_iteration_time_minutes, 24 * 60)
                # hours, minutes = divmod(hours, 60)
                #
                # eta_formatted = f"{int(days)}.{str(int(hours)).zfill(2)}:{str(int(minutes)).zfill(2)}"

                if (i + 1) % 1 == 0:
                    # if dist.get_rank() == 0:
                    print(
                        f'Train - Epoch [{epoch + 1}/{args.num_epochs}], Iteration [{i + 1}/{len(train_dataloader)}], '
                        f'Loss: {train_loss / 100:.4f}, '
                        f'T_1 Accuracy: {train_signal_accuracy.cpu().numpy()[0] / 100:.4f}, '
                        f'T_5 Accuracy: {train_text_accuracy.cpu().numpy()[0] / 100:.4f}')
                    train_signal_accuracy = 0.0
                    train_text_accuracy = 0.0
                    train_loss = 0.0
                i += 1

            # Evaluation
            model.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                    info = batch[1].to(device)
                    spec = batch[2].to(device)
                    labels = torch.tensor(np.arange(len(spec))).to(device)
                    _, logits, loss = model(info, spec)

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
                  f'Val_5 Accuracy: {val_text_accuracy.cpu().numpy()[0]:.4f}')
            pbar.update(1)

            # Update lr
            scheduler.step(val_loss)

            # Save model
            if (epoch + 1) % args.save_interval == 0:
                # if dist.get_rank() == 0:
                save_model_name = args.model_save_path + str(epoch) + '.pt'
                torch.save(model.module.state_dict(), save_model_name)
            if (val_signal_accuracy > best_signal_accuracy) or (val_text_accuracy > best_info_accuracy):
                # if dist.get_rank() == 0:
                if val_signal_accuracy > best_signal_accuracy:
                    best_signal_accuracy = val_signal_accuracy
                if val_text_accuracy > best_info_accuracy:
                    best_info_accuracy = val_text_accuracy
                save_model_name = args.model_save_path + 'model_' + str(epoch) + '.' + str(
                    best_signal_accuracy.cpu().numpy()[0] + best_info_accuracy.cpu().numpy()[0]) + '.pt'
                torch.save(model.state_dict(), save_model_name)


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
