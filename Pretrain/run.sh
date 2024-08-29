#!/bin/bash


OMP_NUM_THREADS=4
nproc_per_node=8

export OMP_NUM_THREADS=$OMP_NUM_THREADS

root_dir="/mnt/seismic/seismic/SeisCLIP"


mkdir /workspace/train/
mkdir /workspace/val/

rsync -avh /mnt/seismic/seismic/SeisCLIP/stead_part/train/ /workspace/train/
rsync -avh /mnt/seismic/seismic/SeisCLIP/stead_part/val/  /workspace/val/

torchrun --nproc_per_node=$nproc_per_node train_seis_clip.py \
    --batch_size 384 \
    --num_epochs 200 \
    --input_dir /workspace/ \
    --model_path $root_dir/ckpt/pretrained_models/v2/ \
    --log_dir $root_dir/ckpt/pretrained_models/v2/tf_logs/