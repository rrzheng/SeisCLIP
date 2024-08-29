#!/bin/bash


OMP_NUM_THREADS=4
nproc_per_node=8

export OMP_NUM_THREADS=$OMP_NUM_THREADS

root_dir="/mnt/seismic/seismic/SeisCLIP"


rsync -avh /mnt/seismic/seismic/SeisCLIP/pnw/ /workspace/

torchrun --nproc_per_node=$nproc_per_node train.py \
    --learning_rate 1e-4 \
    --batch_size 512 \
    --num_epochs 200 \
    --input_dir /workspace/ \
    --pretrained_model $root_dir/ckpt/pretrained_models/v2/199.pt \
    --model_path $root_dir/ckpt/evt_clf/v1/ \
    --log_dir $root_dir/ckpt/evt_clf/v1/tf_logs/