#!/bin/bash
# Model choice
# ResNetDUCHDC,FCN8s,UNet
# Run example
# 1) ./run.sh
# 2) ./run.sh FCN8s
# 3) ./run.sh ResNetDUCHDC
model="${1:-UNet}"
MESHFILES=../data/mesh_files
DATADIR=../data/2d3ds_pano_small/
# create log directory
mkdir -p logs

python train.py \
--batch-size 16 \
--test-batch-size 16 \
--epochs 200 \
--data_folder $DATADIR \
--mesh_folder $MESHFILES \
--fold 3 \
--log_dir logs/log_${model}_f16_cv3_rgbd \
--decay \
--train_stats_freq 5 \
--model ${model} \
--in_ch rgbd \
--lr 1e-3 \
--feat 16

