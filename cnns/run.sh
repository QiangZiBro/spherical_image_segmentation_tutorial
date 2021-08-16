#!/bin/bash
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
--log_dir logs/log_unet_f16_cv3_rgbd \
--decay \
--train_stats_freq 5 \
--model UNet \
--in_ch rgbd \
--lr 1e-3 \
--feat 16

# FCN8s, UNet

