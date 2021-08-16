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

python test.py \
--test-batch-size 16 \
--data_folder $DATADIR \
--mesh_folder $MESHFILES \
--ckpt logs/log_${model}_f16_cv3_rgbd/checkpoint_latest.pth.tar_${model}_best.pth.tar \
--export_file ${model}.npz \
--fold 3 \
--model ${model} \
--in_ch rgbd \
--feat 16

