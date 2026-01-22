#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p ./logs

export CUDA_VISIBLE_DEVICES=0

torchrun --nproc-per-node 1 --master-port 25900 resnet_new_log.py \
  --arch resnet110 \
  --optimizer sgd \
  --batch_size 128 \
  --epochs 60 \
  --lr 0.5 \
  --warmup 10 \
  --print-freq 1 \
  --evaluate \
  --comp_dim 1000000 \
  > "./logs/baseline_cifar10_sgd_resnet110_bs128_ep60_lr05_compdim_off.log" 2>&1
