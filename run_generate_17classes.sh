#!/bin/bash

# Script to generate 17 classes, each with 12 images, CFG=4.0

# Single GPU
torchrun --nnodes=1 --nproc_per_node=1 generate_17classes.py \
  --model SiT-XL/2 \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --per-proc-batch-size=4 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=4.0 \
  --guidance-high=0.7 \
  --global-seed=42 \
  --sample-dir=samples_17classes

# If you want to use multiple GPUs (faster):
# torchrun --nnodes=1 --nproc_per_node=2 generate_17classes.py \
#   --model SiT-XL/2 \
#   --path-type=linear \
#   --encoder-depth=8 \
#   --projector-embed-dims=768 \
#   --per-proc-batch-size=4 \
#   --mode=sde \
#   --num-steps=250 \
#   --cfg-scale=4.0 \
#   --guidance-high=0.7 \
#   --global-seed=42 \
#   --sample-dir=samples_17classes

# If you have a custom checkpoint:
# Add: --ckpt /path/to/your/checkpoint.pt
