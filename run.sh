#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1 
python evaluate_mteb.py \
    --model_name "openelm" \
    --batch_size 8 \
    --device "auto"
    