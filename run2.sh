#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python evaluate_mteb.py \
    --model_name "olmo" \
    --batch_size 16 \
    --device "auto"
    