#!/bin/bash

python evaluate_mteb.py \
    --model_name "olmo" \
    --batch_size 8 \
    --device "cuda:0"
    
