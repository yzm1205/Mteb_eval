#!/bin/bash

python evaluate_mteb.py \
    --model_name "olmo" \
    --batch_size 2 \
    --device "cuda:1"
    
