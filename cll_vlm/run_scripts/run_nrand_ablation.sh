#!/bin/bash

# Bash script to run llava_ablation_study.py with cifar20 dataset and varying n_random from 2 to 10

DATASET="tiny200"
GPU=0

# echo "Starting ablation study for $DATASET with n_random from 2 to 10"
for n_random in {7..10}
do
    if [ $n_random -eq 3 ]; then
        continue
    fi
    echo "=========================================="
    echo "Running with n_random=$n_random"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=${GPU} python llava_ablation_study.py \
        --dataset $DATASET \
        --n_random $n_random
    
    # Check if the previous command succeeded
    if [ $? -eq 0 ]; then
        echo "? Completed n_random=$n_random successfully"
    else
        echo "? Failed at n_random=$n_random"
        exit 1
    fi
    
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="