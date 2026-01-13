#!/bin/bash

# Bash script to run llava_ablation_study.py with different rank intervals
# Assumes the script is in the same directory and Python is available
# Adjust paths, GPU, batch_size, prompt, etc., as needed

# Fixed arguments
PROMPT='<image> Which label does not belong to this image? Answer the question with a single word from [{labels}].'
BATCH_SIZE=64
GPU=3
DATASET="cifar20"  # Or "cifar20" if needed

# List of rank intervals to test (1-based, e.g., [1:4] for ranks 1 to 4)
RANK_INTERVALS=("[8:11]" "[9:12]" "[10:13]" "[11:14]" "[12:15]" "[13:16]" "[14:17]" "[15:18]" "[16:19]" "[17:20]")

# Loop over each rank interval and run the script
for RANK in "${RANK_INTERVALS[@]}"; do
    echo "Running with rank interval: $RANK"
    python llava_ablation_study.py \
        --prompt "$PROMPT" \
        --batch_size $BATCH_SIZE \
        --gpu $GPU \
        --rank "$RANK" \
        --dataset $DATASET
    echo "Finished run for $RANK"
done

echo "All runs completed."