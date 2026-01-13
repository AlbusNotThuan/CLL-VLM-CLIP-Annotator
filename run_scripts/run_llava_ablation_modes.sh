#!/usr/bin/env bash
set -euo pipefail

# Simple runner: loops over modes and runs llava_ablation_study.py for each
# Logs each run to run_llava_ablation_<mode>.log

modes=(random)
prompt="Which label does not belong to this image? Answer the question with a single word from [{labels}]."
batch_size=64
gpu=4

cd "$(dirname "$0")"

for mode in "${modes[@]}"; do
    echo "=== Running mode: $mode ==="
    # logfile="run_llava_ablation_${mode}.log"
    python llava_ablation_study.py \
        --dataset cifar20 \
        --prompt "$prompt" \
        --batch_size $batch_size \
        --gpu $gpu \
        --mode "$mode" 
done

echo "All runs finished."
