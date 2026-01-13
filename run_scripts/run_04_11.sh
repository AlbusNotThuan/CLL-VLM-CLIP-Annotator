#!/bin/bash
# run_04_11.sh
# Tự động chạy 4 trường hợp với CUDA_VISIBLE_DEVICES=5

export CUDA_VISIBLE_DEVICES=2

DATASET="cifar20"
INPUT_CSV="clip_similarity_cifar20.csv"

# 1️⃣ Most relevant (clip_llava.py)
echo "🚀 Running clip_llava.py (Top Most)..."
python clip_llava.py \
    --dataset "$DATASET" \
    --input_csv "$INPUT_CSV" \
    --prompt "Among the following labels, which one is the most relevant of this image? Answer with a single word from [{labels}]." \
    --output_csv "cifar20_top_most.csv" \
    --batch_size 90

# 2️⃣ Most relevant (clip_llava_least.py)
echo "🚀 Running clip_llava_least.py (Bottom Most)..."
python clip_llava_least.py \
    --dataset "$DATASET" \
    --input_csv "$INPUT_CSV" \
    --prompt "Among the following labels, which one is the most relevant of this image? Answer with a single word from [{labels}]." \
    --output_csv "cifar20_bottom_most.csv" \
    --batch_size 90

# 3️⃣ Least relevant (clip_llava.py)
echo "🚀 Running clip_llava.py (Top Least)..."
python clip_llava.py \
    --dataset "$DATASET" \
    --input_csv "$INPUT_CSV" \
    --prompt "Among the following labels, which one is the least relevant of this image? Answer with a single word from [{labels}]." \
    --output_csv "cifar20_top_least.csv" \
    --batch_size 90

# 4️⃣ Least relevant (clip_llava_least.py)
echo "🚀 Running clip_llava_least.py (Bottom Least)..."
python clip_llava_least.py \
    --dataset "$DATASET" \
    --input_csv "$INPUT_CSV" \
    --prompt "Among the following labels, which one is the least relevant of this image? Answer with a single word from [{labels}]." \
    --output_csv "cifar20_bottom_least.csv" \
    --batch_size 90

echo "✅ All four runs completed successfully!"