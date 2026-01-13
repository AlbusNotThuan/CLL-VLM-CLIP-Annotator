#!/bin/bash
# ====================================================
# LLaVA 7B Multi-GPU Sequential Runner
# Uses all GPUs together (device_map="auto")
# ====================================================

MODEL_PATH="llava-hf/llava-v1.6-mistral-7b-hf"
DATA_ROOT="/home/maitanha/cll_vlm/cll_vlm/data/cifar10"
BATCH_SIZE=128
OUTPUT_DIR="/home/maitanha/cll_vlm/cll_vlm/logs"
PYFILE="/home/maitanha/cll_vlm/cll_vlm/main2.py"
EXP_ID="003"   # change this for new experiment batches

# Make sure logs dir exists
mkdir -p "$OUTPUT_DIR"

# Use all GPUs
export CUDA_VISIBLE_DEVICES=1 # run on GPU 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ========================
# Prompts
# ========================
prompts=(
"
Is the label '{label}' incorrect for this image? Reply with exactly one word: YES or NO.
"
)

# ========================
# Run all prompts sequentially
# ========================
day=$(date +%d)
month=$(date +%m)

echo "🚀 Running LLaVA 7B experiments on all GPUs (device_map=auto)"
echo "📅 Experiment ID: ${EXP_ID}, Date: ${day}_${month}"

for i in "${!prompts[@]}"; do
    prompt="${prompts[$i]}"
    run_id=$((i+1))
    csv_file="${day}_${month}_${EXP_ID}_p${run_id}.csv"
    log_file="$OUTPUT_DIR/prompt_${run_id}.log"

    echo "🟢 Running prompt ${run_id}/${#prompts[@]}"
    echo "$prompt"
    echo "📁 Output: $csv_file"

    python $PYFILE \
        --model_path "$MODEL_PATH" \
        --data_root "$DATA_ROOT" \
        --batch_size $BATCH_SIZE \
        --output_dir "$OUTPUT_DIR" \
        --baseprompt "$prompt" \
        --csv_name "$csv_file" \
        | tee "$log_file"

    echo "✅ Finished prompt ${run_id}"
    echo "--------------------------------------------"
done

echo "🎯 All experiments completed. Results saved in $OUTPUT_DIR"
