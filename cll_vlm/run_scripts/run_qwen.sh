#!/bin/bash
# ====================================================
# QWEN VL 7B Multi-GPU Sequential Runner
# Uses all GPUs together (device_map="auto")
# ====================================================

MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
DATA="cifar10"
BATCH_SIZE=16  # Start conservative for QWEN memory testing
OUTPUT_DIR="/home/maitanha/cll_vlm/cll_vlm/ol_cll_logs/cifar10"
PYFILE="/home/maitanha/cll_vlm/cll_vlm/main2.py"
EXP_ID="qwen_001"   # change this for new experiment batches

# Make sure logs dir exists
mkdir -p "$OUTPUT_DIR"

# Use GPU 2 (adjust as needed)
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ========================
# Prompts
# ========================
prompts=(
"
You are identifying whether a label matches an image.
Example 1:
<Image of a truck>
Label: automobile
Question: Does this label exactly describe the main object shown in the image?
Answer: NO
Example 2:
<Image of a deer>
Label: horse
Question: Does this label exactly describe the main object shown in the image?
Answer: NO  
Example 3:
<Image of a cat>
Label: cat
Question: Does this label exactly describe the main object shown in the image?
Answer: YES
Example 4:
<Image of an airplane>
Label: airplane
Question: Does this label exactly describe the main object shown in the image?
Answer: YES
Example 5:
<Image of a ship>
Label: truck
Question: Does this label exactly describe the main object shown in the image?
Answer: NO

Now answer for this case:
Label: '{label}'
Does this label exactly describe the main object shown in the image?
Answer with one word only: YES or NO.
"
)

# ========================
# Run all prompts sequentially
# ========================
day=$(date +%d)
month=$(date +%m)

echo "🚀 Running QWEN VL 7B experiments on GPU"
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
        --model_type qwen \
        --model_path "$MODEL_PATH" \
        --data "$DATA" \
        --batch_size $BATCH_SIZE \
        --baseprompt "$prompt" \
        --csv_name "$csv_file" \
        | tee "$log_file"

    echo "✅ Finished prompt ${run_id}"
    echo "--------------------------------------------"
done

echo "🎯 All experiments completed. Results saved in $OUTPUT_DIR"
