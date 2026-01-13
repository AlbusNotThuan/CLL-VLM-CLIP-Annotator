#!/bin/bash
# ====================================================
# LLaVA 7B Multi-GPU Sequential Runner
# Uses all GPUs together (device_map="auto")
# ====================================================

MODEL_PATH="llava-hf/llava-v1.6-mistral-7b-hf"
DATA_ROOT="/home/maitanha/cll_vlm/cll_vlm/data/cifar10"
BATCH_SIZE=96
OUTPUT_DIR="/home/maitanha/cll_vlm/cll_vlm/logs"
PYFILE="/home/maitanha/cll_vlm/cll_vlm/main2.py"
EXP_ID="001"   # change this for new experiment batches

# Make sure logs dir exists
mkdir -p "$OUTPUT_DIR"

# Use all GPUs
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

echo "🚀 Running LLaVA 7B experiments on 1 GPU"
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
