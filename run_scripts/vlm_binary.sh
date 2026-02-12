#!/bin/bash

# =====================================================
# VLM binary prompting – main4.py
# Running cifar20 dataset on GPU 0, 1, and 2 in PARALLEL
# GPU 0: qwen (bs 256)
# GPU 1: qwen3_2b (bs 512)
# GPU 2: qwen3_8b (bs 256)
# =====================================================

set -e

PROJECT_ROOT="/tmp2/maitanha/vgu/cll_vlm"
CODE_DIR="${PROJECT_ROOT}/cll_vlm"
DATA_NAME="cifar20"
PROMPT_TYPE="binary"
CUSTOM_NAME="processed_labels"

echo "=============================================="
echo "VLM Run – main4.py (GPU 0 // 1 // 2)"
echo "Dataset: ${DATA_NAME}"
echo "Output suffix: ${CUSTOM_NAME}"
echo "Starting parallel runs..."
echo "=============================================="

cd "${CODE_DIR}"

# GPU 0: qwen
(
  export CUDA_VISIBLE_DEVICES=0
  echo "[GPU 0] Starting qwen (bs 512)"
  python main4.py \
    --data_name "${DATA_NAME}" \
    --model_name qwen \
    --batch_size 512 \
    --prompt_type "${PROMPT_TYPE}" \
    --custom_output_name "${CUSTOM_NAME}"
  echo "[GPU 0] Completed."
) > "${PROJECT_ROOT}/run_scripts/log_gpu0_cifar20.log" 2>&1 &
PID0=$!

# GPU 1: qwen3_2b
(
  export CUDA_VISIBLE_DEVICES=1
  echo "[GPU 1] Starting qwen3_2b (bs 1024)"
  python main4.py \
    --data_name "${DATA_NAME}" \
    --model_name qwen3_2b \
    --batch_size 1024 \
    --prompt_type "${PROMPT_TYPE}" \
    --custom_output_name "${CUSTOM_NAME}"
  echo "[GPU 1] Completed."
) > "${PROJECT_ROOT}/run_scripts/log_gpu1_cifar20.log" 2>&1 &
PID1=$!

# GPU 2: qwen3_8b
(
  export CUDA_VISIBLE_DEVICES=2
  echo "[GPU 2] Starting qwen3_8b (bs 512)"
  python main4.py \
    --data_name "${DATA_NAME}" \
    --model_name qwen3_8b \
    --batch_size 512 \
    --prompt_type "${PROMPT_TYPE}" \
    --custom_output_name "${CUSTOM_NAME}"
  echo "[GPU 2] Completed."
) > "${PROJECT_ROOT}/run_scripts/log_gpu2_cifar20.log" 2>&1 &
PID2=$!

echo "Launched GPUs: 0 (PID $PID0), 1 (PID $PID1), 2 (PID $PID2)"
echo "Logs are available at run_scripts/log_gpu*_cifar20.log"

echo "Waiting for all processes to complete..."
wait $PID0 && echo "[GPU 0] SUCCESS" || echo "[GPU 0] FAILED"
wait $PID1 && echo "[GPU 1] SUCCESS" || echo "[GPU 1] FAILED"
wait $PID2 && echo "[GPU 2] SUCCESS" || echo "[GPU 2] FAILED"

echo "=============================================="
echo "All cifar20 runs finished."
echo "=============================================="
