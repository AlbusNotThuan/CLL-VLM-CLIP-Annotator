#!/bin/bash

# =====================================================
# VLM binary prompting – main4.py (không LLaVA)
# GPU 4 và GPU 5 chạy SONG SONG
# GPU 4: qwen3_8b
# GPU 5: qwen 2.5 → qwen3_2b (tuần tự)
# =====================================================

set -e

PROJECT_ROOT="/tmp2/maitanha/vgu/cll_vlm"
CODE_DIR="${PROJECT_ROOT}/cll_vlm"
DATA_NAME="cifar100"
PROMPT_TYPE="label_description"

echo "=============================================="
echo "VLM Run – main4.py (GPU 4 // GPU 5)"
echo "GPU 4: qwen3_8b"
echo "GPU 5: qwen 2.5 → qwen3_2b"
echo "Working dir: ${CODE_DIR}"
echo "=============================================="

cd "${CODE_DIR}"

# -----------------------------
# GPU 4: qwen3_8b
# -----------------------------
(
  export CUDA_VISIBLE_DEVICES=4
  echo "[GPU 4] Bắt đầu: qwen3_8b"
  python main4.py \
    --data_name "${DATA_NAME}" \
    --model_name qwen3_8b \
    --batch_size 128 \
    --prompt_type "${PROMPT_TYPE}"
  echo "[GPU 4] Xong."
) > "${PROJECT_ROOT}/run_scripts/log_gpu4.log" 2>&1 &
PID4=$!

# -----------------------------
# GPU 5: qwen 2.5 rồi qwen3_2b (chạy trong subshell, nền)
# -----------------------------
(
  export CUDA_VISIBLE_DEVICES=5
  echo "[GPU 5] Bắt đầu: qwen 2.5"
  python main4.py \
    --data_name "${DATA_NAME}" \
    --model_name qwen \
    --batch_size 128 \
    --prompt_type "${PROMPT_TYPE}"
  echo "[GPU 5] Xong qwen 2.5 → chạy qwen3_2b"
  python main4.py \
    --data_name "${DATA_NAME}" \
    --model_name qwen3_2b \
    --batch_size 256 \
    --prompt_type "${PROMPT_TYPE}"
  echo "[GPU 5] Tất cả xong."
) > "${PROJECT_ROOT}/run_scripts/log_gpu5.log" 2>&1 &
PID5=$!

echo "Đã khởi chạy: GPU4 PID=${PID4}, GPU5 PID=${PID5}"
echo "Log GPU 4: run_scripts/log_gpu4.log"
echo "Log GPU 5: run_scripts/log_gpu5.log"
echo "Đang chờ cả hai hoàn thành..."
wait $PID4 && echo "[GPU 4] Process kết thúc thành công." || echo "[GPU 4] Process lỗi (exit $?)."
wait $PID5 && echo "[GPU 5] Process kết thúc thành công." || echo "[GPU 5] Process lỗi (exit $?)."
echo "=============================================="
echo "Cả hai GPU đã xong."
echo "=============================================="
