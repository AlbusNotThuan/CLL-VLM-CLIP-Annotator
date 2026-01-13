#!/bin/bash

# ================= GPU =================
export CUDA_VISIBLE_DEVICES=0

# ================= CONFIG =================
PYTHON_FILE=/home/maitanha/cll_vlm/cll_vlm/vlm_describe.py


# ================= RUN 13/12 =================
# setting of output tokens: 32 for not CoT, 512 for CoT
# python ${PYTHON_FILE} \
#     --model_type llava \
#     --data cifar100 \
#     --batch_size 32 \
#     --prompt "Identify the main object in this image." \
#     --output_path /home/maitanha/cll_vlm/cll_vlm/ol_cll_logs/describe/cifar100_llava_prompt1.csv

python ${PYTHON_FILE} \
    --model_type llava \
    --data cifar100 \
    --batch_size 64 \
    --prompt "What type of object is in this photo?" \
    --output_path /home/maitanha/cll_vlm/cll_vlm/ol_cll_logs/describe/cifar100_llava_prompt2.csv

python ${PYTHON_FILE} \
    --model_type llava \
    --data cifar100 \
    --batch_size 64 \
    --prompt "Can you name the specific item pictured here?" \
    --output_path /home/maitanha/cll_vlm/cll_vlm/ol_cll_logs/describe/cifar100_llava_prompt3.csv

python ${PYTHON_FILE} \
    --model_type llava \
    --data cifar100 \
    --batch_size 64 \
    --prompt "Identify the object in this image." \
    --output_path /home/maitanha/cll_vlm/cll_vlm/ol_cll_logs/describe/cifar100_llava_prompt4.csv

# python ${PYTHON_FILE} \
#     --model_type llava \
#     --data cifar100 \
#     --batch_size 128 \
#     --prompt "Is the main object of this image an <label name>? All the answer should be in json format {'answer':'Yes or No','reason':'reason of the answers'}" \
#     --output_path /home/maitanha/cll_vlm/cll_vlm/ol_cll_logs/describe/cifar100_llava_prompt5.csv