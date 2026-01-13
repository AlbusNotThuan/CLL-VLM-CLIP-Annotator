# QWEN VL 7B Integration Guide

## Overview
This implementation adds QWEN VL 7B from HuggingFace as an alternative VQA model alongside LLaVA, maintaining interface compatibility for seamless model switching.

## Files Added/Modified

### New Files
1. **`cll_vlm/models/qwen_classifier.py`**
   - Implements `QWENClassifier` class with identical interface to `LLaVAClassifier`
   - Key methods:
     - `__init__(model_path, baseprompt, device)` - Initialize model and processor
     - `predict(images, labels)` - Binary YES/NO prediction for batch
     - `predict_best_label_batch(images, label_option_list, baseprompt)` - Multi-choice selection
   - Uses `Qwen2VLForConditionalGeneration` and `AutoProcessor` from transformers
   - Implements QWEN-specific conversation format (role-based messages instead of [INST] tags)

2. **`cll_vlm/run_qwen.sh`**
   - Shell script for running QWEN experiments
   - Mirrors `run_llava.sh` pattern
   - Conservative batch size (16) for initial memory testing
   - Configured for GPU 2 by default

3. **`cll_vlm/test_qwen_integration.py`**
   - Integration test script
   - Tests with 4 CIFAR-10 samples
   - Validates model loading, prompt formatting, and answer extraction

### Modified Files
1. **`cll_vlm/main2.py`**
   - Added `--model_type` argument (choices: `llava`, `qwen`)
   - Conditional model instantiation based on `model_type`
   - Automatic default model path selection:
     - `llava` → `llava-hf/llava-v1.6-mistral-7b-hf`
     - `qwen` → `Qwen/Qwen2-VL-7B-Instruct`

2. **`cll_vlm/models/__init__.py`**
   - Added `QWENClassifier` to package exports
   - Updated `__all__` list

## Dependencies

### Required Packages (installed in `venv_cll_qwen`)
```bash
pip install transformers qwen-vl-utils pillow torchvision accelerate
```

### Installed Versions
- `transformers==4.57.1`
- `qwen-vl-utils==0.0.14`
- `pillow==12.0.0`
- `torchvision==0.24.1`
- `accelerate==1.11.0`
- `torch==2.9.1`

## Usage

### 1. Run with QWEN (via script)
```bash
cd /home/maitanha/cll_vlm/cll_vlm
source /home/maitanha/cll_vlm/venv_cll_qwen/bin/activate
./run_qwen.sh
```

### 2. Run with QWEN (direct command)
```bash
python main2.py \
    --model_type qwen \
    --model_path Qwen/Qwen2-VL-7B-Instruct \
    --data cifar10 \
    --batch_size 16 \
    --csv_name qwen_test.csv
```

### 3. Run integration test
```bash
source /home/maitanha/cll_vlm/venv_cll_qwen/bin/activate
python test_qwen_integration.py
```

### 4. Compare with LLaVA
```bash
# Run LLaVA
python main2.py --model_type llava --batch_size 64 --data cifar10

# Run QWEN
python main2.py --model_type qwen --batch_size 16 --data cifar10
```

## Key Implementation Details

### 1. Prompt Format Differences
**LLaVA:**
```python
"[INST]<image>\n{prompt}[/INST]"
```

**QWEN:**
```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": prompt}
        ]
    }
]
```

### 2. Image Processing
**LLaVA:**
- Direct batch processing with `processor(images=..., text=...)`

**QWEN:**
- Uses `process_vision_info()` from `qwen_vl_utils`
- Separates image and video inputs
- Applies chat template before processing

### 3. Output Decoding
**LLaVA:**
- Decodes full output including input
- Extracts answer using `_extract_answer()`

**QWEN:**
- Decodes only generated tokens (skips input)
- Uses same extraction logic for consistency

### 4. Memory Considerations
- **LLaVA batch size:** 64-96 (tested stable)
- **QWEN batch size:** Start with 16 (conservative)
- Both use `torch.float16` for efficiency
- Both support `device_map="auto"` for multi-GPU

## Testing Checklist

- [x] Create `QWENClassifier` with matching interface
- [x] Add model selection to `main2.py`
- [x] Update `models/__init__.py` exports
- [x] Create `run_qwen.sh` script
- [x] Install dependencies in `venv_cll_qwen`
- [ ] **Run integration test** (`test_qwen_integration.py`)
- [ ] **Run small-scale experiment** (100 samples)
- [ ] **Compare results with LLaVA** on same prompts
- [ ] **Optimize batch size** for memory/speed tradeoff

## Next Steps

1. **Run Integration Test:**
   ```bash
   source /home/maitanha/cll_vlm/venv_cll_qwen/bin/activate
   export CUDA_VISIBLE_DEVICES=2
   python test_qwen_integration.py
   ```

2. **Run Small-Scale Test:**
   - Modify `run_qwen.sh` to use subset of data
   - Test prompt formatting and answer extraction
   - Verify CSV output format

3. **Extend to Other Scripts:**
   - `llava_ablation_study.py` → support QWEN
   - `clip_llava.py` → support QWEN
   - Create QWEN-specific ablation scripts

4. **Optimize Performance:**
   - Test different batch sizes (16, 32, 48)
   - Measure GPU memory usage with `nvidia-smi`
   - Compare inference speed with LLaVA

5. **Prompt Engineering:**
   - Test QWEN-specific prompt formats
   - Compare few-shot vs zero-shot performance
   - Analyze answer quality (YES/NO accuracy)

## Troubleshooting

### Issue: Import error for `qwen_vl_utils`
**Solution:** Install with `pip install qwen-vl-utils`

### Issue: CUDA out of memory
**Solution:** Reduce batch size in `run_qwen.sh` (try 8 or 4)

### Issue: Model downloads slowly
**Solution:** Pre-download model: `huggingface-cli download Qwen/Qwen2-VL-7B-Instruct`

### Issue: Wrong answer format
**Solution:** Check `_extract_answer()` logic and test with different prompts

### Issue: Different results from LLaVA
**Expected:** QWEN and LLaVA may have different capabilities and biases. Compare on multiple prompts and analyze patterns.

## Performance Benchmarks (To Be Filled)

| Model | Batch Size | GPU Memory | Time/Batch | Accuracy (CIFAR-10) |
|-------|-----------|------------|------------|---------------------|
| LLaVA | 64        | ~12 GB     | ~X sec     | X%                  |
| QWEN  | 16        | ~? GB      | ~? sec     | ?%                  |

Run benchmarks and update this table after testing.
