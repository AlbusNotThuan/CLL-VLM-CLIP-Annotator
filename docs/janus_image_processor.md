# Janus Image Processor Reference (venv_cll_qwen)

This document describes the exact image processor used by Janus in this repo, its runtime spec, output format, and what must stay compatible if you swap to a faster processor.

## 1) Where it is used in this repo

- Janus wrapper load path:
  - `cll_vlm/models/janus_classifier.py` (loads `VLChatProcessor.from_pretrained(...)`)
- Multimodal call path:
  - `cll_vlm/models/janus_classifier.py` calls processor with `force_batchify=True`
  - then calls `self.model.prepare_inputs_embeds(**prepared)`

## 2) Actual processor implementation (installed package)

In `venv_cll_qwen`, Janus resolves to:

- Package root:
  - `/tmp2/maitanha/vgu/cll_vlm/venv_cll_qwen/lib/python3.13/site-packages/janus`
- Processor class:
  - `/tmp2/maitanha/vgu/cll_vlm/venv_cll_qwen/lib/python3.13/site-packages/janus/models/processing_vlm.py`
  - `class VLChatProcessor` (inherits `ProcessorMixin`)
- Image processor class:
  - `/tmp2/maitanha/vgu/cll_vlm/venv_cll_qwen/lib/python3.13/site-packages/janus/models/image_processing_vlm.py`
  - `class VLMImageProcessor`
- Vision embedding side (consumes pixel tensors + masks):
  - `/tmp2/maitanha/vgu/cll_vlm/venv_cll_qwen/lib/python3.13/site-packages/janus/models/modeling_vlm.py`
  - `MultiModalityCausalLM.prepare_inputs_embeds(...)`

## 3) Runtime spec observed for deepseek-ai/Janus-Pro-7B

From `VLChatProcessor.from_pretrained("deepseek-ai/Janus-Pro-7B", local_files_only=True)` in `venv_cll_qwen`:

- Processor class: `VLChatProcessor`
- Image processor class: `VLMImageProcessor`
- Tokenizer class: `LlamaTokenizerFast`
- `model_input_names`: `['pixel_values']`
- `image_size`: `384`
- `min_size`: `14`
- `image_mean`: `[0.5, 0.5, 0.5]`
- `image_std`: `[0.5, 0.5, 0.5]`
- `rescale_factor`: `1/255` (`0.00392156862745098`)
- `do_normalize`: `True`
- `num_image_tokens`: `576`
- `sft_format`: `deepseek`
- Special tags:
  - `<image_placeholder>`
  - `<begin_of_image>`
  - `<end_of_image>`
  - `<｜▁pad▁｜>`

Important warning observed at runtime:

- `use_fast=True` falls back to slow processor because Janus `VLMImageProcessor` has no fast variant registered.

## 4) Preprocessing algorithm (VLMImageProcessor)

For each input image:

1. Read PIL image as RGB.
2. Compute resized shape preserving aspect ratio so longest side becomes `image_size=384`.
3. Enforce each resized side is at least `min_size=14`.
4. Resize with bicubic + antialias.
5. Pad to square (`384 x 384`) using background color from mean (`(127,127,127)` here).
6. Convert HWC -> CHW.
7. Rescale `[0,255] -> [0,1]` using `1/255`.
8. Normalize per channel with mean/std `[0.5,0.5,0.5]`.

Resulting pixel range is approximately `[-1, 1]`.

## 5) Output format expected by Janus model

`VLChatProcessor(..., force_batchify=True)` returns a batched object with:

- `input_ids`: `torch.int64`, shape `[B, T]`
- `attention_mask`: `torch.int64`, shape `[B, T]`
- `pixel_values`: `torch.float32`, shape `[B, N, 3, 384, 384]`
- `images_seq_mask`: `torch.bool`, shape `[B, T]`
- `images_emb_mask`: `torch.bool`, shape `[B, N, 576]`
- `sft_format`: list of formatted prompt strings

For one image/sample test run:

- `input_ids`: `(1, 628)`
- `attention_mask`: `(1, 628)`
- `pixel_values`: `(1, 1, 3, 384, 384)`
- `images_seq_mask`: `(1, 628)`
- `images_emb_mask`: `(1, 1, 576)`
- image token count matches in both masks: `576`

## 6) Config files used by from_pretrained

Resolved from local HF cache for Janus-Pro-7B:

- `preprocessor_config.json`
- `processor_config.json`
- `tokenizer_config.json`
- `tokenizer.json`
- `config.json`

Notes:

- `tokenizer.model` was not present in local cache, but load succeeded with `tokenizer.json` + fast tokenizer path.
- If tokenizer loading fails when changing tokenizer args (`legacy`, `use_fast`), verify these files exist and are consistent in your cache/snapshot.

## 7) What you must keep compatible when swapping processor

If your goal is speed, these invariants are critical:

1. Keep output tensor shape: `[B, N, 3, 384, 384]`.
2. Keep normalization semantics equivalent to current model expectation (mean/std = 0.5 and scale = 1/255).
3. Keep `num_image_tokens = 576` aligned with vision patch output and token masks.
4. Keep `images_emb_mask` shape `[B, N, 576]` and `images_seq_mask` placement aligned with `<image_placeholder>` expansion.
5. Keep image channel order RGB and CHW layout.

If you change image resolution or patch/token count, the language-side replacement path in `prepare_inputs_embeds` can misalign and break generation.

## 8) Practical options to get faster preprocessing

### Option A (lowest risk)

Keep Janus logic and replace only resize/normalize backend with a faster equivalent while preserving exact outputs:

- Keep `384` square target.
- Keep mean/std and scale.
- Keep CHW float output and mask behavior unchanged.

### Option B

Precompute transformed tensors outside Janus with your own pipeline, then adapt Janus processor to accept precomputed `pixel_values` (requires code changes to processor flow, more invasive).

### Option C (not currently available directly)

Use HF `use_fast=True` image processor path. For Janus-Pro-7B this currently falls back, because no fast image processor class is registered for `VLMImageProcessor`.

## 9) Implementation pointers if you decide to modify

Primary edit points:

- `cll_vlm/models/janus_classifier.py`
  - Where processor is created and used.
- `.../site-packages/janus/models/image_processing_vlm.py`
  - Image resize/pad/normalize implementation.
- `.../site-packages/janus/models/processing_vlm.py`
  - Packing text/image tensors and masks.

Recommended approach:

- Do not edit `site-packages` directly long-term.
- Fork/override processor behavior in project code, then inject it into `self.processor.image_processor` after load, or vendor a custom processor module.

## 10) Quick validation checklist after swap

After implementing a new processor, validate on 5-10 random images:

- Same tensor shape and dtype.
- Similar value distribution (`min`, `max`, mean/std per channel).
- No mismatch in image token counts (`sum(images_seq_mask) == sum(images_emb_mask)`).
- No accuracy regression on a small benchmark slice.
