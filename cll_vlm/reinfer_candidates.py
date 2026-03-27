"""
reinfer_candidates.py

Load multi-label result JSON files (from prompt_v1 runs of qwen3_4b on cifar100),
extract candidate_answer per image, and re-query the VLM using prompt v3 or v4
to pick a single best label from the candidates.

Usage example:
  python reinfer_candidates.py \
      --input_json /path/to/qwen3_4b_cifar100_multi_label_lbs5_prompt_v1.json \
      --model_name qwen3_4b \
      --prompt_version v3 \
      --batch_size 256 \
      --output_dir /path/to/output/dir

The script:
  1. Loads the input JSON.
  2. Loads the CIFAR-100 dataset (to retrieve PIL images by img_idx).
  3. For samples where candidate_answer has > 1 label, runs the VLM with the
     selected prompt to pick a single best label.
  4. For samples where candidate_answer has exactly 1 label, copies it directly.
  5. For samples with empty candidate_answer, the answer remains [].
  6. Writes the updated results as a new JSON next to (or in) the output_dir.
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# ── make sure cll_vlm package is importable ──────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models.qwen_classifier import QWENClassifier, extract_multi_label_full
from qwen_vl_utils import process_vision_info


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_v3(cands):
    return (
        "You are given an image and a candidate label list. "
        "Your task is to select the SINGLE candidate that best matches the MAIN object in the image. "
        "Focus only on the main object, not background, context, or secondary objects.\n"
        "The final answer must be exactly one label from the candidate list, or \"NO\" if none matches.\n"
        f"Candidates: ({', '.join(cands)}).\n"
        "First internally compare each candidate against the image. "
        "Then output ONLY one JSON object in this format:\n"
        "{\"answer\": \"one label from candidates or NO\", "
        "\"reason\": \"short explanation\"}\n"
    )


def build_prompt_v4(cands):
    return (
        "Choose the single best label from the following candidate list for the main object in the image.\n"
        f"Candidates: ({', '.join(cands)}).\n"
        "Use only visible evidence. Ignore background and context.\n"
        "Return ONLY a JSON dict and nothing else, formatted as: \n"
        "{\"answer\": \"one label from candidates\", \"reason\": \"short explanation\"}"
    )

def build_prompt_v5(cands):
    return (
        "You are given an image and a candidate label list. "
        "Your task is to select the SINGLE candidate that best matches the MAIN object in the image. "
        # "Focus only on the main object, not background, context, or secondary objects.\n"
        "The final answer must be exactly one label from the candidate list, or \"NO\" if none matches.\n"
        f"Candidates: ({', '.join(cands)}).\n"
        "Then output ONLY one JSON object in this format:\n"
        "{\"answer\": \"one label from candidates or NO\", "
        "\"reason\": \"short explanation\"}\n"
    )


def build_prompt_v6(cands):
    return (
        "Choose the single best label from the following candidate list for the main object in the image.\n"
        f"Candidates: ({', '.join(cands)}).\n"
        # "Use only visible evidence. Ignore background and context.\n"
        "Return ONLY a JSON dict and nothing else, formatted as: \n"
        "{\"answer\": \"one label from candidates\", \"reason\": \"short explanation\"}"
    )


PROMPT_BUILDERS = {
    "v3": build_prompt_v3,
    "v4": build_prompt_v4,
    "v5": build_prompt_v5,
    "v6": build_prompt_v6,  
}


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-100 image loader  (raw, no transform)
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar100_images(data_root: str, train: bool = True):
    """
    Returns a list of PIL images in dataset order (matching img_idx).
    data_root should be the folder that contains 'cifar-100-python/'.
    """
    split = "train" if train else "test"
    data_path = os.path.join(data_root, "cifar-100-python", split)
    with open(data_path, "rb") as f:
        data = pickle.load(f, encoding="bytes")

    raw = data[b"data"]          # shape (N, 3072)
    images = []
    for row in raw:
        img_arr = row.reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)
        images.append(Image.fromarray(img_arr))
    return images


# ─────────────────────────────────────────────────────────────────────────────
# Batch re-inference helper
# ─────────────────────────────────────────────────────────────────────────────

def reinfer_batch(model: QWENClassifier, images, cand_lists, prompt_version: str):
    """
    Given a list of PIL images and corresponding candidate lists,
    run VLM with the chosen prompt version in a single batch.

    Returns list of (final_answer: list[str], reason: str).
    """
    build_prompt = PROMPT_BUILDERS[prompt_version]

    batch_msgs = []
    for img, cands in zip(images, cand_lists):
        prompt = build_prompt(cands)
        batch_msgs.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }])

    texts = [
        model.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in batch_msgs
    ]
    image_inputs, video_inputs = process_vision_info(batch_msgs)
    inputs = model.processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=model.processor.tokenizer.pad_token_id,
            eos_token_id=None,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = model.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    results = []
    for out, cands in zip(output_texts, cand_lists):
        predicted, reason = extract_multi_label_full(out)
        # Validate: keep only labels that are in the candidate list
        cand_set = set(cands)
        predicted_valid = [p for p in predicted if p in cand_set]
        if not predicted_valid and predicted:
            # Fallback: pick the first candidate as a best-effort guess
            predicted_valid = [cands[0]]
        results.append((predicted_valid, reason, out))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── Load existing results ─────────────────────────────────────────────────
    print(f"[INFO] Loading input JSON: {os.path.basename(args.input_json)}")
    with open(args.input_json, "r") as f:
        records = json.load(f)
    print(f"[INFO] Loaded {len(records)} records.")

    # ── Determine output path ─────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    input_stem = Path(args.input_json).stem          # e.g. qwen3_4b_cifar100_multi_label_lbs5_prompt_v1
    # Replace prompt_v1 suffix with new prompt version
    if "_prompt_v1" in input_stem:
        out_stem = input_stem.replace("_prompt_v1", f"_prompt_{args.prompt_version}")
    else:
        out_stem = input_stem + f"_prompt_{args.prompt_version}"
    output_path = os.path.join(args.output_dir, out_stem + ".json")
    print(f"[INFO] Output will be written to: {os.path.basename(output_path)}")

    # ── Resume logic ──────────────────────────────────────────────────────────
    existing_results = []
    if os.path.exists(output_path):
        print(f"[RESUME] Found existing output file: {os.path.basename(output_path)}")
        with open(output_path, "r") as f:
            content = f.read().strip()
        if content:
            try:
                existing_results = json.loads(content)
            except Exception:
                # try recovering
                while content:
                    last_brace = content.rfind("}")
                    if last_brace == -1:
                        break
                    content = content[: last_brace + 1]
                    try:
                        existing_results = json.loads(content + "\n]")
                        break
                    except json.JSONDecodeError:
                        content = content[:-1]
        print(f"[RESUME] Resuming from index {len(existing_results)}.")

    resume_count = len(existing_results)
    records_to_process = records[resume_count:]
    print(f"[INFO] Records remaining to process: {len(records_to_process)}")

    # ── Split records into two groups ─────────────────────────────────────────
    # Group A: candidate_answer has > 1 label → needs VLM re-inference
    # Group B: candidate_answer has 0 or 1 label → no VLM call needed
    needs_reinfer = []
    no_reinfer = []
    for r in records_to_process:
        cands = r.get("candidate_answer", [])
        if len(cands) > 1:
            needs_reinfer.append(r)
        else:
            no_reinfer.append(r)

    print(f"[INFO] Needs re-inference: {len(needs_reinfer)}, No re-inference: {len(no_reinfer)}")

    if len(needs_reinfer) > 0:
        # ── Load CIFAR-100 images ─────────────────────────────────────────────
        print(f"[INFO] Loading CIFAR-100 dataset from {args.cifar100_root} …")
        all_images = load_cifar100_images(args.cifar100_root, train=True)

        # Apply the same shuffle as main4.py uses (seed=42) to map img_idx → image
        # The dataset in main4.py is shuffled with seed=42 but img_idx reflects the
        # position in the shuffled order. We need to reverse-map img_idx to
        # the shuffled dataset. The simplest approach is to re-apply the same shuffle
        # to get the mapping.
        n = len(all_images)
        rng = np.random.default_rng(42)
        shuffled_indices = rng.permutation(n)  # shuffled_indices[i] = original CIFAR index for shuffled position i
        print(f"[INFO] Dataset shuffle applied (seed=42). First 5 mapping: {shuffled_indices[:5]}")

        # ── Load model ────────────────────────────────────────────────────────
        config_path = args.config_path
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model_path = config["models"][args.model_name]["model_url"]

        print(f"[INFO] Loading model {args.model_name} from {model_path} …")
        model = QWENClassifier(model_path=model_path)

        # ── Run re-inference in batches ───────────────────────────────────────
        batch_size = args.batch_size
        reinfer_results = {}   # img_idx → (final_answer, reason, raw_output)
        print(f"[INFO] Running VLM re-inference with prompt {args.prompt_version}, batch_size={batch_size} …")

        for batch_start in tqdm(range(0, len(needs_reinfer), batch_size), desc="Re-inference batches"):
            batch_records = needs_reinfer[batch_start: batch_start + batch_size]

            batch_images = []
            batch_cands = []
            batch_img_idxs = []

            for r in batch_records:
                idx = r["img_idx"]
                orig_idx = int(shuffled_indices[idx])
                batch_images.append(all_images[orig_idx])
                batch_cands.append(r["candidate_answer"])
                batch_img_idxs.append(idx)

            batch_out = reinfer_batch(model, batch_images, batch_cands, args.prompt_version)

            for img_idx, (final_ans, reason, raw_out) in zip(batch_img_idxs, batch_out):
                reinfer_results[img_idx] = (final_ans, reason, raw_out)
    else:
        reinfer_results = {}

    # ── Merge all results back in original order ──────────────────────────────
    print("[INFO] Merging results …")
    merged = list(existing_results)

    for r in records_to_process:
        img_idx = r["img_idx"]
        cands = r.get("candidate_answer", [])

        if len(cands) > 1 and img_idx in reinfer_results:
            final_ans, reason, raw_out = reinfer_results[img_idx]
            new_r = dict(r)
            new_r["new_answer"] = final_ans
            if reason:
                new_r["new_reason"] = reason
            new_r[f"raw_reinfer_{args.prompt_version}"] = raw_out
        elif len(cands) == 1:
            new_r = dict(r)
            new_r["new_answer"] = cands  # already a single-element list
        else:
            new_r = dict(r)
            new_r["new_answer"] = r.get("answer", [])

        merged.append(new_r)

    # ── Write output ──────────────────────────────────────────────────────────
    print(f"[INFO] Writing {len(merged)} records to {os.path.basename(output_path)} …")
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    print("[DONE]")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-run VLM on candidate answers (prompt v3 / v4 / v5 / v6 )")

    parser.add_argument(
        "--input_json", type=str, required=True,
        help="Path to input JSON file (prompt_v1 result).",
    )
    parser.add_argument(
        "--model_name", type=str, default="qwen3_4b",
        choices=["qwen", "qwen3_2b", "qwen3_4b", "qwen3_8b", "qwen3_30b_a3b"],
        help="Model name (key in config.yaml).",
    )
    parser.add_argument(
        "--prompt_version", type=str, required=True,
        choices=["v3", "v4", "v5", "v6"],  
        help="Which prompt version to use for re-inference.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/tmp2/maitanha/vgu/cll_vlm/cll_vlm/ol_cll_logs/multi_label/json/cifar100",
        help="Directory to write output JSON.",
    )
    parser.add_argument(
        "--cifar100_root", type=str,
        default="/tmp2/maitanha/vgu/cll_vlm/cll_vlm/data/cifar100",
        help="Root dir containing cifar-100-python/ folder.",
    )
    parser.add_argument(
        "--config_path", type=str,
        default="/tmp2/maitanha/vgu/cll_vlm/cll_vlm/config/config.yaml",
        help="Path to config YAML.",
    )

    args = parser.parse_args()
    main(args)