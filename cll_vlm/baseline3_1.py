"""
Baseline 3.1: CLIP → VLM two-stage pipeline.

Stage 1 (CLIP): Given an image and all dataset labels, compute cosine similarity
                with "A photo of a {label}" prompts and select top-k candidates.
Stage 2 (VLM) : Feed the image + top-k candidates to qwen3_2b and ask it to
                pick the single best label.

Output path:
  results/baseline3/baseline3_1/{dataset}/
  baseline3_1_{clip_model}_{vlm_model}_{dataset}_topk{K}[_{custom}].json
"""

import torch
import os
import sys
import json
import re
import yaml
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project root resolution (same trick as main4.py)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # cll_vlm/
ROOT_DIR = os.path.dirname(SCRIPT_DIR)                    # repo root

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from dataset.cifar10 import CIFAR10Dataset
from dataset.cifar20 import CIFAR20Dataset, CIFAR100Dataset
from dataset.tiny200 import Tiny200Dataset
from models.clip_model import CLIPModel
from models.qwen_classifier import QWENClassifier, extract_multi_label_full
from qwen_vl_utils import process_vision_info

# ---------------------------------------------------------------------------
# CLIP model key → official name mapping
# ---------------------------------------------------------------------------
CLIP_MODEL_MAP = {
    "vitb32": "ViT-B/32",
    "vitl14": "ViT-L/14",
}

VLM_MODEL_KEYS = ["llava", "llava_13b", "qwen", "qwen3_2b", "qwen3_4b", "qwen3_8b", "qwen3_30b_a3b"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)


def load_dataset(data_name: str, data_path: str, data_mode: str = "train"):
    train = (data_mode == "train")
    if data_name == "cifar10":
        dataset = CIFAR10Dataset(root=data_path, train=train, transform=None)
        fine_classes = list(dataset.classes)
    elif data_name == "cifar20":
        dataset = CIFAR20Dataset(root=data_path, train=train, transform=None)
        fine_classes = [CIFAR20Dataset.preprocess_label(l) for l in dataset.classes]
    elif data_name == "cifar100":
        dataset = CIFAR100Dataset(root=data_path, train=train, transform=None)
        fine_classes = [CIFAR100Dataset.preprocess_label(l) for l in dataset.get_fine_classes()]
    elif data_name == "tiny200":
        dataset = Tiny200Dataset(root=data_path, train=train, transform=None)
        fine_classes = list(dataset.classes)
    else:
        raise ValueError(f"Unsupported dataset: {data_name}")
    return dataset, fine_classes


def sanitize_clip_name(key: str) -> str:
    """Return the user-facing key (already clean, e.g. 'vitb32')."""
    return key


def build_output_path(base_dir: str, clip_key: str, vlm_model: str, data_name: str,
                      topk: int, custom: str, test: bool) -> str:
    name_parts = [
        "baseline3_1",
        sanitize_clip_name(clip_key),
        vlm_model,
        data_name,
        f"topk{topk}",
    ]
    if test:
        name_parts.append("test")
    if custom:
        name_parts.append(custom)
    filename = "_".join(name_parts) + ".json"
    out_dir = os.path.join(base_dir, "results", "baseline3", "baseline3_1", data_name)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


# ---------------------------------------------------------------------------
# CLIP top-k selection
# ---------------------------------------------------------------------------

@torch.no_grad()
def clip_topk(clip_model: CLIPModel, images: list,
              label_texts: list, text_feats: torch.Tensor, topk: int):
    """Return (list_of_topk_labels, list_of_topk_scores) for a batch of images."""
    img_feats = clip_model.encode_image(images)                      # (B, D)
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
    txt_feat = text_feats / text_feats.norm(dim=-1, keepdim=True)  # (N, D)
    sims = img_feats @ txt_feat.T                                  # (B, N)
    k = min(topk, len(label_texts))
    topk_scores, topk_indices = sims.topk(k, dim=-1)
    
    batch_topk_labels = []
    batch_topk_scores = []
    for i in range(len(images)):
        batch_topk_labels.append([label_texts[idx] for idx in topk_indices[i].tolist()])
        batch_topk_scores.append(topk_scores[i].tolist())
        
    return batch_topk_labels, batch_topk_scores


# ---------------------------------------------------------------------------
# VLM inference for a batch of (image, candidates) pairs
# ---------------------------------------------------------------------------

def vlm_pick(vlm_model: QWENClassifier,
             images: list, candidate_lists: list) -> list:
    """
    Ask VLM to pick one label from each image's candidates.
    Returns list of (answer_str_or_None, reason_str).
    """
    if not images:
        return []

    batch_msgs = []
    for img, cands in zip(images, candidate_lists):
        prompt = (
            "You are given an image. "
            "Examine the image carefully and identify which objects from the candidate list are present.\n"
            f"Candidates: ({', '.join(cands)}).\n"
            "From this list, choose only ONE label that is most likely present in the image. "
            "Do not include any label that is not in the candidate list. "
            "If you think none of the candidates are present, reply with exactly \"NO\".\n"
            "Provide a short reason for your answer.\n"
            "Your answer should be ONLY a JSON dict and nothing else, formatted as: "
            "{\"answer\": \"your chosen label\" or \"NO\", \"reason\": \"short explanation\"}"
            "Please don't reply in other formats."
        )
        batch_msgs.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }])

    texts = [
        vlm_model.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in batch_msgs
    ]
    is_qwen = vlm_model.__class__.__name__ == "QWENClassifier"
    
    if is_qwen:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(batch_msgs)
        inputs = vlm_model.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(vlm_model.device)
    else:
        # Standard HuggingFace implementations (like LLaVA) use images directly
        inputs = vlm_model.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(vlm_model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(vlm_model.model.dtype)

    # Resolve tokenizer dynamically
    tokenizer = getattr(vlm_model.processor, 'tokenizer', vlm_model.processor)
    pad_token_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, 'eos_token_id', None) # Fallback if pad_token_id is not set

    with torch.no_grad():
        generated_ids = vlm_model.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=None, # Keep eos_token_id as None as per original code
        )

    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = vlm_model.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    results = []
    for raw in output_texts:
        labels, reason = extract_multi_label_full(raw)
        answer = labels[0] if labels else "NO"
        results.append((answer, reason if isinstance(reason, str) else ""))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    config = load_config(args.config_path)
    data_cfg = config["data"]
    data_mode = data_cfg.get("mode", "train")
    data_path = data_cfg["paths"][args.data_name]
    shuffle_seed = data_cfg.get("shuffle_seed", 42)

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    dataset, fine_classes = load_dataset(args.data_name, data_path, data_mode)
    label_pool = fine_classes

    original_dataset, shuffled_dataset = dataset.get_shuffled_labels_dataset(seed=shuffle_seed)

    # -----------------------------------------------------------------------
    # Output path & resume logic
    # -----------------------------------------------------------------------
    # Derive a clean short model name for the filename
    vlm_short = args.model_name  # e.g. qwen3_2b

    output_path = build_output_path(
        base_dir=SCRIPT_DIR,
        clip_key=args.clip_model,
        vlm_model=vlm_short,
        data_name=args.data_name,
        topk=args.topk,
        custom=args.custom_output_name or "",
        test=args.test,
    )
    print(f"[INFO] Output path: {output_path}")

    # Resume
    resume_count = 0
    if not args.img_idx and os.path.exists(output_path):
        print(f"[RESUME] Found existing file: {output_path}")
        with open(output_path, "r") as f:
            content = f.read().strip()
        if content:
            valid_data = []
            if content.endswith("]"):
                try:
                    valid_data = json.loads(content)
                except Exception:
                    pass
            if not valid_data:
                while content:
                    last = content.rfind("}")
                    if last == -1:
                        break
                    content = content[: last + 1]
                    try:
                        valid_data = json.loads(content + "\n]")
                        break
                    except json.JSONDecodeError:
                        content = content[:-1]
            if valid_data:
                resume_count = len(valid_data)
                print(f"[RESUME] Resuming from index {resume_count}.")
                with open(output_path, "w") as f:
                    f.write("[\n")
                    for i, item in enumerate(valid_data):
                        if i > 0:
                            f.write(",\n")
                        json.dump(item, f, indent=2)
                    f.flush()

    # -----------------------------------------------------------------------
    # img_idx single-sample mode
    # -----------------------------------------------------------------------
    if args.img_idx is not None:
        single_mode = True
        idx = args.img_idx
        indices = np.array([idx])
    else:
        single_mode = False
        if resume_count > 0:
            indices = np.arange(resume_count, len(shuffled_dataset))
            shuffled_dataset = shuffled_dataset.get_subset_by_indices(indices)
            original_dataset = original_dataset.get_subset_by_indices(indices)
            print(f"[RESUME] Sliced datasets. New size: {len(shuffled_dataset)}")
        if args.test:
            n = min(args.batch_size, len(shuffled_dataset))
            idxs = np.arange(n)
            shuffled_dataset = shuffled_dataset.get_subset_by_indices(idxs)
            original_dataset = original_dataset.get_subset_by_indices(idxs)
            print(f"[DEBUG] --test mode: truncated to first {len(shuffled_dataset)} samples")

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    clip_model_name = CLIP_MODEL_MAP[args.clip_model]
    print(f"[INFO] Loading CLIP '{clip_model_name}' ...")
    clip_model = CLIPModel(model_name=clip_model_name)

    print(f"[INFO] Loading VLM '{args.model_name}' ...")
    if args.model_name in ["llava", "llava_13b"]:
        from models.llava_classifier import LLaVAClassifier
        model_path = config["models"][args.model_name]["model_url"]
        vlm_model = LLaVAClassifier(model_path=model_path)
    elif args.model_name in ["qwen", "qwen3_2b", "qwen3_4b", "qwen3_8b", "qwen3_30b_a3b"]:
        model_path = config["models"][args.model_name]["model_url"]
        vlm_model = QWENClassifier(model_path=model_path)
    else:
        raise ValueError(f"Unsupported model '{args.model_name}'.")

    # -----------------------------------------------------------------------
    # Pre-compute CLIP text features for the label pool (once)
    # -----------------------------------------------------------------------
    print(f"[INFO] Pre-computing CLIP text features for {len(label_pool)} labels ...")
    text_prompts = [f"A photo of a {lbl}" for lbl in label_pool]
    text_feats = clip_model.encode_text(text_prompts)  # (N, D)
    print("[INFO] CLIP text features ready.")

    # -----------------------------------------------------------------------
    # Single-sample mode
    # -----------------------------------------------------------------------
    if single_mode:
        img, true_lbl_idx = original_dataset[idx]
        _, shuf_lbl_idx = shuffled_dataset[idx]
        true_label = fine_classes[true_lbl_idx]
        shuffled_label = fine_classes[shuf_lbl_idx]

        topk_labels_batch, topk_scores_batch = clip_topk(clip_model, [img], label_pool, text_feats, args.topk)
        topk_labels = topk_labels_batch[0]
        topk_scores = topk_scores_batch[0]
        print(f"[CLIP] img_idx={idx} | true_label={true_label} | CLIP top-{args.topk}: {list(zip(topk_labels, [f'{s:.4f}' for s in topk_scores]))}")

        vlm_results = vlm_pick(vlm_model, [img], [topk_labels])
        answer, reason = vlm_results[0]
        print(f"[VLM]  answer={answer} | reason={reason}")

        result = {
            "img_idx": idx,
            "true_label": true_label,
            "shuffled_label": shuffled_label,
            "clip_topk_candidates": topk_labels,
            "answer": answer,
            "reason": reason,
        }
        single_out_path = output_path.replace(".json", f"_imgidx{idx}.json")
        os.makedirs(os.path.dirname(single_out_path), exist_ok=True)
        with open(single_out_path, "w") as f:
            json.dump([result], f, indent=2)
        print(f"[INFO] Saved single-sample result to: {single_out_path}")
        return

    # -----------------------------------------------------------------------
    # Batch mode inference
    # -----------------------------------------------------------------------
    dataloader = DataLoader(
        shuffled_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=data_cfg.get("num_workers", 4),
    )

    is_first = (resume_count == 0)
    output_file = open(output_path, "a" if resume_count > 0 else "w")
    if resume_count == 0:
        output_file.write("[\n")

    dataset_pos = resume_count
    local_pos = 0

    try:
        for images, shuffled_labels in tqdm(dataloader, desc="Baseline 3.1 Inference"):
            batch_len = len(images)
            true_labels_idx = [
                original_dataset[i][1]
                for i in range(local_pos, local_pos + batch_len)
            ]
            true_labels = [fine_classes[i] for i in true_labels_idx]
            shuffled_labels_text = [fine_classes[i] for i in shuffled_labels]

            # Stage 1: CLIP top-k for each image using batched encoding
            topk_cands_batch, topk_scores_batch = clip_topk(clip_model, list(images), label_pool, text_feats, args.topk)

            # Stage 2: VLM pick final answer
            vlm_results = vlm_pick(vlm_model, images, topk_cands_batch)

            # Write results
            for i, (answer, reason) in enumerate(vlm_results):
                result = {
                    "img_idx": dataset_pos + i,
                    "true_label": true_labels[i],
                    "shuffled_label": shuffled_labels_text[i],
                    "clip_topk_candidates": topk_cands_batch[i],
                    "answer": answer,
                    "reason": reason,
                }
                if not is_first:
                    output_file.write(",\n")
                else:
                    is_first = False
                json.dump(result, output_file, indent=2)
            output_file.flush()

            local_pos += batch_len
            dataset_pos += batch_len

    finally:
        output_file.write("\n]")
        output_file.close()

    print(f"[INFO] Done. Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="Baseline 3.1: CLIP → VLM two-stage classification")

    parser.add_argument("--data_name", type=str, required=True,
                        choices=["cifar10", "cifar20", "cifar100", "tiny200"])
    parser.add_argument("--model_name", type=str, required=True,
                        choices=VLM_MODEL_KEYS,
                        help="VLM model key (maps to model_url in config)")
    parser.add_argument("--clip_model", type=str, default="vitb32",
                        choices=list(CLIP_MODEL_MAP.keys()),
                        help="CLIP model key: vitb32 or vitl14 (default: vitb32)")
    parser.add_argument("--topk", type=int, default=5,
                        help="Number of CLIP top-k candidates passed to VLM (default: 5)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="DataLoader batch size (default: 64)")
    parser.add_argument("--custom_output_name", type=str, default=None,
                        help="Optional custom suffix for output filename")
    parser.add_argument("--config_path", type=str,
                        default="/tmp2/maitanha/vgu/cll_vlm/cll_vlm/config/config.yaml")
    parser.add_argument("--test", action="store_true", default=False,
                        help="Truncate dataset to first batch_size samples for debugging")
    parser.add_argument("--img_idx", type=int, default=None,
                        help="Run on a single sample by its dataset index")

    args = parser.parse_args()

    main(args)
