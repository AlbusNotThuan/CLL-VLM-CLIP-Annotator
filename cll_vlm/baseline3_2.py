"""
Baseline 3.2: VLM → CLIP two-stage pipeline.

Stage 1 (VLM) : Generate or load an existing multi_label result JSON produced
                by main4.py. Each entry's 'candidate_answer' holds the labels
                the VLM selected across all label-batch rounds.
                If the file is missing or incomplete, this script will automatically
                invoke main4.py to run/complete the VLM phase.
Stage 2 (CLIP) : For each image, use CLIP to pick the SINGLE best label from
                the candidate_answer list.
                Special case: if candidate_answer is EMPTY (VLM picked nothing),
                CLIP is still used to pick top-1 from the full label pool
                (the output field 'candidate_answer' still shows [] to record
                that VLM found nothing).

Output path:
  results/baseline3/baseline3_2/{dataset}/
  baseline3_2_{vlm_model}_{clip_model}_{dataset}_lbs{N}[_{custom}].json
"""

import torch
import os
import sys
import json
import yaml
import numpy as np
from argparse import ArgumentParser, Namespace
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # cll_vlm/
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from dataset.cifar10 import CIFAR10Dataset
from dataset.cifar20 import CIFAR20Dataset, CIFAR100Dataset
from dataset.tiny200 import Tiny200Dataset
from models.clip_model import CLIPModel

# Import main4 so we can invoke Stage 1 automatically
import main4

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


def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)


def build_output_path(base_dir: str, vlm_model: str, clip_key: str,
                      data_name: str, lbs: int, custom: str, test: bool) -> str:
    name_parts = [
        "baseline3_2",
        vlm_model,
        clip_key,
        data_name,
        f"lbs{lbs}",
    ]
    if test:
        name_parts.append("test")
    if custom:
        name_parts.append(custom)
    filename = "_".join(name_parts) + ".json"
    out_dir = os.path.join(base_dir, "results", "baseline3", "baseline3_2", data_name)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


def get_main4_input_path(config: dict, args: Namespace, lbs: int) -> str:
    """Predict the exact path main4.py uses to save its output json."""
    lbs_suffix = f"lbs{lbs}"
    if args.test:
        lbs_suffix += "_test"
    output_file_name = f"{args.model_name}_{args.data_name}_multi_label_{lbs_suffix}"
    if args.vlm_custom_name:
        output_file_name += f"_{args.vlm_custom_name}"
    output_file_name += ".json"
    
    return os.path.join(
        config["workspace"],
        "cll_vlm/ol_cll_logs/multi_label/json",
        args.data_name,
        output_file_name,
    )


# ---------------------------------------------------------------------------
# CLIP best-label selection
# ---------------------------------------------------------------------------

@torch.no_grad()
def clip_best(clip_model: CLIPModel, img_feat: torch.Tensor,
              label_texts: list, text_feats: torch.Tensor):
    """
    Pick the single best label for the image from label_texts using cosine sim.
    img_feat must be a pre-computed, pre-normalized (1, D) tensor.
    """
    txt_feat = text_feats / text_feats.norm(dim=-1, keepdim=True)  # (N, D)
    sims = (img_feat @ txt_feat.T).squeeze(0)                      # (N,)
    best_idx = sims.argmax().item()
    return label_texts[best_idx], sims[best_idx].item()


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
    original_dataset, shuffled_dataset = dataset.get_shuffled_labels_dataset(seed=shuffle_seed)

   # (Moved lbs and full_label_pool initialization down so they can be inferred from input_json)

    # -----------------------------------------------------------------------
    # Resolve input_json (VLM stage 1 output) and variables
    # -----------------------------------------------------------------------
    if args.input_json:
        input_json_path = args.input_json
        print(f"[STAGE 1 - VLM] Using provided input JSON (skipping main4.py): {input_json_path}")
        
        # Auto-infer lbs from filename so user doesn't need to provide it
        import re
        m = re.search(r"lbs(\d+)", os.path.basename(input_json_path))
        if m:
            lbs = int(m.group(1))
            print(f"[INFO] Inferred lbs={lbs} from input JSON filename")
        else:
            lbs = args.label_batch_size if args.label_batch_size is not None else len(fine_classes)
            print(f"[WARN] Could not infer lbs from filename; using lbs={lbs}")
            
    else:
        lbs = args.label_batch_size if args.label_batch_size is not None else len(fine_classes)
        input_json_path = get_main4_input_path(config, args, lbs)
        print(f"[STAGE 1 - VLM] Checking VLM input JSON: {input_json_path}")
        
        # We invoke main4.py so that it can generate the file from scratch or resume!
        vlm_args = Namespace(
            data_name=args.data_name,
            model_name=args.model_name,
            prompt_type="multi_label",
            label_batch_size=args.label_batch_size,
            batch_size=args.batch_size,
            topk=args.vlm_topk,  # main4's topk argument
            custom_output_name=args.vlm_custom_name,
            config_path=args.config_path,
            test=args.test,
            label_description_path=None
        )
        
        # Run main4 (handles running from scratch OR resuming seamlessly)
        print(f"[STAGE 1 - VLM] Delegating generation/resume check to main4.py...")
        main4.main(vlm_args)
        print(f"[STAGE 1 - VLM] Phase Complete. File is ready: {input_json_path}")

    # Initialize full_label_pool AFTER lbs is determined
    full_label_pool = fine_classes[:lbs]

    # Load VLM result JSON
    with open(input_json_path, "r") as f:
        vlm_results = json.load(f)
    print(f"[INFO] Loaded {len(vlm_results)} entries from VLM result JSON")

    # Build an img_idx → entry lookup for random access
    vlm_lookup = {entry["img_idx"]: entry for entry in vlm_results}

    # -----------------------------------------------------------------------
    # Output path & resume logic
    # -----------------------------------------------------------------------
    output_path = build_output_path(
        base_dir=SCRIPT_DIR,
        vlm_model=args.model_name,
        clip_key=args.clip_model,
        data_name=args.data_name,
        lbs=lbs,
        custom=args.custom_output_name or "",
        test=args.test,
    )
    print(f"[INFO] Output path: {output_path}")

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
    else:
        single_mode = False
        if resume_count > 0:
            indices = np.arange(resume_count, len(shuffled_dataset))
            shuffled_dataset = shuffled_dataset.get_subset_by_indices(indices)
            original_dataset = original_dataset.get_subset_by_indices(indices)
            print(f"[RESUME] Sliced datasets. New size: {len(shuffled_dataset)}")
        if args.test: # Truncate to first batch output equivalent
            n = min(args.batch_size, len(shuffled_dataset))
            idxs = np.arange(n)
            shuffled_dataset = shuffled_dataset.get_subset_by_indices(idxs)
            original_dataset = original_dataset.get_subset_by_indices(idxs)
            print(f"[DEBUG] --test mode: truncated to first {len(shuffled_dataset)} samples")

    # -----------------------------------------------------------------------
    # Load CLIP
    # -----------------------------------------------------------------------
    clip_model_name = CLIP_MODEL_MAP[args.clip_model]
    print(f"[STAGE 2 - CLIP] Loading CLIP '{clip_model_name}' ...")
    clip_model = CLIPModel(model_name=clip_model_name)

    # Pre-compute text features for the full label pool (used as fallback)
    print(f"[STAGE 2 - CLIP] Pre-computing CLIP text features for full label pool ({len(full_label_pool)} labels)...")
    full_text_prompts = [f"A photo of a {lbl}" for lbl in full_label_pool]
    full_text_feats = clip_model.encode_text(full_text_prompts)  # (N, D)
    full_text_feats_norm = full_text_feats / full_text_feats.norm(dim=-1, keepdim=True)
    print("[STAGE 2 - CLIP] CLIP text features ready.")

    # -----------------------------------------------------------------------
    # Single-sample mode
    # -----------------------------------------------------------------------
    if single_mode:
        img, true_lbl_idx = original_dataset[idx]
        _, shuf_lbl_idx = shuffled_dataset[idx]
        true_label = fine_classes[true_lbl_idx]
        shuffled_label = fine_classes[shuf_lbl_idx]

        vlm_entry = vlm_lookup.get(idx)
        if vlm_entry is None:
            print(f"[WARN] img_idx={idx} not found in VLM JSON (even after main4 generation). Skipping.")
            return

        cands = vlm_entry.get("candidate_answer") or vlm_entry.get("answer") or []
        if not isinstance(cands, list):
            cands = [cands] if cands and str(cands).upper() != "NO" else []

        vlm_answer = vlm_entry.get("answer", [])
        use_fallback = len(cands) == 0

        img_feat = clip_model.encode_image(img)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        if use_fallback:
            sims = (img_feat @ full_text_feats_norm.T).squeeze(0)
            best_idx = sims.argmax().item()
            clip_answer = full_label_pool[best_idx]
            clip_score = sims[best_idx].item()
            print(f"[CLIP] img_idx={idx} | candidate_answer=[] (VLM empty) | fallback clip_answer={clip_answer}")
        else:
            cand_prompts = [f"A photo of a {lbl}" for lbl in cands]
            cand_feats = clip_model.encode_text(cand_prompts)
            cand_best, clip_score = clip_best(clip_model, img_feat, cands, cand_feats)
            clip_answer = cand_best
            print(f"[CLIP] img_idx={idx} | candidates={cands} | clip_answer={clip_answer}")

        result = {
            "img_idx": idx,
            "true_label": true_label,
            "shuffled_label": shuffled_label,
            "candidate_answer": cands if not use_fallback else [],
            "clip_answer": clip_answer,
            "vlm_answer": vlm_answer,
            "fallback_used": use_fallback,
        }
        single_out_path = output_path.replace(".json", f"_imgidx{idx}.json")
        os.makedirs(os.path.dirname(single_out_path), exist_ok=True)
        with open(single_out_path, "w") as f:
            json.dump([result], f, indent=2)
        print(f"[INFO] Saved single-sample result to: {single_out_path}")
        return

    # -----------------------------------------------------------------------
    # Batch mode
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
        for images, shuffled_labels in tqdm(dataloader, desc="Baseline 3.2 Stage 2 (CLIP)"):
            batch_len = len(images)
            true_labels_idx = [
                original_dataset[i][1]
                for i in range(local_pos, local_pos + batch_len)
            ]
            true_labels = [fine_classes[i] for i in true_labels_idx]
            shuffled_labels_text = [fine_classes[i] for i in shuffled_labels]

            # Pre-compute and normalize all image features for the batch in one forward pass
            img_feats = clip_model.encode_image(list(images))
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            for i in range(batch_len):
                img_idx = dataset_pos + i
                img_feat = img_feats[i].unsqueeze(0)  # (1, D)
                true_label = true_labels[i]
                shuffled_label = shuffled_labels_text[i]

                vlm_entry = vlm_lookup.get(img_idx)
                if vlm_entry is None:
                    print(f"[WARN] img_idx={img_idx} not found in VLM JSON. Skipping.")
                    continue

                cands = vlm_entry.get("candidate_answer") or vlm_entry.get("answer") or []
                if not isinstance(cands, list):
                    cands = [cands] if cands and str(cands).upper() != "NO" else []

                vlm_answer = vlm_entry.get("answer", [])
                use_fallback = len(cands) == 0

                if use_fallback:
                    sims = (img_feat @ full_text_feats_norm.T).squeeze(0)
                    best_idx_clip = sims.argmax().item()
                    clip_answer = full_label_pool[best_idx_clip]
                else:
                    cand_prompts = [f"A photo of a {lbl}" for lbl in cands]
                    cand_feats = clip_model.encode_text(cand_prompts)
                    clip_answer, clip_score = clip_best(clip_model, img_feat, cands, cand_feats)

                result = {
                    "img_idx": img_idx,
                    "true_label": true_label,
                    "shuffled_label": shuffled_label,
                    "candidate_answer": cands if not use_fallback else [],
                    "clip_answer": clip_answer,
                    "vlm_answer": vlm_answer,
                    "fallback_used": use_fallback,
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
    parser = ArgumentParser(description="Baseline 3.2: VLM → CLIP two-stage classification")

    parser.add_argument("--data_name", type=str, required=True,
                        choices=["cifar10", "cifar20", "cifar100", "tiny200"])
    
    parser.add_argument("--model_name", type=str, required=True,
                        choices=VLM_MODEL_KEYS,
                        help="VLM model name (e.g. qwen3_2b) to invoke main4.py for stage 1")
    
    parser.add_argument("--clip_model", type=str, default="vitb32",
                        choices=list(CLIP_MODEL_MAP.keys()),
                        help="CLIP model key: vitb32 or vitl14 (default: vitb32)")
    
    parser.add_argument("--label_batch_size", type=int, default=None,
                        help="Number of labels passed to VLM strictly in Phase 1 (determines multi-pass strategy in main4.py)")
    
    parser.add_argument("--vlm_topk", type=int, default=1,
                        help="TopK setting passed specifically to VLM in main4 Phase 1 (default: 1)")

    parser.add_argument("--input_json", type=str, default=None,
                        help="[Optional] Manually provide the VLM JSON file; otherwise it is dynamically resolved from main4's logs.")
    
    parser.add_argument("--vlm_custom_name", type=str, default=None,
                        help="Custom suffix string used in the VLM's JSON filename (e.g. prompt_v1)")
    
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Number of samples per DataLoader batch (default: 64)")
    parser.add_argument("--custom_output_name", type=str, default=None,
                        help="Optional custom suffix for output filename")
    parser.add_argument("--config_path", type=str,
                        default="/tmp2/maitanha/vgu/cll_vlm/cll_vlm/config/config.yaml")
    parser.add_argument("--test", action="store_true", default=False,
                        help="Truncate to first batch_size entries for debugging")
    parser.add_argument("--img_idx", type=int, default=None,
                        help="Run on a single sample by its dataset index")

    args = parser.parse_args()
    main(args)
