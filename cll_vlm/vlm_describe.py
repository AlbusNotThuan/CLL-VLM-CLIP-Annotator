import torch
import os
from pathlib import Path
import re
import csv
import random
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm
import pdb
from dataset.cifar10 import CIFAR10Dataset
from dataset.cifar20 import CIFAR20Dataset, CIFAR100Dataset

from models.llava_classifier import LLaVAClassifier
from models.clip_model import CLIPModel
from PIL import Image

# from models.qwen_classifier import QWENClassifier

def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)

def load_dataset(data_name):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "cll_vlm/data")

    data_root_path = os.path.join(DATA_DIR, data_name)

    if data_name == "cifar10":
        dataset = CIFAR10Dataset(
            root=data_root_path,
            train=True,
            transform=None
        )
    elif data_name == "cifar20":
        dataset = CIFAR20Dataset(
            root=data_root_path,
            train=True,
            transform=None
        )
    elif data_name == "cifar100":
        dataset = CIFAR100Dataset(
            root = data_root_path,
            train=True,
            transform=None
        )
    else:
        raise ValueError(f"Dataset '{data_name}' chưa được hỗ trợ trong hàm load_dataset.")
    
    return dataset

def normalize_text(s):
    return s.lower().replace("_", " ").strip()

def preprocess_label(label: str) -> str:
    if label.startswith("vehicles_1"):
        return "transportation vehicles"
    if label.startswith("vehicles_2"):
        return "industrial and military vehicles"
    # general case
    return label.replace("_", " ")


def main(args):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = load_dataset(args.data_name)
    orig_dataset, shuffled_dataset = dataset.get_shuffled_labels_dataset(seed=42)

    dataloader = DataLoader(shuffled_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    if args.data_name == "cifar100":
        fine_classes = dataset.get_fine_classes()
        coarse_classes = dataset.get_coarse_classes()
        """
        Load CLIP image-coarse_class similarity matrix
        """
        coarse_sim_path = "/home/maitanha/cll_vlm/cll_vlm/cifar_clip_similarity/cifar100_image_coarse.npy"
        if coarse_sim_path is None:
            raise ValueError("Please provide path to coarse similarity matrix for CIFAR100")
        coarse_sim = np.load(coarse_sim_path)  # (50000, 20)

    # Load model based on model_type
    if args.model_type == "llava":
        model = LLaVAClassifier.build_model(args)
    elif args.model_type == "qwen":
        pass
    elif args.model_types == "blip2":
        pass
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # Load CLIP model
    clip_model = CLIPModel(device=device)

    # =========================
    # Prepare output CSV
    # =========================
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    columns = [
        "index",
        "true_label",
        "random_label",
        "true_coarse",
        "random_coarse",
        "top3_coarse",
        "clip_pos",
        "clip_neg",
        "vlm_caption",
        "predicted"
    ]

    if not os.path.exists(args.output_path):
        df_init = pd.DataFrame({
            "index": pd.Series(dtype="int"),
            "true_label": pd.Series(dtype="string"),
            "random_label": pd.Series(dtype="string"),
            "true_coarse": pd.Series(dtype="string"),
            "random_coarse": pd.Series(dtype="string"),
            "top3_coarse": pd.Series(dtype="string"),
            "clip_pos": pd.Series(dtype="float"),
            "clip_neg": pd.Series(dtype="float"),
            "vlm_caption": pd.Series(dtype="string"),
            "predicted": pd.Series(dtype="string"),
        })
        df_init.to_csv(args.output_path, index=False)

    # =========================
    # Stage 1: coarse filter
    # =========================
    dataset_pos = 0
    topk = 3

    rows = []
    stage2_indices = []
    stage2_labels = []

    for images, shuffled_fine_indices in tqdm(dataloader, desc="Stage 1"):
        batch_len = len(images)

        # True fine and coarse labels
        true_fine_indices = [orig_dataset[i][1] for i in range(dataset_pos, dataset_pos + batch_len)]
        
        for i in range(batch_len):
            idx = dataset_pos + i

            # true label
            true_fine_idx = true_fine_indices[i]
            true_label = fine_classes[true_fine_idx]
            true_coarse_idx = dataset.fine_to_coarse(true_fine_idx)
            true_coarse = coarse_classes[true_coarse_idx]

            # shuffled label
            shuffled_fine_idx = shuffled_fine_indices[i]
            shuffled_label = fine_classes[shuffled_fine_idx]
            shuffled_coarse_idx = dataset.fine_to_coarse(shuffled_fine_idx)
            shuffled_coarse = coarse_classes[shuffled_coarse_idx]

            # CLIP coarse topk
            sim_row = coarse_sim[idx]  # (20,)
            topk_idx = sim_row.argsort()[-topk:][::-1]
            topk_names = [coarse_classes[j] for j in topk_idx]
            
            # decision
            if shuffled_coarse not in topk_names:
                predicted = "CL"
                vlm_caption = ""
                clip_pos = np.nan
                clip_neg = np.nan
            else:
                predicted = ""
                vlm_caption = ""
                clip_pos = np.nan
                clip_neg = np.nan
                stage2_indices.append(idx)
                stage2_labels.append(shuffled_label)
            
            rows.append([
                idx,
                true_label,
                shuffled_label,
                true_coarse,
                shuffled_coarse,
                ",".join(topk_names),
                clip_pos,
                clip_neg,
                vlm_caption,
                predicted
            ])

        dataset_pos += batch_len

    # Save stage 1 results
    df = pd.DataFrame(rows, columns=columns)
    
    df["index"] = df["index"].astype(int)
    df["vlm_caption"] = df["vlm_caption"].astype("string")
    df["predicted"] = df["predicted"].astype("string")

    print(f"[Stage 1] CL = {(df['predicted'] == 'CL').sum()} / {len(df)}")
    print(f"[Stage 1] → Stage 2 = {len(stage2_indices)}")
    print(f"DataFrame dtypes: {df.dtypes}")

    df.to_csv(args.output_path, index=False)

    # =========================
    # Stage 2: CLIP pos/neg + LLaVA YES / NO
    # =========================
    if len(stage2_indices) == 0:
        print("[Stage 2] No samples.")
        return

    df = pd.read_csv(args.output_path)
    
    # ===== FIX dtypes =====
    df["index"] = df["index"].astype(int)
    df["vlm_caption"] = df["vlm_caption"].astype("string")
    
    answers_all = []
    clip_pos_all = []
    clip_neg_all = []

    for start in tqdm(range(0, len(stage2_indices), args.batch_size), desc="Stage 2"):
        end = min(start + args.batch_size, len(stage2_indices))
        
        batch_indices = stage2_indices[start:end]
        batch_labels = stage2_labels[start:end]
        batch_images = [dataset[i][0] for i in batch_indices]

        # CLIP pos / neg
        for img, label in zip(batch_images, batch_labels):
            clean_label = preprocess_label(label)
            pos_prompt = f"This is a photo of a {clean_label}"
            neg_prompt = f"This is not a photo of a {clean_label}"

            img_feat = clip_model.encode_image(img)
            pos_feat = clip_model.encode_text(pos_prompt)
            neg_feat = clip_model.encode_text(neg_prompt)

            clip_pos = clip_model.compute_similarity(img_feat, pos_feat)
            clip_neg = clip_model.compute_similarity(img_feat, neg_feat)

            clip_pos_all.append(clip_pos)
            clip_neg_all.append(clip_neg)
        
        # LLaVA YES / NO
        answers = model.predict(batch_images, batch_labels)
        answers_all.extend(answers)
    
    for idx, ans, cp, cn in zip(stage2_indices, answers_all, clip_pos_all, clip_neg_all):
        if ans == "YES" and cp > cn:
            predicted = "OL"
        else:
            predicted = "UNKNOWN"
        
        df.loc[df["index"] == idx, "vlm_caption"] = ans
        df.loc[df["index"] == idx, "predicted"] = predicted
        df.loc[df["index"] == idx, "clip_pos"] = cp
        df.loc[df["index"] == idx, "clip_neg"] = cn
    
    df.to_csv(args.output_path, index=False)
    print("[Stage 2] CSV updated with captions and OL/UNKNOWN predictions")

    # with open(args.output_path, "a", newline="") as f:
    #     writer = csv.writer(f)
        
    #     if not file_exists:
    #         writer.writerow(["index", "true_label", "random_label", "super_class", "vlm_caption", "predicted"])
    
    #     # ===== Stage 1: Check if shuffle label is in top-k similar classes =====
    #     topk = 3
    #     index = 0
    #     dataset_pos = 0

    #     num_cl = 0
    #     total = 0
    #     for images, shuffled_labels in tqdm(dataloader):
    #         batch_len = len(images)

    #         # ===== true labels from orig_dataset =====
    #         true_label_indices = [orig_dataset[i][1] for i in range(dataset_pos, dataset_pos + batch_len)]
    #         true_labels = [dataset.classes[idx] for idx in true_label_indices]

    #         # ===== random labels (from shuffled_dataset batch) =====
    #         shuffled_label_indices = shuffled_labels
    #         shuffled_labels_name = [dataset.classes[idx] for idx in shuffled_label_indices]

    #         for i in range(batch_len):
    #             img_idx = dataset_pos + i
                
    #             shuffled_fine_idx = shuffled_label_indices[i]
    #             shuffled_coarse_idx = dataset.fine_to_coarse(shuffled_fine_idx)
    #             shuffled_coarse_name = superclasses[shuffled_coarse_idx]

    #             # top-k superclasses based on CLIP similarity
    #             sim_row = clip_sim[img_idx]  # (100,)
    #             topk_coarse_idx = sim_row.argsort()[-topk:][::-1]
    #             topk_coarse_names = [superclasses[j] for j in topk_coarse_idx]

    #             # check condition
    #             in_topk = shuffled_coarse_idx in topk_coarse_idx
    #             predicted = "CL" if in_topk else ""
                
    #             if predicted == "CL":
    #                 num_cl += 1
    #             total += 1
    #             # write to CSV
    #             writer.writerow([
    #                 index,
    #                 true_labels[i],
    #                 shuffled_labels_name[i],
    #                 shuffled_coarse_name,
    #                 "",  # VLM caption empty for stage 1
    #                 predicted
    #             ])

    #             index += 1

    #         dataset_pos += batch_len

    #         print(f"[Stage 1] CL samples: {num_cl}/{total} ({num_cl/total:.2%})")

    #     # Stage 2: VLM description generation
    #     # select samples not predicted as CL
    #     df = pd.read_csv(args.output_path)
    #     df_stage2 = df[df["predicted"].isna() | (df["predicted"] == "")]
        
    #     print(f"[Stage 2] Running LLaVA on {len(df_stage2)} samples")

    #     if len(df_stage2) > 0:
    #         # Process in batches to avoid VRAM issues
    #         stage2_batch_size = args.batch_size
            
    #         # Collect indices and labels first (lightweight)
    #         stage2_indices_list = []
    #         stage2_random_labels_list = []
            
    #         for _, row in df_stage2.iterrows():
    #             stage2_indices_list.append(int(row["index"]))
    #             stage2_random_labels_list.append(row["random_label"])

    #         # Process in batches
    #         decoded_results = []
    #         for batch_start in tqdm(range(0, len(stage2_indices_list), stage2_batch_size), desc="Stage 2 VLM"):
    #             batch_end = min(batch_start + stage2_batch_size, len(stage2_indices_list))
    #             batch_indices = stage2_indices_list[batch_start:batch_end]
    #             batch_random_labels = stage2_random_labels_list[batch_start:batch_end]
                
    #             # Load images for this batch only
    #             batch_images = []
    #             for idx in batch_indices:
    #                 img, _ = dataset[idx]
    #                 batch_images.append(img)

    #             # ===== build prompts =====
    #             batch_prompts = [
    #                 f"[INST]<image>\n{args.prompt}[/INST]"
    #                 for _ in batch_images
    #             ]

    #             # Process batch
    #             inputs = processor(
    #                 images=batch_images,
    #                 text=batch_prompts,
    #                 padding=True,
    #                 return_tensors="pt"
    #             ).to(device)

    #             with torch.no_grad():
    #                 outputs = model.generate(
    #                     **inputs,
    #                     max_new_tokens=64 if not args.chain_of_thought else 512,
    #                     pad_token_id=processor.tokenizer.eos_token_id
    #                 )

    #             decoded = processor.batch_decode(
    #                 outputs,
    #                 skip_special_tokens=True,
    #                 clean_up_tokenization_spaces=False
    #             )
                
    #             decoded_results.extend(decoded)
                
    #             # Clear GPU cache after each batch
    #             del inputs, outputs, batch_images, batch_prompts
    #             if torch.cuda.is_available():
    #                 torch.cuda.empty_cache()

    #         # ===== decision & update CSV =====
    #         for idx, random_label, caption in zip(
    #             stage2_indices_list, stage2_random_labels_list, decoded_results
    #         ):
    #             caption = caption.strip()

    #             if caption_contains_class(caption, random_label):
    #                 predicted = "OL"
    #             else:
    #                 predicted = "UNKNOWN"

    #             df.loc[df["index"] == idx, "vlm_caption"] = caption
    #             df.loc[df["index"] == idx, "predicted"] = predicted

    #         # save back
    #         df.to_csv(args.output_path, index=False)
    #         print("[Stage 2] CSV updated with captions and OL/UNKNOWN predictions")
    #     else:
    #         print("[Stage 2] No samples to process.")

    # # with open(args.output_path, "w", newline="") as f:
    # #     writer = csv.writer(f)
    # #     writer.writerow(["index", "true_label", "random_label", "predicted"])

    # #     index = 0
    # #     dataset_pos = 0

    # #     for images, shuffled_labels in tqdm(dataloader):
    # #         batch_len = len(images)

    # #         # ===== true labels from orig_dataset =====
    # #         true_label_indices = [
    # #             orig_dataset[i][1] for i in range(dataset_pos, dataset_pos + batch_len)
    # #         ]
    # #         true_labels = [dataset.classes[idx] for idx in true_label_indices]

    # #         # ===== random labels (from shuffled_dataset batch) =====
    # #         # shuffled_labels here are indices -> convert to names
    # #         random_labels = [dataset.classes[idx] for idx in shuffled_labels]

    #         # # ===== build prompts (same prompt for whole batch) =====
    #         # # (semantic grounding prompt, no label included)
    #         # prompts = [f"[INST]<image>\n{args.prompt}[/INST]" for _ in images]

    #         # inputs = processor(
    #         #     images=images,
    #         #     text=prompts,
    #         #     padding=True,
    #         #     return_tensors="pt"
    #         # ).to(device)

    #         # outputs = model.generate(
    #         #     **inputs,
    #         #     max_new_tokens=32 if not args.chain_of_thought else 512,
    #         #     pad_token_id=processor.tokenizer.eos_token_id
    #         # )

    #         # decoded = processor.batch_decode(
    #         #     outputs,
    #         #     skip_special_tokens=True,
    #         #     clean_up_tokenization_spaces=False
    #         # )

    # #         # ===== write CSV =====
    # #         for tl, rl, text in zip(true_labels, random_labels, decoded):
    # #             writer.writerow([index, tl, rl, text.strip()])
    # #             index += 1

    # #         dataset_pos += batch_len

    # # print(f"Done. Results saved at {args.output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="OL CLL Data Collection - VLM Description Generation")

    # ===== Dataset & IO =====
    parser.add_argument(
        "--data_name",
        type=str,
        required=True,
        choices=["cifar10", "cifar20", "cifar100"],
        help="Dataset name"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output CSV file"
    )

    # ===== Model =====
    parser.add_argument(
        "--model_type",
        type=str,
        default="llava",
        choices=["llava"],
        help="Model type (currently only llava is supported)"
    )

    # ===== Inference =====
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference (used for both Stage 1 and Stage 2)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt for semantic grounding (no label included)"
    )
    parser.add_argument(
        "--chain_of_thought",
        action="store_true",
        help="Enable chain-of-thought (NOT recommended for LLaVA Mistral)"
    )

    args = parser.parse_args()
    main(args)
