import torch
import os
from pathlib import Path
import re
import csv
import random
import numpy as np
import pandas as pd
import yaml
import json
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm
import pdb
from dataset.cifar10 import CIFAR10Dataset
from dataset.cifar20 import CIFAR20Dataset, CIFAR100Dataset
from dataset.tiny200 import Tiny200Dataset
from dataset.caltech101 import Caltech101Dataset

from models.llava_classifier import LLaVAClassifier
from models.qwen_classifier import QWENClassifier
# from models.qwen_classifier_conv import QWENClassifier
from models.clip_model import CLIPModel
from PIL import Image

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

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
    elif data_name == "tiny200":
        dataset = Tiny200Dataset(
            root=data_root_path,
            train=True,
            transform=None
        )
    else:
        raise ValueError(f"Dataset '{data_name}' chưa được hỗ trợ trong hàm load_dataset.")
    
    return dataset

def main(args):
    # Config
    config = load_config(args.config_path)
    root_dir = config["workspace"]
    data_cfg = config["data"]

    data_name = args.data_name or data_cfg["dataset"]
    data_mode = data_cfg.get("mode", "train")
    data_path = data_cfg["paths"][data_name]
    batch_size = args.batch_size
    num_workers = data_cfg.get("num_workers", 4)
    shuffle_seed = data_cfg.get("shuffle_seed", 42)
    
    output_name = args.model_name + "_" + args.data_name + "_" + args.prompt_type
    # Desire output path
    if args.prompt_type == "binary":
        output_dir = os.path.join(config["output"]["binary"], data_name)
        os.makedirs(output_dir, exist_ok=True)
        if args.custom_output_name is None:
            output_path = os.path.join(output_dir, output_name + ".json")
        else:
            output_path = os.path.join(output_dir, output_name + "_" + args.custom_output_name + ".json")
        print(f"[DEBUG] Output path: {output_path}")
    elif args.prompt_type == "multi_label":
        base_output_dir = os.path.join(config["workspace"], "cll_vlm/ol_cll_logs/multi_label/json")
        output_dir = os.path.join(base_output_dir, data_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = None  # will be set after fine_classes is loaded (need num labels for default lbs)

    # =========================
    # Build label_batches for multi_label (needs fine_classes, loaded below)
    # =========================
    label_batches = None  # will be populated after dataset loading

    # =========================
    # Load dataset
    # =========================
    if data_name == "cifar10":
        dataset = CIFAR10Dataset(
            root=data_path,
            train=(data_mode=="train"),
            transform=None,
        )
        fine_classes = list(dataset.classes)
    elif data_name == "cifar20":
        dataset = CIFAR20Dataset(
            root=data_path,
            train=(data_mode=="train"),
            transform=None,
        )
        fine_classes_raw = list(dataset.classes)  # 20 coarse classes
        fine_classes = [
            CIFAR20Dataset.preprocess_label(lbl)
            for lbl in fine_classes_raw
        ]
    elif data_name == "cifar100":
        dataset = CIFAR100Dataset(
            root=data_path,
            train=(data_mode=="train"),
            transform=None,
        )
        fine_classes_raw = dataset.get_fine_classes()
        fine_classes = [
            dataset.preprocess_label(lbl)
            for lbl in fine_classes_raw
        ]
        coarse_classes = dataset.get_coarse_classes()
    elif data_name == "tiny200":
        dataset = Tiny200Dataset(
            root=data_path,
            train=(data_mode=="train"),
            transform=None,
        )
        fine_classes = list(dataset.classes)
    elif data_name == "caltech101":
        dataset = Caltech101Dataset(
            root=data_path,
            train=(data_mode=="train"),
            transform=None,
        )
        fine_classes_raw = dataset.classes
        fine_classes = [
            dataset.preprocess_label(lbl)
            for lbl in fine_classes_raw
        ]
    else:
        raise ValueError(f"Dataset '{data_name}' chưa được hỗ trợ trong hàm load_dataset.")

    # =========================
    # Build label_batches for multi_label (in order, no shuffle)
    # =========================
    if args.prompt_type == "multi_label":
        label_batch_size = args.label_batch_size if args.label_batch_size is not None else len(fine_classes)
        label_batches = [
            fine_classes[i:i + label_batch_size]
            for i in range(0, len(fine_classes), label_batch_size)
        ]
        # Finalise output path now that we know label_batch_size
        lbs_suffix = f"lbs{label_batch_size}"
        if args.test:
            lbs_suffix += "_test"
        output_file_name = f"{args.model_name}_{data_name}_multi_label_{lbs_suffix}"
        if args.custom_output_name:
            output_file_name += f"_{args.custom_output_name}"
        output_file_name += ".json"
        
        output_path = os.path.join(
            config["workspace"],
            "cll_vlm/ol_cll_logs/multi_label/json",
            data_name,
            output_file_name,
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"[DEBUG] multi_label output path: {output_path}")
        if args.test:
            # Only generate raw_answers file in test mode
            raw_output_path = output_path.replace(".json", "_raw_answers.json")
            print(f"[DEBUG] raw_answers output path: {raw_output_path}")
        print(f"[DEBUG] label_batch_size={label_batch_size}, num_batches={len(label_batches)}")

    # =========================
    # Get shuffled dataset
    # =========================
    original_dataset, shuffled_dataset = dataset.get_shuffled_labels_dataset(
        seed=shuffle_seed
    )

    # [RESUME LOGIC]
    resume_count = 0
    if output_path and os.path.exists(output_path):
        print(f"[RESUME] Found existing file: {output_path}")
        with open(output_path, 'r') as f:
            content = f.read().strip()
        if content:
            valid_existing_data = []
            if content.endswith(']'):
                try:
                    valid_existing_data = json.loads(content)
                except Exception:
                    pass
            
            if not valid_existing_data:
                # Interrupted write, find last valid complete object
                while content:
                    last_brace = content.rfind('}')
                    if last_brace == -1: break
                    content = content[:last_brace+1]
                    try:
                        valid_existing_data = json.loads(content + '\n]')
                        break
                    except json.JSONDecodeError:
                        content = content[:-1]
            
            if valid_existing_data:
                resume_count = len(valid_existing_data)
                print(f"[RESUME] Resuming from index {resume_count}. Rewriting clean file.")
                with open(output_path, 'w') as f:
                    f.write('[\n')
                    for i, item in enumerate(valid_existing_data):
                        if i > 0: f.write(',\n')
                        json.dump(item, f, indent=2)
                    f.flush()

    if resume_count > 0:
        indices = np.arange(resume_count, len(shuffled_dataset))
        shuffled_dataset = shuffled_dataset.get_subset_by_indices(indices)
        original_dataset = original_dataset.get_subset_by_indices(indices)
        print(f"[RESUME] Sliced datasets. New size: {len(shuffled_dataset)}")

    # [DEBUGGING]: Truncate to first batch only when --test is set
    if args.test:
        indices = np.arange(min(batch_size, len(shuffled_dataset)))
        shuffled_dataset = shuffled_dataset.get_subset_by_indices(indices)
        original_dataset = original_dataset.get_subset_by_indices(indices)
        print(f"[DEBUG] --test mode: truncated to first {len(shuffled_dataset)} samples")


    dataloader = DataLoader(
        shuffled_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    # =========================
    # Load model
    # =========================
    if args.model_name in ["llava", "llava_13b"]:
        model_path = config["models"][args.model_name]["model_url"]
        model = LLaVAClassifier(model_path=model_path)
    elif args.model_name in ["qwen", "qwen3_2b", "qwen3_4b", "qwen3_8b", "qwen3_30b_a3b"]:
        model_path = config["models"][args.model_name]["model_url"]
        model = QWENClassifier(model_path=model_path)
    else:
        raise ValueError(f"Unsupported model '{args.model_name}'.")
    
    # =========================
    # Run inference batch by batch with incremental saving
    # =========================
    dataset_pos = resume_count
    local_dataset_pos = 0
    total_results = resume_count
    is_first_batch = (resume_count == 0)

    # Ensure output directory exists and open file for incremental writing
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # If we resumed, open in 'a' mode, array already started
        if resume_count > 0:
            output_file = open(output_path, "a")
        else:
            output_file = open(output_path, "w")
            output_file.write("[\n")  # Start JSON array
    else:
        output_file = None

    try:
        for images, shuffled_labels in tqdm(dataloader, desc="Running Inference"):
            batch_len = len(images)

            # true labels must come from original dataset
            true_batch = [
                original_dataset[i][1]
                for i in range(local_dataset_pos, local_dataset_pos + batch_len)
            ]

            # Call model for this specific batch
            extra_kwargs = {}
            if args.prompt_type == "multi_label":
                extra_kwargs["label_batches"] = label_batches
                extra_kwargs["topk"] = args.topk
                if args.test and 'raw_output_path' in locals():
                    extra_kwargs["raw_output_path"] = raw_output_path

            batch_results = model.generate_batch_results(
                data=images,
                shuffled_label_indices=shuffled_labels, # already indices from dataloader
                true_label_indices=true_batch,
                fine_classes=fine_classes,
                prompt_type=args.prompt_type,
                output_path=None, # We save incrementally, not at the end
                batch_size=batch_len, # Process the whole dataloader batch
                start_idx=dataset_pos,
                label_description_path=args.label_description_path or config.get("label_description_json"),
                **extra_kwargs,
            )
            
            # Write batch results immediately to file
            if output_file:
                for result in batch_results:
                    if not is_first_batch:
                        output_file.write(",\n")
                    else:
                        is_first_batch = False
                    json.dump(result, output_file, indent=2)
                output_file.flush()  # Ensure data is written to disk
            
            total_results += len(batch_results)
            local_dataset_pos += batch_len
            dataset_pos += batch_len

    finally:
        # Close JSON array and file
        if output_file:
            output_file.write("\n]")
            output_file.close()

if __name__ == "__main__":
    parser = ArgumentParser(description="OL CLL Data Collection")

    # Dataset & IO
    parser.add_argument(
        "--data_name",
        type=str,
        required=True,
        default="cifar100",
        choices=["cifar10", "cifar20", "cifar100", "tiny200", "caltech101"],
        help="Name of the dataset to use (e.g., cifar10, cifar20, cifar100, tiny200, caltech101).",
    )
    parser.add_argument(
        "--custom_output_name",
        type=str,
        required=False,
        help="Custom name for the output json.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/tmp2/maitanha/vgu/cll_vlm/cll_vlm/config/config.yaml",
        help="Path to the configuration YAML file.",
    )

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["llava", "llava_13b", "qwen", "qwen3_2b", "qwen3_4b", "qwen3_8b", "qwen3_30b_a3b"],
        help="Type of the model to use (e.g., llava, qwen, janus).",
    )

    #Inference
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        required=True,
        choices=["binary", "label_description", "multi_label"],
        help="Type of prompt to use (binary, label_description, or multi_label).",
    )
    parser.add_argument(
        "--label_batch_size",
        type=int,
        default=None,
        help="Number of labels per batch for multi_label prompt type. Defaults to all labels if not set.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=1,
        help="Max number of labels the VLM can select per label batch (multi_label only). Default: 1.",
    )
    parser.add_argument(
        "--label_description_path",
        type=str,
        default=None,
        help="Path to label description JSON (required for prompt_type=label_description). Overrides config.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Test mode: truncate to first batch and save raw VLM answers for inspection.",
    )

    args = parser.parse_args()
    main(args)