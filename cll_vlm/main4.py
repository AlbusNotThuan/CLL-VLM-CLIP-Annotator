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

from models.llava_classifier import LLaVAClassifier
from models.qwen_classifier import QWENClassifier
from models.janus_classifier import JanusClassifier
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

def normalize_text(s):
    return s.lower().replace("_", " ").strip()

# def preprocess_label(label: str) -> str:
#     if label.startswith("vehicles_1"):
#         return "transportation vehicles"
#     if label.startswith("vehicles_2"):
#         return "industrial and military vehicles"
#     # general case
#     return label.replace("_", " ")

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
    elif args.prompt_type == "label_description":
        base_output_dir = config["output"].get("label_description", os.path.join(config["workspace"], "cll_vlm/ol_cll_logs/label_description"))
        output_dir = os.path.join(base_output_dir, data_name)
        os.makedirs(output_dir, exist_ok=True)
        if args.custom_output_name is None:
            output_path = os.path.join(output_dir, output_name + ".json")
        else:
            output_path = os.path.join(output_dir, output_name + "_" + args.custom_output_name + ".json")
        print(f"[DEBUG] Output path: {output_path}")
    elif args.prompt_type == "multiple":
        output_path = None  # set when implemented

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
            CIFAR100Dataset.preprocess_label(lbl)
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
    else:
        raise ValueError(f"Dataset '{data_name}' chưa được hỗ trợ trong hàm load_dataset.")

    # =========================
    # Get shuffled dataset
    # =========================
    original_dataset, shuffled_dataset = dataset.get_shuffled_labels_dataset(
        seed=shuffle_seed
    )

    # [DEBUGGING]: Truncate to first batch only with diverse classes
    if args.custom_output_name == "test":
        # Create shuffled indices to get diverse classes (original data is sorted by class)
        indices = np.arange(len(shuffled_dataset))
        np.random.seed(shuffle_seed)
        np.random.shuffle(indices)
        indices = indices[:batch_size]
        
        # Use get_subset_by_indices to handle both data and targets correctly
        shuffled_dataset = shuffled_dataset.get_subset_by_indices(indices)
        original_dataset = original_dataset.get_subset_by_indices(indices)

        print(f"[DEBUG] Truncated to {len(shuffled_dataset)} random samples for testing")


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
    elif args.model_name in ["qwen", "qwen3_2b", "qwen3_8b", "qwen3_30b_a3b"]:
        model_path = config["models"][args.model_name]["model_url"]
        model = QWENClassifier(model_path=model_path)
    elif args.model_name == "janus":
        model_path = config["models"][args.model_name]["model_url"]
        model = JanusClassifier(model_path=model_path)
    else:
        raise ValueError(f"Unsupported model '{args.model_name}'.")
    
    # =========================
    # Run inference batch by batch
    # =========================
    results = []
    dataset_pos = 0

    # Ensure output directory exists early
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for images, shuffled_labels in tqdm(dataloader, desc="Running Inference"):
        batch_len = len(images)

        # true labels must come from original dataset
        true_batch = [
            original_dataset[i][1]
            for i in range(dataset_pos, dataset_pos + batch_len)
        ]

        # Call model for this specific batch
        batch_results = model.generate_batch_results(
            data=images,
            shuffled_label_indices=shuffled_labels, # already indices from dataloader
            true_label_indices=true_batch,
            fine_classes=fine_classes,
            prompt_type=args.prompt_type,
            output_path=None, # We'll save all at once at the end
            batch_size=batch_len, # Process the whole dataloader batch
            start_idx=dataset_pos,
            label_description_path=args.label_description_path or config.get("label_description_json"),
        )
        
        results.extend(batch_results)
        dataset_pos += batch_len




    # =========================
    # Save results (JSON)
    # =========================
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Done] Saved {len(results)} records → {output_path}")
    else:
        print(f"[Done] No output path set; {len(results)} records in memory.")

if __name__ == "__main__":
    parser = ArgumentParser(description="OL CLL Data Collection")

    # Dataset & IO
    parser.add_argument(
        "--data_name",
        type=str,
        required=True,
        default="cifar100",
        choices=["cifar10", "cifar20", "cifar100", "tiny200"],
        help="Name of the dataset to use (e.g., cifar10, cifar20, cifar100, tiny200).",
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
        choices=["llava", "llava_13b", "qwen", "qwen3_2b", "qwen3_8b", "janus", "qwen3_30b_a3b"],
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
        choices=["binary", "multiple", "label_description"],
        help="Type of prompt to use (binary, multiple, or label_description).",
    )
    parser.add_argument(
        "--label_description_path",
        type=str,
        default=None,
        help="Path to label description JSON (required for prompt_type=label_description). Overrides config.",
    )

    args = parser.parse_args()
    main(args)