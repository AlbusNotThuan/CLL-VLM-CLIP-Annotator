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

from models.llava_classifier import LLaVAClassifier
from models.qwen_classifier import QWENClassifier
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
    # Config
    config = load_config(args.config_path)
    root_dir = config["workspace"]
    data_cfg = config["data"]

    data_name = args.data_name or data_cfg["dataset"]
    data_mode = data_cfg.get("mode", "train")
    data_path = data_cfg["paths"][data_name]
    batch_size = args.batch_size or data_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 4)
    shuffle_seed = data_cfg.get("shuffle_seed", 42)
    
    # Desire output path
    if args.prompt_type == "binary":
        output_dir= config["output"]["binary"]
        output_path = os.path.join(output_dir, args.output_name)
    elif args.prompt_type == "multiple":
        pass

    # =========================
    # Load dataset
    # =========================
    if data_name == "cifar10":
        dataset = CIFAR10Dataset(
            root=data_path,
            train=(data_mode=="train"),
            transform=None,
        )
    elif data_name == "cifar20":
        dataset = CIFAR20Dataset(
            root=data_path,
            train=(data_mode=="train"),
            transform=None,
        )
    elif data_name == "cifar100":
        dataset = CIFAR100Dataset(
            root=data_path,
            train=(data_mode=="train"),
            transform=None,
        )
        fine_classes = dataset.get_fine_classes()
        coarse_classes = dataset.get_coarse_classes()
    else:
        raise ValueError(f"Dataset '{data_name}' chưa được hỗ trợ trong hàm load_dataset.")

    original_dataset, shuffled_dataset = dataset.get_shuffled_labels_dataset(
        seed=shuffle_seed
    )

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
    if args.model_name == "llava":
        pass
    elif args.model_name == "qwen":
        model_path = config["models"][args.model_name]["model_url"]
        model = QWENClassifier(model_path=model_path)
    else:
        raise ValueError(f"Unsupported model '{args.model_name}'.")
    
    # =========================
    # Collect data for inference
    # =========================
    images_all = []
    true_label_indices = []
    shuffled_label_indices = []

    dataset_pos = 0

    for images, shuffled_labels in tqdm(dataloader, desc="Collecting samples"):
        batch_len = len(images)

        # true labels must come from original dataset
        true_batch = [
            original_dataset[i][1]
            for i in range(dataset_pos, dataset_pos + batch_len)
        ]

        images_all.extend(images)
        true_label_indices.extend(true_batch)
        shuffled_label_indices.extend(shuffled_labels)

        dataset_pos += batch_len

    print(f"[Data] Total images collected: {len(images_all)}")

    # DEBUGGING: Truncate to first batch only
    images_all = images_all[:batch_size]
    true_label_indices = true_label_indices[:batch_size]
    shuffled_label_indices = shuffled_label_indices[:batch_size]

    print(f"[DEBUG] Truncated to first batch: {len(images_all)} samples")

    # =========================
    # Run inference
    # =========================
    if args.model_name == "qwen":
        results = model.generate_batch_results(
            data=images_all,
            shuffled_label_indices=shuffled_label_indices,
            true_label_indices=true_label_indices,
            fine_classes=fine_classes,
            prompt_type=args.prompt_type,   # "binary"
            output_path=None,               # save manually
            batch_size=batch_size,
            start_idx=0,
        )
    else:
        raise NotImplementedError("Only QWEN binary is wired here.")

    # =========================
    # Save results (JSON)
    # =========================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Done] Saved {len(results)} records → {output_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="OL CLL Data Collection")

    # Dataset & IO
    parser.add_argument(
        "--data_name",
        type=str,
        required=True,
        default="cifar100",
        choices=["cifar10", "cifar20", "cifar100"],
        help="Name of the dataset to use (e.g., cifar10, cifar20, cifar100).",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="Path to save the output results (JSON file).",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/maitanha/cll_vlm/cll_vlm/config/config.yaml",
        help="Path to the configuration YAML file.",
    )

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["llava", "qwen"],
        help="Type of the model to use (e.g., llava, qwen).",
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
        choices=["binary", "multiple"],
        help="Type of prompt to use (binary or multiple).",
    )

    args = parser.parse_args()
    main(args)