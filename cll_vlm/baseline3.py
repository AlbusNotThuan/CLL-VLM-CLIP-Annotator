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
from models.qwen_classifier import QWENClassifier, extract_multi_label_full
from models.clip_model import CLIPModel
from PIL import Image
from qwen_vl_utils import process_vision_info

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
        fine_classes_raw = dataset.get_fine_classes()
        fine_classes = [
            CIFAR100Dataset.preprocess_label(lbl)
            for lbl in fine_classes_raw
        ]
        coarse_classes = dataset.get_coarse_classes()
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

    # Load dataset
    dataset = load_dataset(data_name)

    round1_result = "/tmp2/maitanha/vgu/cll_vlm/cll_vlm/ol_cll_logs/multi_label/json/cifar100/qwen3_2b_cifar100_multi_label_lbs20.json"
    model_name = "qwen3_2b"
    model_path = config["models"][model_name]["model_url"]
    
    # Initialize model
    model = QWENClassifier(model_path)

    with open(round1_result, "r") as f:
        round1_results = json.load(f)
    
    batch_size = args.batch_size
    
    if args.test:
        round1_results = round1_results[:batch_size]
        print(f"[DEBUG] --test mode: truncated to first {len(round1_results)} samples")
    
    output_dir = "/tmp2/maitanha/vgu/cll_vlm/cll_vlm/ol_cll_logs/multi_label/round2/cifar100"
    os.makedirs(output_dir, exist_ok=True)
    round1_name = round1_result.split("/")[-1].split(".")[0]
    out_file_name = f"{round1_name}_round2"
    if args.test:
        out_file_name += "_test"
    out_file = os.path.join(output_dir, f"{out_file_name}.json")
    
    final_results = []

    # Process batch by batch over the round1_results
    for i in tqdm(range(0, len(round1_results), batch_size), desc="Round 2 Inference"):
        batch_items = round1_results[i:i+batch_size]
        batch_images = []
        batch_messages = []
        
        for item in batch_items:
            img_idx = item["img_idx"]
            candidates = item.get("answer", [])
            
            image, _ = dataset[img_idx]
            
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy().transpose(1, 2, 0)
                if image.max() <= 1.0:
                    image = (image * 255).astype("uint8")
                else:
                    image = image.astype(np.uint8)
                image = Image.fromarray(image).copy()
                
            batch_images.append(image)
            
            if not candidates or candidates == "NO":
                candidates_str = "None"
            elif isinstance(candidates, list):
                candidates_str = ", ".join(candidates)
            else:
                candidates_str = str(candidates)
                
            prompt = (
                "You are given an image. "
                "Examine the image carefully and identify which objects from the candidate list are present.\n"
                f"Candidates: ({candidates_str}).\n"
                "From this list, choose only ONE label that is most likely present in the image. "
                "Do not include any label that is not in the candidate list. "
                "If you think none of the candidates are present, reply with exactly \"NO\".\n"
                "Provide a short reason for your answer.\n"
                
                "Before you make the final response, carefully review if your answer ONLY contains labels in the candidates. "
                "Your answer should be ONLY a JSON dict and nothing else, formatted as: "
                "{\"answer\": [...], \"reason\": \"...\"}\n"
                "Please don't reply in other formats."
            )
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            batch_messages.append(messages)
            
        texts = [
            model.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
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
                max_new_tokens=128,
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
        
        for item, out in zip(batch_items, output_texts):
            print(f"[DEBUG] Raw answer:\n{out}\n{'-'*50}")
            predicted, reason = extract_multi_label_full(out)
            
            # Chuyển đổi thành định dạng y hệt file round 1
            new_item = {
                "img_idx": item.get("img_idx"),
                "true_label": item.get("true_label"),
                "shuffled_label": item.get("shuffled_label"),
                "answer": predicted,  # predicted vốn dĩ đã là list
                "reason": [reason] if reason else []  # Đóng gói reason vào list
            }
            final_results.append(new_item)
            
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"Finished Round 2. Results saved to {out_file}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="/tmp2/maitanha/vgu/cll_vlm/cll_vlm/config/config.yaml")
    parser.add_argument("--data_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test", action="store_true", default=False, help="Test mode: truncate to first batch")
    args = parser.parse_args()
    main(args)
