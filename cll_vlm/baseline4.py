import os
import sys
import argparse
import random
from typing import Final
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader


def generate_random_candidates(class_list, n_random=5, seed=None):
    """
    Generate random candidate labels for an image.
    Randomly selects n_random labels from class_list (may or may not include true label).
    
    Args:
        class_list: List of all class names
        n_random: Number of random labels to generate
        seed: Random seed (optional)
    
    Returns:
        list: List of n_random randomly selected label names
    """
    if seed is not None:
        random.seed(seed)
    
    # Randomly sample n_random labels from all classes
    candidate_labels = random.sample(class_list, min(n_random, len(class_list)))
    
    return candidate_labels


class RandomCandidateDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset that generates random candidate labels.
    Returns: (image, true_label_idx, candidate_labels, index)
    """
    def __init__(self, base_dataset, class_list, n_random=5, seed=42):
        self.base_dataset = base_dataset
        self.class_list = class_list
        self.n_random = n_random
        self.seed = seed
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, true_label_idx = self.base_dataset[idx]
        
        # Generate random candidates with seed based on index for reproducibility
        candidate_labels = generate_random_candidates(
            self.class_list, 
            self.n_random,
            seed=self.seed + idx  # Different seed per image
        )
        
        return img, true_label_idx, candidate_labels, idx


def collate_fn(batch):
    """Custom collate function for batching"""
    images, true_labels, candidate_labels_list, indices = zip(*batch)
    return list(images), list(true_labels), list(candidate_labels_list), list(indices)


def main(args):
    """Main execution function"""
    from models.llava_classifier import LLaVAClassifier
    from models.qwen_classifier import QWENClassifier
    from dataset.cifar10 import CIFAR10Dataset
    from dataset.cifar20 import CIFAR20Dataset, CIFAR100Dataset
    from dataset.tiny200 import Tiny200Dataset
    from dataset.mnist import MNISTDataset
    from dataset.kmnist import KMNISTDataset
    
    torch.cuda.empty_cache()
    
    # Configure GPU device
    if "," in args.gpu:
        device = "cuda:0"
        device_map = "auto"
        print(f"🔹 Using physical GPUs: {args.gpu} (mapped to PyTorch as cuda:0, cuda:1, ...)")
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_map = None
        print(f"🔹 Using single GPU")
    
    print("🔹 Using model:", args.model)
    print("🔹 Using dataset:", args.dataset)
    print("🔹 Number of random candidates:", args.n_random)
    print("🔹 Random seed:", args.seed)
    print("🔹 Prompt:", args.prompt)

    # ========== LOAD DATASET ==========
    if args.dataset == "cifar10":
        root_path = "/tmp2/maitanha/vgu/data/cifar10"
        dataset = CIFAR10Dataset(root=root_path, train=True)
    elif args.dataset == "cifar20":
        root_path = "/tmp2/maitanha/vgu/data/cifar20"
        dataset = CIFAR20Dataset(root=root_path, train=True)
    elif args.dataset == "cifar100":
        root_path = "/tmp2/maitanha/vgu/data/cifar100"
        dataset = CIFAR100Dataset(root=root_path, train=True)
    elif args.dataset == "tiny200":
        root_path = "/tmp2/maitanha/vgu/data/tiny-imagenet-200"
        dataset = Tiny200Dataset(root=root_path, train=True)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples from {args.dataset} dataset")
    
    # Get class list from dataset
    class_list = dataset.classes
    print(f"Dataset has {len(class_list)} classes")

    # ========== CREATE DATALOADER ==========
    wrapped_dataset = RandomCandidateDataset(
        base_dataset=dataset,
        class_list=class_list,
        n_random=args.n_random,
        seed=args.seed
    )
    
    dataloader = DataLoader(
        wrapped_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Created DataLoader with batch_size={args.batch_size}")

    # ========== MODEL PATH MAPPING ==========
    MODEL_PATHS = {
        "llava": "llava-hf/llava-v1.6-mistral-7b-hf",
        "llava_13b": "llava-hf/llava-v1.6-vicuna-13b-hf",
        "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen3_2b": "Qwen/Qwen3-VL-2B-Instruct",
        "qwen3_8b": "Qwen/Qwen3-VL-8B-Instruct",
        "qwen3_30b_a3b": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    }

    # ========== LOAD VLM MODEL ==========
    if args.model in ["llava", "llava_13b"]:
        model_path = args.model_path or MODEL_PATHS[args.model]
        print(f"Loading LLaVA model: {model_path}")
        classifier = LLaVAClassifier(
            model_path=model_path,
            baseprompt=args.prompt,
            device=device,
            device_map=device_map
        )
    elif args.model in ["qwen", "qwen3_2b", "qwen3_8b", "qwen3_30b_a3b"]:
        model_path = args.model_path or MODEL_PATHS[args.model]
        print(f"Loading Qwen model: {model_path}")
        classifier = QWENClassifier(
            model_path=model_path,
            device=device if device_map is None else None
        )
    else:
        raise ValueError(f"Unknown model: {args.model}. Choose from: {list(MODEL_PATHS.keys())}")
    
    # Print GPU memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    # ========== PREPARE OUTPUT FILE ==========
    if not os.path.exists("results/baseline4"):
        os.makedirs("results/baseline4")
    
    output_path = f"results/baseline4/{args.model}_{args.dataset}_nrand{args.n_random}_seed{args.seed}.csv"
    
    # Write CSV header
    with open(output_path, 'w') as f:
        f.write("index,true_label,candidate_labels,answer,ol_flag\n")
    
    print(f"Writing results to {output_path}")

    # ========== RUN BATCH INFERENCE ==========
    for batch_images, batch_true_labels, batch_candidate_labels, batch_indices in tqdm(dataloader, desc=f"Querying {args.model}"):
        
        # Run batch prediction with candidate labels
        predicted_labels = classifier.predict_best_label_batch(
            batch_images,
            batch_candidate_labels,
            baseprompt=args.prompt
        )

        # Write results immediately after each batch
        with open(output_path, 'a') as f:
            for i, pred_label in enumerate(predicted_labels):
                true_label_name = class_list[batch_true_labels[i]]
                candidate_labels_str = str(batch_candidate_labels[i]).replace(',', ';')
                
                # Normalize prediction and set ol_flag based on VLM response
                # ol_flag = 0: VLM says none of the candidates match (returns "None")
                # ol_flag = 1: VLM selects a label from candidates
                # Check if first 4 chars are "none" (case-insensitive) to handle inconsistent VLM outputs
                if pred_label is None or (isinstance(pred_label, str) and pred_label.strip()[:4].lower() == "none"):
                    answer = "None"
                    ol_flag = 0
                else:
                    # Take only the first token to avoid hallucination
                    # Replace line breaks with space, strip punctuation and whitespace, then take first word
                    cleaned = pred_label.replace('\n', ' ').replace('\r', ' ').strip()
                    first_token = cleaned.split()[0] if cleaned else pred_label
                    # Remove trailing punctuation from first token
                    answer = first_token.rstrip('.,!?;:')
                    ol_flag = 1
                
                f.write(f"{batch_indices[i]},{true_label_name},\"{candidate_labels_str}\",{answer},{ol_flag}\n")

    # ========== DISPLAY RESULTS ==========
    print(f"\nSaved classification results to {output_path}")
    results_df = pd.read_csv(output_path)
    print(results_df.head(10))
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"Total samples: {len(results_df)}")
    print(f"OL flag distribution (1=ordinary, 0=complementary):")
    print(results_df['ol_flag'].value_counts())
    
    # Count valid answers vs None
    none_count = (results_df['answer'] == 'None').sum()
    valid_count = len(results_df) - none_count
    print(f"\nAnswer distribution:")
    print(f"Valid label answers: {valid_count}")
    print(f"None answers: {none_count}")


if __name__ == "__main__":
    default_prompt = """
        "You are given an image and a list of candidate labels. "
        "Your task is to identify whether ANY SINGLE label in the list "
        "clearly and confidently matches the main subject of the image.\n\n"
        f"Candidates: [{labels}]\n\n"
        "Rules:\n"
        "- If you are CONFIDENT that exactly one label matches, return that label.\n"
        "- If NONE of the candidates fits confidently, return one word only: "None".\n"
        "- Do NOT guess. Only answer if you are sure.\n"
        "Return format: either a single label from the candidates or a word "None". Don't include any explanation or punctuation or extra characters."
    """

    # default_prompt = """
    #     "Examine the image against these categories: [{labels}].
    #     "Decision Process:"
    #     "If the image does not clearly and accurately represent one of the categories above, your answer must be 'None'."
    #     " If a precise match exists, provide that category name."
    #     "Constraint: Output exactly one word. No punctuation, no explanation."
    # """

    # default_prompt = """
    #     "Your task is to describe the image"
    # """
    # Parse arguments BEFORE importing torch
    parser = argparse.ArgumentParser(description="Baseline 4 - Random Label Candidate Selection with VLM")
    parser.add_argument("--model", type=str, default="llava",
                        choices=["llava", "llava_13b", "qwen", "qwen3_2b", "qwen3_8b", "qwen3_30b_a3b"],
                        help="Choose VLM model variant")
    parser.add_argument("--model_path", type=str, default=None, 
                        help="Custom model path (optional, uses defaults if not specified)")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                        help="Choose a dataset (e.g., cifar10, cifar20, cifar100, tiny200)")
    parser.add_argument("--prompt", type=str, 
                        default=default_prompt,
                        help="Custom prompt for VLM classification")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size for inference")
    parser.add_argument("--gpu", type=str, default="0", 
                        help="GPU to use (e.g., '0' for single GPU, '4,5' for multi-GPU)")
    parser.add_argument("--n_random", type=int, default=5, 
                        help="Number of random labels to present as candidates")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set CUDA_VISIBLE_DEVICES BEFORE importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"🔹 Set CUDA_VISIBLE_DEVICES={args.gpu} (before importing torch)")
    
    # Execute main function
    main(args)
