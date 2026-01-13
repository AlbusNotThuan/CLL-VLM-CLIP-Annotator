import os
import argparse
import pdb
import torch
import pandas as pd
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.llava_classifier import LLaVAClassifier
from dataset.cifar10 import CIFAR10Dataset
from dataset.cifar20 import CIFAR20Dataset, CIFAR100Dataset
from dataset.tiny200 import Tiny200Dataset
from dataset.mnist import MNISTDataset
from dataset.kmnist import KMNISTDataset


def generate_random_label_lookup(class_list, n_random=4, seed=42, noise=True):
    """
    Generate a lookup table mapping each class index to random candidate labels.
    Including the true label from candidates.
    
    Args:
        class_list: List of all class names in the dataset
        n_random: Number of random labels to select for each class
        seed: Random seed for reproducibility
        noise: Whether the genration includes noise
    
    Returns:
        Dictionary mapping class index to list of candidate label names
    """
    random.seed(seed)
    num_classes = len(class_list)
    
    lookup = {}
    for k in range(num_classes):
        if noise:
            # Randomly select n_random labels with noise
            lookup[k] = random.sample(class_list, min(n_random, num_classes))
        else:
            # Generate candidate labels without noise (true label + random others)
            true_label = class_list[k]
            other_labels = [label for label in class_list if label != true_label]
            selected_labels = random.sample(other_labels, min(n_random - 1, len(other_labels)))
            lookup[k] = selected_labels
        
    
    return lookup


def _true_to_comp(class_list, n_random=4, seed=42, noise=True):
    """
    Return the complementary label lookup table for all classes using random selection.
    
    Args:
        class_list: List of all class names in the dataset
        n_random: Number of random labels to select for each class
        seed: Random seed for reproducibility
        noise: Whether the genration includes noise
    
    Returns:
        Dictionary mapping class index to list of candidate label names
    """
    return generate_random_label_lookup(class_list, n_random, seed, noise)


class ComplementaryLabelDataset(torch.utils.data.Dataset):
    """Wrapper dataset that returns (image, true_label, candidate_labels, index)"""
    def __init__(self, base_dataset, class_list, n_random=4, seed=42, noise=True):
        """
        Args:
            base_dataset: The underlying dataset
            class_list: List of all class names in the dataset
            n_random: Number of random labels to select for each class
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        self.class_list = class_list
        
        # Create lookup table once during initialization
        self.label_lookup = _true_to_comp(self.class_list, n_random, seed, noise)
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, true_label = self.base_dataset[idx]
        
        # Get candidate labels from lookup table
        candidate_labels = self.label_lookup[true_label]
        
        # Shuffle candidate labels to randomize their order
        perm = torch.randperm(len(candidate_labels)).tolist()
        candidate_labels = [candidate_labels[i] for i in perm]
        
        return img, true_label, candidate_labels, idx


def collate_fn(batch):
    """Custom collate function to handle variable-length candidate labels"""
    images, true_labels, candidate_labels_list, indices = zip(*batch)
    return list(images), list(true_labels), list(candidate_labels_list), list(indices)


def main():
    # ========== ARGUMENTS ==========
    parser = argparse.ArgumentParser(description="Stage 2 - LLaVA Complementary Label Selector (batch mode)")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Choose a Dataset to run (e.g., cifar10)")
    parser.add_argument("--prompt", type=str, default="Which label does not belong to this image? Answer the question with a single word from [{labels}]",
                        help="Custom prompt for LLaVA, e.g. '<image> Which label does not belong to this image? Answer the question with a single word from [{labels}].")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for LLaVA inference")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_random", type=int, default=4, help="Number of random labels to select for each class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--noise", action='store_true', help="Whether to add noise to candidate labels")
    args = parser.parse_args()

    torch.cuda.empty_cache()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Using device: {device}")
    print("🔹 Using n_random:", args.n_random)
    print("🔹 Using seed:", args.seed)
    print("🔹 Using dataset:", args.dataset)
    print("🔹 Using noise:", args.noise)
    print("🔹 Prompt:", args.prompt)

    # ========== LOAD DATA ==========
    if args.dataset == "cifar10":
        root_path = "/home/maitanha/cll_vlm/cll_vlm/data/cifar10"
        dataset = CIFAR10Dataset(root=root_path, train=True)
    elif args.dataset == "cifar20":
        root_path = "/home/maitanha/cll_vlm/cll_vlm/data/cifar20"
        dataset = CIFAR20Dataset(root=root_path, train=True)
    elif args.dataset == "cifar100":
        root_path = "/home/maitanha/cll_vlm/cll_vlm/data/cifar100"
        dataset = CIFAR100Dataset(root=root_path, train=True)
    elif args.dataset == "tiny200":
        root_path = "/home/maitanha/data/tiny/tiny-imagenet-200"
        dataset = Tiny200Dataset(root=root_path, train=True)
    elif args.dataset == "mnist":
        root_path = "/home/maitanha/cll_vlm/cll_vlm/data/mnist"
        dataset = MNISTDataset(root=root_path, train=True, download=True)
    elif args.dataset == "kmnist":
        root_path = "/home/maitanha/cll_vlm/cll_vlm/data/kmnist"
        dataset = KMNISTDataset(root=root_path, train=True, download=True)
    # elif args.dataset == "fmnist":
    #     root_path = "/home/maitanha/cll_vlm/cll_vlm/data/fashion-mnist"
    #     dataset = FashionMNISTDataset(root=root_path, train=True, download=True)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples from {args.dataset} dataset")
    
    # Get class list from dataset
    class_list = dataset.classes
    print(f"Dataset has {len(class_list)} classes")

    # ========== CREATE DATALOADER ==========
    # Wrap the dataset with ComplementaryLabelDataset to add candidate labels
    wrapped_dataset = ComplementaryLabelDataset(
        base_dataset=dataset,
        class_list=class_list,
        n_random=args.n_random,
        seed=args.seed,
        noise=args.noise
    )
    
    # Create DataLoader with custom collate function
    dataloader = DataLoader(
        wrapped_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Keep original order for consistent indexing
        num_workers=4,  # Parallel data loading (adjust based on your CPU)
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False  # Speed up data transfer to GPU
    )
    
    print(f"Created DataLoader with batch_size={args.batch_size}, num_workers=4")

    # ========== LOAD LLaVA ==========
    print("Loading LLaVA model...")
    llava = LLaVAClassifier(
        model_path="llava-hf/llava-v1.6-mistral-7b-hf",
        baseprompt=args.prompt,
        device=device
    )

    # ========== PREPARE OUTPUT FILE ==========
    # create if not exists a csv result file
    if not os.path.exists("results"):
        os.makedirs("results")
    
    filename = f"llava_{args.dataset}_nrand={args.n_random}_seed={args.seed}"
    output_path = f"results/{filename}.csv"
    
    # Write CSV header
    with open(output_path, 'w') as f:
        f.write("index,true_label,candidate_labels,complementary_label\n")
    
    print(f"Writing results to {output_path}")

    # ========== RUN BATCH INFERENCE ==========
    for batch_images, batch_true_labels, batch_label_options, batch_data_indices in tqdm(dataloader, desc="Querying LLaVA (batch mode)"):
        
        # # DEBUG 
        # pdb.set_trace()

        # === run batch prediction ===
        best_labels = llava.predict_best_label_batch(
            batch_images,
            batch_label_options,
            baseprompt=args.prompt
        )

        # === collect and write results immediately ===
        with open(output_path, 'a') as f:
            for i, best_label in enumerate(best_labels):
                # Convert candidate_labels list to string representation
                candidate_labels_str = str(batch_label_options[i]).replace(',', ';')
                f.write(f"{batch_data_indices[i]},{batch_true_labels[i]},\"{candidate_labels_str}\",{best_label}\n")

    # ========== DISPLAY RESULTS ==========
    print(f"\nSaved complementary results to {output_path}")
    comp_df = pd.read_csv(output_path)
    print(comp_df.head(10))


if __name__ == "__main__":
    main()
