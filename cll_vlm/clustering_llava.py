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


def generate_random_label_lookup(class_list, n_random=4, seed=42, n_clusters=None):
    """
    Generate a lookup table mapping each class/cluster index to random candidate labels.
    Including the true label from candidates (noise=True).
    
    Args:
        class_list: List of all class names in the dataset
        n_random: Number of random labels to select for each class/cluster
        seed: Random seed for reproducibility
        n_clusters: If not None, create cluster-based lookup (maps cluster_id -> class_names)
                    If None, create class-based lookup (maps class_idx -> class_names)
    
    Returns:
        Dictionary mapping class/cluster index to list of candidate label names
    """
    random.seed(seed)
    num_classes = len(class_list)
    
    lookup = {}
    if n_clusters is not None:
        # Cluster mode: map each cluster to random class labels
        for cluster_id in range(n_clusters):
            lookup[cluster_id] = random.sample(class_list, min(n_random, num_classes))
    else:
        # Standard mode: map each class to random class labels (with noise)
        for k in range(num_classes):
            lookup[k] = random.sample(class_list, min(n_random, num_classes))
    
    return lookup


class ComplementaryLabelDataset(torch.utils.data.Dataset):
    """Wrapper dataset that returns (image, true_label, cluster_id, candidate_labels, index)"""
    def __init__(self, base_dataset, class_list, n_random=4, seed=42, cluster_labels=None):
        """
        Args:
            base_dataset: The underlying dataset
            class_list: List of all class names in the dataset
            n_random: Number of random labels to select for each class/cluster
            seed: Random seed for reproducibility
            cluster_labels: Optional numpy array of cluster assignments (shape: [n_samples])
                           If provided, uses cluster-based label selection
        """
        self.base_dataset = base_dataset
        self.class_list = class_list
        self.cluster_labels = cluster_labels
        
        # Determine number of clusters if in cluster mode
        n_clusters = None
        if cluster_labels is not None:
            n_clusters = len(set(cluster_labels))
        
        # Create lookup table once during initialization
        self.label_lookup = generate_random_label_lookup(self.class_list, n_random, seed, n_clusters)
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, true_label = self.base_dataset[idx]
        
        # Determine label key based on mode
        if self.cluster_labels is not None:
            # Cluster mode: use cluster_id as key
            label_key = int(self.cluster_labels[idx])
            cluster_id = label_key
        else:
            # Standard mode: use true_label as key
            label_key = true_label
            cluster_id = None
        
        # Get candidate labels from lookup table
        candidate_labels = self.label_lookup[label_key]
        
        # Shuffle candidate labels to randomize their order
        perm = torch.randperm(len(candidate_labels)).tolist()
        candidate_labels = [candidate_labels[i] for i in perm]
        
        return img, true_label, cluster_id, candidate_labels, idx


def collate_fn(batch):
    """Custom collate function to handle variable-length candidate labels"""
    images, true_labels, cluster_ids, candidate_labels_list, indices = zip(*batch)
    return list(images), list(true_labels), list(cluster_ids), list(candidate_labels_list), list(indices)


def main():
    # ========== ARGUMENTS ==========
    parser = argparse.ArgumentParser(description="Stage 2 - LLaVA Complementary Label Selector (batch mode)")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Choose a Dataset to run (e.g., cifar10)")
    parser.add_argument("--prompt", type=str, default="Which label does not belong to this image? Answer the question with a single word from [{labels}]",
                        help="Custom prompt for LLaVA, e.g. '<image> Which label does not belong to this image? Answer the question with a single word from [{labels}].")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for LLaVA inference")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--k_mean", type=int, default=None, help="Number of k-means clusters (None to disable clustering)")
    parser.add_argument("--pretrain_path", type=str, default=None, help="Full path to pretrain checkpoint (required if --k_mean is set)")
    parser.add_argument("--n_random", type=int, default=4, help="Number of random labels to select for each class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true", help="Debug mode: skip LLaVA query, export cluster assignments only")
    args = parser.parse_args()

    torch.cuda.empty_cache()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Using device: {device}")
    print(f"🔹 Mode: {'cluster-based' if args.k_mean else 'standard random'}")
    if args.k_mean:
        print(f"🔹 K-means clusters: {args.k_mean}")
    print("🔹 Using n_random:", args.n_random)
    print("🔹 Using seed:", args.seed)
    print(f"🔹 Debug mode: {args.debug}")
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

    # ========== OPTIONAL CLUSTERING ==========
    cluster_labels = None
    if args.k_mean is not None:
        if args.pretrain_path is None:
            raise ValueError("--pretrain_path must be provided when using --k_mean")
        
        print(f"\n🔹 Loading pretrain model from: {args.pretrain_path}")
        print(f"🔹 Performing k-means clustering with k={args.k_mean}...")
        
        result = dataset.get_feature_clusters(
            pretrain_path=args.pretrain_path,
            n_clusters=args.k_mean,
            random_state=args.seed
        )
        
        cluster_labels = result['cluster_labels']
        cluster_counts = result['cluster_counts']
        
        # Print cluster statistics
        counts_list = list(cluster_counts.values())
        print(f"🔹 Clustered {total_samples} samples into {args.k_mean} clusters")
        print(f"🔹 Cluster distribution: min={min(counts_list)}, max={max(counts_list)}, mean={sum(counts_list)/len(counts_list):.1f}")
        print(f"🔹 Cluster counts: {cluster_counts}")

    # ========== CREATE DATALOADER ==========
    # Wrap the dataset with ComplementaryLabelDataset to add candidate labels
    wrapped_dataset = ComplementaryLabelDataset(
        base_dataset=dataset,
        class_list=class_list,
        n_random=args.n_random,
        seed=args.seed,
        cluster_labels=cluster_labels
    )
    
    # Print lookup table in debug mode or cluster mode
    if args.debug or args.k_mean is not None:
        print(f"\n🔹 Label lookup table ({len(wrapped_dataset.label_lookup)} entries):")
        for key, labels in sorted(wrapped_dataset.label_lookup.items())[:10]:  # Show first 10
            key_name = f"cluster_{key}" if cluster_labels is not None else class_list[key]
            print(f"  {key_name}: {labels}")
        if len(wrapped_dataset.label_lookup) > 10:
            print(f"  ... ({len(wrapped_dataset.label_lookup) - 10} more entries)")
    
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

    # ========== PREPARE OUTPUT FILE ==========
    # create if not exists a csv result file
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Generate filename based on mode
    if args.k_mean is not None:
        filename = f"llava_{args.dataset}_kmean={args.k_mean}_nrand={args.n_random}_seed={args.seed}"
    else:
        filename = f"llava_{args.dataset}_nrand={args.n_random}_seed={args.seed}"
    
    if args.debug:
        filename += "_debug"
    
    output_path = f"results/{filename}.csv"
    
    # Write CSV header
    with open(output_path, 'w') as f:
        f.write("index,true_label,cluster_id,candidate_labels,complementary_label\n")
    
    print(f"Writing results to {output_path}")

    # ========== DEBUG MODE: Export cluster assignments without LLaVA ==========
    if args.debug:
        print("\n🔹 Debug mode: Exporting cluster assignments without querying LLaVA...")
        for batch_images, batch_true_labels, batch_cluster_ids, batch_label_options, batch_data_indices in tqdm(dataloader, desc="Exporting cluster data"):
            with open(output_path, 'a') as f:
                for i in range(len(batch_images)):
                    candidate_labels_str = str(batch_label_options[i]).replace(',', ';')
                    cluster_id_str = batch_cluster_ids[i] if batch_cluster_ids[i] is not None else ""
                    f.write(f"{batch_data_indices[i]},{batch_true_labels[i]},{cluster_id_str},\"{candidate_labels_str}\",\n")
        
        print(f"\n✅ Debug export complete: {output_path}")
        comp_df = pd.read_csv(output_path)
        print(comp_df.head(20))
        return
    
    # ========== LOAD LLaVA ==========
    print("Loading LLaVA model...")
    llava = LLaVAClassifier(
        model_path="llava-hf/llava-v1.6-mistral-7b-hf",
        baseprompt=args.prompt,
        device=device
    )

    # ========== RUN BATCH INFERENCE ==========
    for batch_images, batch_true_labels, batch_cluster_ids, batch_label_options, batch_data_indices in tqdm(dataloader, desc="Querying LLaVA (batch mode)"):
        
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
                cluster_id_str = batch_cluster_ids[i] if batch_cluster_ids[i] is not None else ""
                f.write(f"{batch_data_indices[i]},{batch_true_labels[i]},{cluster_id_str},\"{candidate_labels_str}\",{best_label}\n")

    # ========== DISPLAY RESULTS ==========
    print(f"\nSaved complementary results to {output_path}")
    comp_df = pd.read_csv(output_path)
    print(comp_df.head(10))


if __name__ == "__main__":
    main()
