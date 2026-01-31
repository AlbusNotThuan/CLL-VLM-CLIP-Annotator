import os
import sys
import argparse

def collate_fn(batch):
    """Custom collate function"""
    images, true_labels = zip(*batch)
    indices = list(range(len(images)))
    return list(images), list(true_labels), indices


def main(args):
    import torch
    import pandas as pd
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from models.llava_classifier import LLaVAClassifier
    from dataset.cifar10 import CIFAR10Dataset
    from dataset.cifar20 import CIFAR20Dataset, CIFAR100Dataset
    from dataset.tiny200 import Tiny200Dataset
    from dataset.mnist import MNISTDataset
    from dataset.kmnist import KMNISTDataset
    
    torch.cuda.empty_cache()
    
    # Multi-GPU or single GPU
    if "," in args.gpu:
        device = "cuda:0"
        device_map = "auto"
        print(f"🔹 Using physical GPUs: {args.gpu} (mapped to PyTorch as cuda:0, cuda:1, ...)")
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_map = None
        print(f"🔹 Using single GPU")
    
    print("🔹 Using dataset:", args.dataset)
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
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Created DataLoader with batch_size={args.batch_size}")

    # ========== LOAD LLaVA ==========
    print("Loading LLaVA model...")
    llava = LLaVAClassifier(
        model_path="llava-hf/llava-v1.6-mistral-7b-hf",
        baseprompt=args.prompt,
        device=device,
        device_map=device_map
    )
    
    # Print memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    # ========== PREPARE OUTPUT FILE ==========
    if not os.path.exists("results"):
        os.makedirs("results")
    
    output_path = f"results/llava_{args.dataset}_true.csv"
    
    # Write CSV header
    with open(output_path, 'w') as f:
        f.write("index,true_label,predicted_label\n")
    
    print(f"Writing results to {output_path}")

    # ========== RUN BATCH INFERENCE ==========
    for batch_images, batch_true_labels, batch_indices in tqdm(dataloader, desc="Querying LLaVA"):
        
        # All images get the same label options (all classes)
        batch_label_options = [class_list for _ in range(len(batch_images))]

        # Run batch prediction
        predicted_labels = llava.predict_best_label_batch(
            batch_images,
            batch_label_options,
            baseprompt=args.prompt
        )

        # Write results
        with open(output_path, 'a') as f:
            for i, pred_label in enumerate(predicted_labels):
                f.write(f"{batch_indices[i]},{batch_true_labels[i]},{pred_label}\n")

    # ========== DISPLAY RESULTS ==========
    print(f"\nSaved classification results to {output_path}")
    results_df = pd.read_csv(output_path)
    print(results_df.head(10))   


if __name__ == "__main__":
    # Parse arguments BEFORE importing torch
    parser = argparse.ArgumentParser(description="LLaVA True Label Classification")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Choose a Dataset to run (e.g., cifar10)")
    parser.add_argument("--prompt", type=str, default="Select the closest label from [{labels}]. Output strictly one label, no explanation or reasoning. If the image is unclear, choose the most probable label.",
                        help="Custom prompt for LLaVA classification")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for LLaVA inference")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use (e.g., '0' for single GPU, '4,5' for multi-GPU)")
    args = parser.parse_args()
    
    # Set CUDA_VISIBLE_DEVICES BEFORE importing torch (inside main)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"🔹 Set CUDA_VISIBLE_DEVICES={args.gpu} (before importing torch)")
    
    # Now call main with args
    main(args)