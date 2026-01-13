import os
import argparse
import ast
import torch
import pandas as pd
from tqdm import tqdm

from models.llava_classifier import LLaVAClassifier
from dataset.cifar10 import CIFAR10Dataset
from dataset.cifar20 import CIFAR20Dataset

def main():
    # ========== ARGUMENTS ==========
    parser = argparse.ArgumentParser(description="Stage 3 - LLaVA Least-Match Selector (batch mode)")
    parser.add_argument("--dataset", type=str, required=True, help="Choose a Dataset to run (e.g., cifar10)")    
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to CLIP similarity CSV file (contains similarities list)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt for LLaVA (e.g. '<image> Which of the following labels least describes this image? Answer with one word from [{labels}].')")
    # parser.add_argument("--root_path", type=str, default="/home/maitanha/cll_vlm/cll_vlm/data/cifar10",
                        # help="Path to CIFAR-10 dataset root")
    parser.add_argument("--output_csv", type=str, default="clip_llava_least_result.csv",
                        help="Output CSV file for LLaVA least-match results")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for LLaVA inference")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ========== LOAD DATA ==========
    if args.dataset == "cifar10":
        root_path = "/home/maitanha/cll_vlm/cll_vlm/data/cifar10"
        dataset = CIFAR10Dataset(root=root_path, train=True)
        label_names = dataset.classes
    elif args.dataset == "cifar20":
        root_path = "/home/maitanha/cll_vlm/cll_vlm/data/cifar20"
        dataset = CIFAR20Dataset(root=root_path, train=True)
        raw_labels = dataset.classes
        
        label_names = []
        for label in raw_labels:
            # nếu bắt đầu bằng 'vehicles_', chuyển thành 'vehicles'
            if label.startswith('vehicles_'):
                label_names.append('vehicles')
            else:
                # thay dấu _ bằng khoảng trắng
                label_names.append(label.replace('_', ' '))

    # Load similarity results
    df = pd.read_csv(args.input_csv)
    df["similarities"] = df["similarities"].apply(ast.literal_eval)
    total_samples = len(df)
    print(f"Loaded {total_samples} samples from {args.input_csv}")

    # ========== LOAD LLaVA ==========
    print("Loading LLaVA model...")
    llava = LLaVAClassifier(
        model_path="llava-hf/llava-v1.6-mistral-7b-hf",
        baseprompt=args.prompt,
        device=device
    )

    # ========== RUN BATCH INFERENCE ==========
    least_records = []
    batch_size = args.batch_size

    for start_idx in tqdm(range(0, total_samples, batch_size), desc="Querying LLaVA (least mode)"):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_rows = df.iloc[start_idx:end_idx]

        batch_images = []
        batch_label_options = []
        batch_true_labels = []
        batch_data_indices = []

        # === build batch ===
        for _, row in batch_rows.iterrows():
            data_index = int(row["index"])               # from Stage 1
            true_label = row["true_label"]
            sims = torch.tensor(row["similarities"])

            # sort ascending → lấy 4 nhãn có similarity thấp nhất
            sorted_idx = torch.argsort(sims, descending=False).tolist()
            least_labels = [label_names[j] for j in sorted_idx[:4]]

            img, _ = dataset[data_index]

            batch_images.append(img)
            batch_label_options.append(least_labels)
            batch_true_labels.append(true_label)
            batch_data_indices.append(data_index)

        best_labels = llava.predict_best_label_batch(
            batch_images,
            batch_label_options,
            baseprompt=args.prompt
        )

        for i, best_label in enumerate(best_labels):
            least_records.append({
                "index": batch_data_indices[i],
                "true_label": batch_true_labels[i],
                "candidate_labels": batch_label_options[i],
                "complementary_label": best_label
            })

    # ========== SAVE RESULTS ==========
    least_df = pd.DataFrame(least_records)
    least_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved least-match results to {args.output_csv}")
    print(least_df.head(10))


if __name__ == "__main__":
    main()
