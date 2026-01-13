import os
import csv
import random
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm
import pdb
from dataset.cifar10 import CIFAR10Dataset
from dataset.cifar20 import CIFAR20Dataset, CIFAR100Dataset
from models.llava_classifier import LLaVAClassifier
from models.qwen_classifier import QWENClassifier


def collate_fn(batch):
    """Dataloader collate: returns list of images and list of labels"""
    images, labels = zip(*batch)
    return list(images), list(labels)

def load_dataset(data_name):
    """
    Hàm if/else để load dataset dựa trên tên được truyền vào.
    Giả định folder data nằm tại ./data/{data_name}
    """
    # Cấu hình đường dẫn root cho từng loại data (có thể sửa lại tùy cấu trúc thư mục thực tế)
    data_root_path = os.path.join("./data", data_name)

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

def main(args):
    dataset = load_dataset(args.data)

    orig_dataset, shuffled_dataset = dataset.get_shuffled_labels_dataset(seed=42)

    dataloader = DataLoader(
        shuffled_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    # Load VQA model based on model_type
    if args.model_type == "qwen":
        model = QWENClassifier(args.model_path, args.baseprompt)
    elif args.model_type == "llava":
        model = LLaVAClassifier(args.model_path, args.baseprompt)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    log_dir = f"/home/maitanha/cll_vlm/cll_vlm/ol_cll_logs/{args.data}"
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    csv_path = os.path.join(log_dir, args.csv_name if args.csv_name else f"results_{args.data}.csv")

        
    # Ghi kết quả
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "random_label", "predicted", "raw_answer"])

        index = 0
        dataset_pos = 0
        for images, shuffled_labels in tqdm(dataloader):
            
            batch_len = len(images)

            true_label_indices = [orig_dataset[i][1] for i in range(dataset_pos, dataset_pos + batch_len)]
            true_labels = [dataset.classes[idx] for idx in true_label_indices]
            
            # convert shuffled label indices -> names so model.predict gets names (optional)
            shuffled_label_names = [dataset.classes[idx] for idx in shuffled_labels]

            # pdb.set_trace()

            # Hỏi LLaVA
            answers = model.predict(images, shuffled_label_names)

            # Ghi vào CSV
            for tl, rl, ans in zip(true_labels, shuffled_label_names, answers):
                predicted = "OL" if ans == "YES" else "CL"
                writer.writerow([index, tl, rl, predicted, ans])
                index += 1

            dataset_pos += batch_len

    print(f"✅ Done. Results saved at {csv_path}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["llava", "qwen"], default="llava",
                        help="VQA model type to use")
    parser.add_argument("--model_path", type=str, default=None,
                        help="HuggingFace model path (default: llava-hf/llava-v1.6-mistral-7b-hf for llava, Qwen/Qwen2-VL-7B-Instruct for qwen)")
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--csv_name", type=str, default=None, help="Custom CSV filename")
    parser.add_argument("--baseprompt", type=str, default="Does the label '{label}' match this image? Answer with only a single word: YES or NO.")
    
    args = parser.parse_args()
    
    # Set default model_path based on model_type if not provided
    if args.model_path is None:
        if args.model_type == "llava":
            args.model_path = "llava-hf/llava-v1.6-mistral-7b-hf"
        elif args.model_type == "qwen":
            args.model_path = "Qwen/Qwen2-VL-7B-Instruct"
    
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

    # batch = [(5, 2), (7, 1), (0, 2)]
    # images, labels = collate_fn(batch)
    # print(labels)  # [5, 7, 0]
