import os
import csv
import random
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm
import pdb
from dataset.cifar10 import CIFAR10Dataset
from models.llava_classifier import LLaVAClassifier
import pandas as pd

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
    elif data_name == "cifar100":
        # Ví dụ nếu bạn có class CIFAR100Dataset
        # dataset = CIFAR100Dataset(root=data_root_path, train=True, transform=None)
        raise NotImplementedError("Cần import và định nghĩa CIFAR100Dataset trước")
    elif data_name == "cifar20":
        # Ví dụ cho CIFAR20 (thường là CIFAR100 với coarse labels)
        # dataset = CIFAR20Dataset(root=data_root_path, train=True, transform=None)
        raise NotImplementedError("Cần định nghĩa cách load CIFAR20")
    else:
        raise ValueError(f"Dataset '{data_name}' chưa được hỗ trợ trong hàm load_dataset.")
    
    return dataset

def main(args):
    correspond1_round0_stage2 = pd.read_csv("/home/maitanha/cll_vlm/cll_vlm/logs/correspond_1_stage2.csv")
    correspond1_round0_stage2_OL = correspond1_round0_stage2[correspond1_round0_stage2["predicted"] == "OL"]
    threshold = correspond1_round0_stage2_OL["clip_similarity"].quantile(0.8)

    #round0_OL_20 = correspond1_round0_stage2_OL[correspond1_round0_stage2_OL["clip_similarity"] >= threshold].copy()
    round0_OL_80 = correspond1_round0_stage2_OL[correspond1_round0_stage2_OL["clip_similarity"] < threshold].copy()
    class MockConfig:
        def __init__(self):
            self.debug = True

    # Example usage
    cfg = MockConfig()  # Replace with actual config object

    dataset = CIFAR10Dataset(
        root='/home/maitanha/cll_vlm/cll_vlm/data/cifar10',
        train=True,
        transform=None  # No transform for visualization
    )
    dataset.cfg = cfg
    _, shuffled_1 = dataset.get_shuffled_labels_dataset(seed=42)

    subset_indices = round0_OL_80["index"].tolist()

    subset_dataset = shuffled_1.get_subset_by_indices(subset_indices)

    # for comparison
    # subset_orig = subset_dataset

    # print(type(subset_dataset))  # <class 'dataset.cifar10.CIFAR10Dataset'>
    # print(len(subset_dataset))
    orig_dataset, shuffled_dataset = subset_dataset.get_shuffled_labels_dataset(seed=42)

    dataloader = DataLoader(shuffled_dataset, batch_size=64, collate_fn=collate_fn, shuffle=False)

    # Load LLaVA model
    model = LLaVAClassifier(args.model_path, args.baseprompt)

    # Tạo thư mục output
    os.makedirs(args.output_dir, exist_ok=True)
    if hasattr(args, "csv_name") and args.csv_name:
        csv_path = os.path.join(args.output_dir, args.csv_name)
    else:
        csv_path = os.path.join(args.output_dir, "result.csv")
        
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
    parser.add_argument("--model_path", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--data_root", type=str, default="./data/cifar10")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="./logs/")
    parser.add_argument("--csv_name", type=str, default=None, help="Custom CSV filename")
    parser.add_argument("--baseprompt", type=str, default="Does the label '{label}' match this image? Answer with only a single word: YES or NO.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

    # batch = [(5, 2), (7, 1), (0, 2)]
    # images, labels = collate_fn(batch)
    # print(labels)  # [5, 7, 0]
