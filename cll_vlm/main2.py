import os
import csv
import random
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.cifar10 import CIFAR10Dataset
from models.llava_classifier import LLaVAClassifier


def collate_fn(batch):
    """Dataloader collate: returns list of images and list of labels"""
    images, labels = zip(*batch)
    return list(images), list(labels)


def main(args):
    # Load CIFAR-10 dataset (50k ảnh train)
    dataset = CIFAR10Dataset(
        root=args.data_root,
        train=True,
        transform=None  # không augment, để nguyên ảnh
    )

    orig_dataset, shuffled_dataset = dataset.get_shuffled_labels_dataset(seed=42)

    dataloader = DataLoader(
        shuffled_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    # Load LLaVA model
    model = LLaVAClassifier(args.model_path)

    # Tạo thư mục output
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results2.csv")

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
    parser.add_argument("--data_root", type=str, default="/home/hamt/cll_vlm/cll_vlm/data/cifar10")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="./logs/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

    # batch = [(5, 2), (7, 1), (0, 2)]
    # images, labels = collate_fn(batch)
    # print(labels)  # [5, 7, 0]
