import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.cifar10 import CIFAR10Dataset
from dataset.cifar20 import CIFAR20Dataset, CIFAR100Dataset
from models.clip_model import CLIPModel

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
    # ======= GPU setup =======
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = f"cuda:{args.gpu_id}"
    else:
        device = "cpu"
    print(f"[INFO] Using device: {device}")
    
    torch.backends.cudnn.benchmark = True

    # ======= Load CLIP =======
    clip_model = CLIPModel(model_name="ViT-L/14@336px", device=device)

    # ======= Load CSV =======
    df = pd.read_csv(args.input_csv)
    print(f"[INFO] Loaded {len(df)} rows from {args.input_csv}")

    # ======= Load CIFAR dataset =======
    dataset = load_dataset(args.data)
    orig_dataset, shuffled_dataset = dataset.get_shuffled_labels_dataset(seed=42)

    dataloader = DataLoader(
        shuffled_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )


    # ======= Prepare all data =======
    all_indices = range(len(df))
    images = [shuffled_dataset[idx][0] for idx in all_indices]
    labels = [df.loc[idx, "random_label"] for idx in all_indices]
    print(f"[INFO] Total samples (OL + CL): {len(images)}")

    batch_size = args.batch_size
    pos_similarities = [None] * len(df)
    neg_similarities = [None] * len(df)

    # ======= Encode theo batch =======
    print(f"[INFO] Computing CLIP similarities in batches of {batch_size} for all samples...")
    for i in tqdm(range(0, len(images), batch_size)):
        batch_imgs = images[i : i + batch_size]

        # Positive texts
        pos_batch_texts = [f"A photo of a {label}" for label in labels[i : i + batch_size]]

        # Negative texts
        neg_batch_texts = [f"A photo that is not a {label}" for label in labels[i : i + batch_size]]

        img_feats = clip_model.encode_image(batch_imgs)
        pos_text_feats = clip_model.encode_text(pos_batch_texts)
        neg_text_feats = clip_model.encode_text(neg_batch_texts)

        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        pos_text_feats /= pos_text_feats.norm(dim=-1, keepdim=True)
        neg_text_feats /= neg_text_feats.norm(dim=-1, keepdim=True)

        pos_sims = (img_feats * pos_text_feats).sum(dim=-1).cpu().numpy()
        neg_sims = (img_feats * neg_text_feats).sum(dim=-1).cpu().numpy()

        for j, sim in enumerate(pos_sims):
            pos_similarities[i + j] = sim

        for j, sim in enumerate(neg_sims):
            neg_similarities[i + j] = sim

    df["clip_pos_similarity"] = pos_similarities
    df["clip_neg_similarity"] = neg_similarities

    # ======= Save stage2 CSV =======
    base_name = os.path.splitext(os.path.basename(args.input_csv))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    stage2_csv_path = os.path.join(args.output_dir, f"{base_name}_stage2.csv")
    df.to_csv(stage2_csv_path, index=False)
    print(f"[SAVED] Wrote file with similarities → {stage2_csv_path}")

    # # ======= OL filtering thresholds =======
    # ol_df = df[df["predicted"] == "OL"].dropna(subset=["clip_similarity"])
    # sims_ol = ol_df["clip_similarity"].values
    # p5_ol, p10_ol, p20_ol = np.percentile(sims_ol, [95, 90, 80])
    # print(f"[INFO] OL Thresholds (bottom): 5%={p5_ol:.4f}, 10%={p10_ol:.4f}, 20%={p20_ol:.4f}")


    # def keep_top_ol_flip_rest(threshold, percent):
    #     new_df = df.copy()
    #     keep_mask = (new_df["predicted"] == "OL") & (new_df["clip_similarity"] >= threshold)
    #     flip_mask = (new_df["predicted"] == "OL") & (~keep_mask)
    #     new_df.loc[flip_mask, "predicted"] = "CL"
    #     out_path = os.path.join(args.output_dir, f"{base_name}_stage2_top_ol_{percent}.csv")
    #     new_df.to_csv(out_path, index=False)
    #     print(f"[SAVED] {out_path} — Kept {keep_mask.sum()} top OL | Flipped {flip_mask.sum()} OL → CL")
    
    # for p, th in [(5, p5_ol), (10, p10_ol), (20, p20_ol)]:
    #     keep_top_ol_flip_rest(th, p)


if __name__ == "__main__":
    parser = ArgumentParser(description="Stage 2: Compute CLIP similarity for all samples (OL + CL) and create bottom OL / top CL splits")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--data", type=str, required=True, help="Choose from: cifar10, cifar20, cifar100, ...")
    parser.add_argument("--output_dir", type=str, default="./ol_cll_logs/stage2", help="Output directory")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for CLIP encoding")
    args = parser.parse_args()
    main(args)
