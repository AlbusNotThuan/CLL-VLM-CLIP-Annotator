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

    # Conditional imports for models that may not be available
    try:
        from models.qwen_classifier import QWENClassifier
    except ImportError:
        QWENClassifier = None

    try:
        from models.janus_classifier import JanusClassifier
    except ImportError:
        JanusClassifier = None

    try:
        from models.deepseek_vl2_classifier import DeepSeekVL2Classifier
    except ImportError:
        DeepSeekVL2Classifier = None

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
    
    print("🔹 Using model:", args.model)
    print("🔹 Using dataset:", args.dataset)
    print("🔹 Prompt:", args.prompt)

    # ========== LOAD DATA ==========
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
    elif args.dataset == "mnist":
        root_path = "/tmp2/maitanha/vgu/data/mnist"
        dataset = MNISTDataset(root=root_path, train=True, download=True)
    elif args.dataset == "kmnist":
        root_path = "/tmp2/maitanha/vgu/data/kmnist"
        dataset = KMNISTDataset(root=root_path, train=True, download=True)
    # elif args.dataset == "fmnist":
    #     root_path = "/tmp2/maitanha/vgu/data/fashion-mnist"
    #     dataset = FashionMNISTDataset(root=root_path, train=True, download=True)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples from {args.dataset} dataset")
    
    # Get class list from dataset
    class_list = dataset.classes
    
    # # Replace "_" with " " in class names for better readability
    # class_list = [c.replace("_", " ") for c in class_list]

    print(f"Dataset has {len(class_list)} classes")

    # # Custom ordered class list for cifar100
    # final_class_list = [
    #     "flatfish","mouse","otter","lobster","ray","caterpillar","leopard","crocodile",
    #     "possum","can","streetcar","plain","oak_tree","skunk","castle","kangaroo",
    #     "tulip","beaver","willow_tree","bridge","pear","wardrobe","rocket","television",
    #     "plate","poppy","whale","lizard","tank","sweet_pepper","elephant","shrew",
    #     "bus","dinosaur","crab","keyboard","lawn_mower","cattle","lamp","pine_tree",
    #     "camel","telephone","snail","palm_tree","mushroom","bicycle","motorcycle","skyscraper",
    #     "couch","table","orange","clock","shark","cockroach","spider","train",
    #     "worm","pickup_truck","baby","tractor","road","dolphin","bowl","chair",
    #     "wolf","fox","bed","orchid","chimpanzee","sunflower","rabbit","turtle",
    #     "mountain","cloud","girl","trout","house","maple_tree","raccoon","rose",
    #     "snake","lion","tiger","aquarium_fish","hamster","butterfly","squirrel","boy",
    #     "bottle","apple","forest","cup","sea","bee","bear","man",
    # ]
    # print(f"Using custom class list: {final_class_list[:10]} ... {final_class_list[-10:]}")
    # print(f"Total classes: {len(final_class_list)}")

    # # Add missing classes from dataset class list to the beginning of the final class list (to ensure all classes are included)
    # missing = [c for c in class_list if c not in final_class_list]

    # class_list = missing + final_class_list
    # print(f"Using custom class list: {class_list[:10]} ... {class_list[-10:]}")
    # print(f"Total classes: {len(class_list)}")


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

    # ========== MODEL PATH MAPPING ==========
    MODEL_PATHS = {
        "llava": "llava-hf/llava-v1.6-mistral-7b-hf",
        "llava_13b": "llava-hf/llava-v1.6-vicuna-13b-hf",
        "qwen2_7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen2_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen3_2b": "Qwen/Qwen3-VL-2B-Instruct",
        "qwen3_4b": "Qwen/Qwen3-VL-4B-Instruct",
        "qwen3_8b": "Qwen/Qwen3-VL-8B-Instruct",
        "qwen3_30b_a3b": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "janus": "deepseek-ai/Janus-Pro-7B",
        "deepseek_vl2": "deepseek-ai/deepseek-vl2-small",
    }

    # ========== LOAD MODEL ==========
    if args.model in ["llava", "llava_13b"]:
        model_path = args.model_path or MODEL_PATHS[args.model]
        print(f"Loading LLaVA model: {model_path}")
        classifier = LLaVAClassifier(
            model_path=model_path,
            baseprompt=args.prompt,
            device=device,
            device_map=device_map
        )
    elif args.model in ["qwen2_7b", "qwen2_3b", "qwen3_2b", "qwen3_4b", "qwen3_8b", "qwen3_30b_a3b"]:
        if QWENClassifier is None:
            raise ImportError(f"QWENClassifier is not available. Please upgrade transformers or use a different environment.")
        model_path = args.model_path or MODEL_PATHS[args.model]
        print(f"Loading Qwen model: {model_path}")
        classifier = QWENClassifier(
            model_path=model_path,
            device=device if device_map is None else None
        )
        classifier.set_temperature(args.temp)
    elif args.model == "janus":
        if JanusClassifier is None:
            raise ImportError(f"JanusClassifier is not available. Please upgrade transformers or use a different environment.")
        model_path = args.model_path or MODEL_PATHS[args.model]
        print(f"Loading Janus model: {model_path}")
        classifier = JanusClassifier(
            model_path=model_path,
            baseprompt=args.prompt,
            device=device if device_map is None else None
        )
        classifier.set_temperature(args.temp)

    elif args.model in ["deepseek_vl2", "deepseek_vl2_small", "deepseek_vl2_tiny"]:
        if DeepSeekVL2Classifier is None:
            raise ImportError(f"DeepSeekVL2Classifier is not available. Please upgrade transformers or use a different environment.")
        model_path = args.model_path or MODEL_PATHS[args.model]
        print(f"Loading DeepSeek VL2 model: {model_path}")
        classifier = DeepSeekVL2Classifier(
            model_path=model_path,
            device=device if device_map is None else None
        )
        classifier.set_temperature(args.temp)
    else:
        raise ValueError(f"Unknown model: {args.model}. Choose from: {list(MODEL_PATHS.keys())}")
    
    # Print memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    # ========== PREPARE OUTPUT FILE ==========
    if not os.path.exists("results/baseline2"):
        os.makedirs("results/baseline2")

    
    output_path = f"results/baseline2/{args.model}_{args.dataset}_true_temp{args.temp}_p2.csv"
    
    # Write CSV header
    with open(output_path, 'w') as f:
        f.write("index,true_label,predicted_label\n")
    
    print(f"Writing results to {output_path}")

    # ========== RUN BATCH INFERENCE ==========
    for batch_images, batch_true_labels, batch_indices in tqdm(dataloader, desc=f"Querying {args.model}"):
        
        # All images get the same label options (all classes)
        batch_label_options = [class_list for _ in range(len(batch_images))]

        # Run batch prediction
        predicted_labels = classifier.predict_best_label_batch(
            batch_images,
            batch_label_options,
            baseprompt=args.prompt
        )

        # Write results
        with open(output_path, 'a') as f:
            for i, pred_label in enumerate(predicted_labels):
                # Take only the first token to avoid hallucination
                # Replace line breaks with space, strip punctuation and whitespace, then take first word
                cleaned = pred_label.replace('\n', ' ').replace('\r', ' ').strip()
                first_token = cleaned.split()[0] if cleaned else pred_label
                # Remove trailing punctuation from first token
                answer = first_token.rstrip('.,!?;:')
                
                # answer = pred_label.strip()
                f.write(f"{batch_indices[i]},{batch_true_labels[i]},{answer}\n")

    # ========== DISPLAY RESULTS ==========
    print(f"\nSaved classification results to {output_path}")
    results_df = pd.read_csv(output_path)
    print(results_df.head(10))   


if __name__ == "__main__":

    # default_prompt = """
    #     Select the closest label from [{labels}]. Output strictly ONLY one label, no explanation or reasoning. If the image is unclear, choose the most probable label.
    # """

    # default_prompt = """
    #     "Examine the image carefully and choose which object is present from the following candidate list.\n"
    #     "Candidates: [{labels}].\n"
    #     "From this list, output only the name of the object that is present.\n"
    #     "Do not answer with any object that is not in the candidate list. Don't provide any explanation or reasoning,  use the exact wording from the candidate list."
    #     "Example answer: apple (if the object in the image is an apple, your answer should only be the label "apple")"
    # """

    # default_prompt = """
    #     ""Examine the image carefully and return the mose probable label from the following candidate list.\n"
    #     "Candidates: [{labels}].\n"
    #     "From this list, output only the true label of the image.\n"
    #     "Do not answer with any object that is not in the candidate list. Don't provide any explanation or reasoning,  use the exact wording from the candidate list.""
    # """

    default_prompt = """
        "You are given an image. "
        "Examine the image carefully and identify which objects from the candidate list are present.\n"
        "Candidates: [{labels}].\n"
        "From this list, choose EXACTLY ONE label that is most likely present in the image. "
        "Do no include any label that is not in the candidate list.""
        "If the image is blurry or unclear, you have to choose the most probable label from the candidate list."
        "Do not provide any explanation or reasoning",
        "Before you make the final response, carefully review if your answer IS a label in the candidate list. If your answer contains any label that is not in the candidate list, please revise your answer to only include labels from the candidate list."
    """

    # Parse arguments BEFORE importing torch
    parser = argparse.ArgumentParser(description="VLM True Label Classification (LLaVA/Qwen variants)")
    parser.add_argument("--model", type=str, default="llava",
                        choices=["llava", "llava_13b", "qwen2_7b", "qwen2_3b", "qwen3_4b", "qwen3_2b", "qwen3_8b", "qwen3_30b_a3b", "janus", "deepseek_vl2"],
                        help="Choose model variant")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature for sampling (if applicable)")
    parser.add_argument("--model_path", type=str, default=None, help="Custom model path (optional, uses defaults if not specified)")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Choose a Dataset to run (e.g., cifar10)")
    parser.add_argument("--prompt", type=str, default=default_prompt,
                        help="Custom prompt for VLM classification")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use (e.g., '0' for single GPU, '4,5' for multi-GPU)")
    args = parser.parse_args()
    
    # Set CUDA_VISIBLE_DEVICES BEFORE importing torch (inside main)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"🔹 Set CUDA_VISIBLE_DEVICES={args.gpu} (before importing torch)")
    
    # Now call main with args
    main(args)