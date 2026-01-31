
import sys
import os
import torch
import json
import yaml
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = "/tmp2/maitanha/vgu/cll_vlm"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cll_vlm.dataset.cifar20 import CIFAR100Dataset
from cll_vlm.models import QWENClassifier, LLaVAClassifier

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # === Cấu hình ===
    config_path = os.path.join(PROJECT_ROOT, "cll_vlm/config/config.yaml")
    config = load_config(config_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "qwen"
    dataset_name = "cifar100"
    
    save_dir = "/tmp2/maitanha/vgu/cll_vlm/cll_vlm/ol_cll_logs"
    save_path = os.path.join(save_dir, f"{model_name}_{dataset_name}_label_description_v2.json")
    os.makedirs(save_dir, exist_ok=True)
    
    # Load Dataset for Labels
    print("Loading dataset labels...")
    dataset_root = config["data"]["paths"]["cifar100"]
    dataset = CIFAR100Dataset(root=dataset_root, train=True)
    fine_classes_raw = dataset.get_fine_classes()
    fine_classes = [CIFAR100Dataset.preprocess_label(lbl) for lbl in fine_classes_raw]
    print(f"Loaded {len(fine_classes)} labels.")

    # Load Model
    print(f"Loading model: {model_name}...")
    if model_name == "llava":
        model_path = config["models"][model_name]["model_url"]
        # LLaVA might not have batch text generation implemented yet
        model = LLaVAClassifier(model_path=model_path)
    elif model_name in ["qwen", "qwen3_2b", "qwen3_8b"]:
        model_path = config["models"][model_name]["model_url"]
        model = QWENClassifier(model_path=model_path, device=device)
    else:
        raise ValueError(f"Unsupported model '{model_name}'.")

    # === PROMPT ===
    system_content_general = "You are a helpful assistant who can describe any object in the world based on how it looks or where it is typically found."

    # prompt_general_visual = "What are the distinguishable characteristics that can be used to differentiate a {label} from other objects based on just a photo? \
    #                         Produce an exhaustive list of visual attributes, shape, color, structure, material, and contextual or habitat cues that help identify it visually. \
    #                         Each description should be a single sentence starting with 'An object that...'. \
    #                         The object name should not appear in the description. \
    #                         Structure your response as a list of single sentences."

    prompt_general_visual = "What are the distinguishable VISUAL characteristics that can be used to differentiate a {label} from other objects based ONLY on a photo? \
                            Produce an exhaustive list of **strictly visual attributes** such as shape, color, texture, structure, and physical appearance. \
                            **EXCLUDE** any non-visual traits like taste, smell, sound, usage, nutritional value, or internal feelings. \
                            Each description should be a single sentence starting with 'An object that...'. \
                            The object name should not appear in the description. \
                            Structure your response as a list of single sentences."
                            
    prompt_general_location = "In what environments or contexts is a {label} typically found or used? \
                            Produce a list of typical locations, habitats, or usage settings where one might expect to see it. \
                            Each sentence should begin with 'An object that is found in...' or 'An object typically used in...'. \
                            Do not mention the name of the object in any sentence. \
                            Structure your response as a list of single sentences."

    # Prepare Prompts
    print("Preparing prompts...")
    all_labels = fine_classes
    visual_prompts = [prompt_general_visual.format(label=label) for label in all_labels]
    location_prompts = [prompt_general_location.format(label=label) for label in all_labels]

    # Batch Generation
    BATCH_SIZE = 32
    
    print(f"Generating visual descriptions (Batch Size: {BATCH_SIZE})...")
    visual_responses = model.generate_text_batch(
        visual_prompts, 
        system_content=system_content_general, 
        batch_size=BATCH_SIZE
    )

    print(f"Generating location descriptions (Batch Size: {BATCH_SIZE})...")
    location_responses = model.generate_text_batch(
        location_prompts, 
        system_content=system_content_general, 
        batch_size=BATCH_SIZE
    )

    # Parse and Store
    print("Parsing results...")
    label_descriptions = {}
    for i, label in enumerate(all_labels):
        # Parse Visual
        v_resp = visual_responses[i]
        v_lines = [line.strip("-• ").strip() for line in v_resp.strip().split("\n") if line.strip()]
        v_lines = [
            line for line in v_lines 
            if line.lower().startswith("an object") and line.strip().endswith((".", "!", "?"))
        ]

        # Parse Location
        l_resp = location_responses[i]
        l_lines = [line.strip("-• ").strip() for line in l_resp.strip().split("\n") if line.strip()]
        l_lines = [
            line for line in l_lines 
            if line.lower().startswith("an object") and line.strip().endswith((".", "!", "?"))
        ]

        label_descriptions[label] = {
            "visual": v_lines,
            "context": l_lines
        }

    # Save
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(label_descriptions, f, indent=2, ensure_ascii=False)

    print(f"Saved label descriptions to {save_path}")

if __name__ == "__main__":
    main()
