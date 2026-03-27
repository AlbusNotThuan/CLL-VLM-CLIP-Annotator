import sys
import os
import json
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse

# Add parent directory to path to find cll_vlm
# Based on the file structure seen in list_dir
sys.path.append("/tmp2/maitanha/vgu/cll_vlm")

from cll_vlm.models.qwen_classifier import QWENClassifier, extract_multi_label_full
from cll_vlm.dataset.cifar20 import CIFAR100Dataset
from qwen_vl_utils import process_vision_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to the json file to fix")
    parser.add_argument("--model_path", type=str, required=True, help="Model path (e.g. Qwen/Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for VLM inference")
    args = parser.parse_args()

    json_path = args.json_path
    model_path = args.model_path
    batch_size = args.batch_size
    
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} does not exist.")
        return

    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    print(f"Loading CIFAR100 dataset...")
    dataset = CIFAR100Dataset(
        root="/tmp2/maitanha/vgu/cll_vlm/cll_vlm/data/cifar100",
        train=True,
        transform=None,
    )
    # Use seed 42 to match original shuffled dataset
    _, shuffled_dataset = dataset.get_shuffled_labels_dataset(seed=42)
    
    print(f"Loading model {model_path}...")
    classifier = QWENClassifier(model_path=model_path, device="auto")
    processor = classifier.processor
    model = classifier.model
    device = classifier.device

    # Identify which items need a second pass (answer list has more than 1 item)
    # We will process all items in order and save incrementally
    base, ext = os.path.splitext(json_path)
    out_path = f"{base}_prompt_v1{ext}"
    
    print(f"Refining results and saving incrementally to {out_path}...")
    
    # We'll use a sliding batch approach to refine items in order while saving
    refine_indices = [idx for idx, item in enumerate(data) if len(item.get("answer", [])) > 1]
    refine_results = {} # map idx -> refined_item
    
    print(f"Total items needing second pass: {len(refine_indices)}")

    with open(out_path, 'w') as f:
        f.write("[\n")
        
        # We'll iterate through all items. If we hit an item needing refinement, 
        # we check if we've already processed its batch. If not, we process the next batch.
        refine_ptr = 0
        pbar = tqdm(total=len(data))
        
        for i, item in enumerate(data):
            if i in refine_indices:
                # If this item needs refinement and isn't in our results cache, 
                # process the next batch of refinement items.
                if i not in refine_results:
                    batch_idxs = refine_indices[refine_ptr : refine_ptr + batch_size]
                    refine_ptr += len(batch_idxs)
                    
                    # Process batch
                    batch_msgs = []
                    for b_idx in batch_idxs:
                        b_item = data[b_idx]
                        img_idx = b_item["img_idx"]
                        candidate_answer = b_item["answer"]
                        
                        img, _ = shuffled_dataset[img_idx]
                        if isinstance(img, torch.Tensor):
                            img_np = img.cpu().numpy().transpose(1, 2, 0)
                            if img_np.max() <= 1.0:
                                img_np = (img_np * 255).astype("uint8")
                            else:
                                img_np = img_np.astype(np.uint8)
                            img = Image.fromarray(img_np).copy()

                        prompt = (
                            "You are given an image. "
                            "Examine the image carefully and identify which objects from the candidate list are present.\n"
                            f"Candidates: ({', '.join(candidate_answer)}).\n"
                            "From this list, choose only ONE label that is most likely present in the image. "
                            "Do not include any label that is not in the candidate list. "
                            "If none of the candidates are present, return an empty list.\n"
                            "Return ONLY a JSON object and nothing else, formatted as: "
                            "{\"answer\": [...], \"reason\": \"...\"}"
                        )
                        batch_msgs.append([{
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": prompt},
                            ],
                        }])

                    if batch_msgs:
                        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_msgs]
                        image_inputs, video_inputs = process_vision_info(batch_msgs)
                        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True).to(device)

                        with torch.no_grad():
                            generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False, 
                                                         pad_token_id=processor.tokenizer.pad_token_id, eos_token_id=None)

                        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                        for b_idx, out in zip(batch_idxs, output_texts):
                            predicted, reason = extract_multi_label_full(out)
                            orig_item = data[b_idx]
                            refined = {
                                "img_idx": orig_item["img_idx"],
                                "true_label": orig_item.get("true_label"),
                                "shuffled_label": orig_item.get("shuffled_label"),
                                "candidate_answer": orig_item["answer"],
                                "answer": predicted,
                                "reason": orig_item.get("reason", [])
                            }
                            if reason:
                                if isinstance(refined["reason"], list):
                                    refined["reason"].append(reason)
                                else:
                                    refined["reason"] = [refined["reason"], reason]
                            refine_results[b_idx] = refined

                # Use refined item
                item_to_save = refine_results[i]
                # Cleanup cache to save memory
                del refine_results[i]
            else:
                # Use original item
                item_to_save = {
                    "img_idx": item["img_idx"],
                    "true_label": item.get("true_label"),
                    "shuffled_label": item.get("shuffled_label"),
                    "candidate_answer": item.get("candidate_answer", item.get("answer", [])),
                    "answer": item.get("answer", []),
                    "reason": item.get("reason", [])
                }

            # Save to file
            if i > 0:
                f.write(",\n")
            json.dump(item_to_save, f, indent=2)
            if i % 10 == 0:
                f.flush()
            pbar.update(1)
            
        f.write("\n]\n")
        pbar.close()

    print(f"Refined results saved to {out_path}")

if __name__ == '__main__':
    main()
