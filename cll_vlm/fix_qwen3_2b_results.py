import sys
import os
sys.path.append("/tmp2/maitanha/vgu/cll_vlm")
import json
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse

from cll_vlm.models.qwen_classifier import QWENClassifier, extract_multi_label_full
from cll_vlm.dataset.cifar20 import CIFAR100Dataset
from qwen_vl_utils import process_vision_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to the json file to fix")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for VLM inference")
    args = parser.parse_args()

    json_path = args.json_path
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    dataset = CIFAR100Dataset(
        root="/tmp2/maitanha/vgu/cll_vlm/cll_vlm/data/cifar100",
        train=True,
        transform=None,
    )
    _, shuffled_dataset = dataset.get_shuffled_labels_dataset(seed=42)
    
    model_path = "Qwen/Qwen3-VL-2B-Instruct"
    classifier = QWENClassifier(model_path=model_path, device="auto")
    processor = classifier.processor
    model = classifier.model
    device = classifier.device

    # Batching config
    batch_size = args.batch_size
    
    # Identify which items need a second pass
    second_pass_items = []
    
    for idx, item in enumerate(data):
        ans = item.get("answer", [])
        if len(ans) > 1:
            second_pass_items.append({"idx": idx, "item": item})
        else:
            item["candidate_answer"] = ans
            item["answer"] = ans
            
    print(f"Total items needing second pass: {len(second_pass_items)}")
    
    for i in tqdm(range(0, len(second_pass_items), batch_size)):
        batch = second_pass_items[i:i+batch_size]
        
        batch_msgs = []
        batch_images = []
        
        for b in batch:
            img_idx = b["item"]["img_idx"]
            candidate_answer = b["item"]["answer"]
            
            # Load image from the shuffled dataset
            img, _ = shuffled_dataset[img_idx]
            
            # convert tensor to PIL
            if isinstance(img, torch.Tensor):
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype("uint8")
                else:
                    img_np = img_np.astype(np.uint8)
                img = Image.fromarray(img_np).copy()

            batch_images.append(img)
            
            prompt = (
                "You are given an image. "
                "Examine the image carefully and identify which object from the candidate list is present.\n"
                f"Candidates: ({', '.join(candidate_answer)}).\n"
                "From this list, choose only ONE label that is most likely present in the image. "
                "Do not include any label that is not in the candidate list. "
                "If you think none of the candidates are present, reply with exactly \"NO\".\n"
                "Provide a short reason for your answer.\n"
                "Before you make the final response, carefully review if your answer ONLY contains a label in the candidates.\n"
                "Your answer should be ONLY a JSON dict and nothing else, formatted as: "
                "{\"answer\": \"your chosen label\" or \"NO\", \"reason\": \"short explanation\"}\n"
                "Please don't reply in other formats."
            )
            batch_msgs.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }])
            
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_msgs
        ]
        image_inputs, video_inputs = process_vision_info(batch_msgs)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=None,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for b, out in zip(batch, output_texts):
            predicted, reason = extract_multi_label_full(out)
            item = b["item"]
            
            # Save candidates
            item["candidate_answer"] = item["answer"]
            item["answer"] = predicted
            
            if reason:
                if isinstance(item["reason"], list):
                    item["reason"].append(reason)
                else:
                    item["reason"] = [item["reason"], reason]
                    
    # Format and save
    final_data = []
    for item in data:
        new_item = {
            "img_idx": item["img_idx"],
            "true_label": item["true_label"],
            "shuffled_label": item["shuffled_label"],
            "candidate_answer": item.get("candidate_answer", item["answer"]),
            "answer": item.get("answer", []),
            "reason": item.get("reason", [])
        }
        final_data.append(new_item)

    out_path = json_path
    with open(out_path, 'w') as f:
        json.dump(final_data, f, indent=2)
        
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
