import json
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import List, Dict, Union
from tqdm import tqdm
from PIL import Image
import numpy as np
import re
from collections import Counter


def extract_all_reasons(raw: str):
    """
    Extract all JSON objects from Qwen output.
    Return:
        decision: YES / NO
        reasons: list[str]
    """
    if not raw:
        return "NO", []

    matches = re.findall(r"\{.*?\}", raw, flags=re.DOTALL)
    answers = []
    reasons = []

    for m in matches:
        try:
            obj = json.loads(m)
            if "answer" in obj:
                answers.append(obj["answer"].upper())
            if "reason" in obj and isinstance(obj["reason"], str):
                reasons.append(obj["reason"].strip())
        except Exception:
            continue

    if answers:
        decision = Counter(answers).most_common(1)[0][0]
    else:
        decision = "NO"

    # remove duplicate reasons while preserving order
    seen = set()
    clean_reasons = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            clean_reasons.append(r)

    return decision, clean_reasons


class QWENClassifier:
    def __init__(self, model_path="Qwen/Qwen2.5-VL-7B-Instruct", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            # low_cpu_mem_usage=True,
            device_map= None
        ).to(self.device)

    # @classmethod
    # def build_model(cls, args):
    #     return cls(args.model_path)
    
    # def process_qwen(data, label_names, model_info, config=None, output_dir=None, start_idx=0):
    #     if config is None:
    #         config = {}

    #     if output_dir:
    #         os.makedirs(output_dir, exist_ok=True)


    def generate_batch_results(self, data, shuffled_label_indices, true_label_indices, fine_classes, prompt_type, output_path, batch_size, start_idx=0):
        total_images = len(data)
        num_batches = (total_images + batch_size - 1) // batch_size
        results = []

        for batch_idx in tqdm(range(num_batches)):
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = min((batch_idx + 1) * batch_size, total_images)

            batch_images = data[batch_start_idx:batch_end_idx]

            batch_shuffled_labels = [
                fine_classes[idx] for idx in shuffled_label_indices[batch_start_idx:batch_end_idx]
            ]
            batch_true_labels = [
                fine_classes[idx] for idx in true_label_indices[batch_start_idx:batch_end_idx]
            ]
            
            if isinstance(batch_images, torch.Tensor) and batch_images.dim() == 3:
                images = batch_images.unsqueeze(0)    # Add batch dimension [1, C, H, W]

            if isinstance(batch_images, torch.Tensor):
                imgs = []

                # Convert to PIL image
                if batch_images.dim() == 4:
                    for i in range (batch_images.shape[0]):
                        image = batch_images[i, :]
                        image = image.cpu().numpy().transpose(1, 2, 0)
                        if image.max() <= 1.0:
                            image = (image * 255).astype("uint8")
                        else:
                            image = image.astype(np.uint8)
                        image = Image.fromarray(image)
                        imgs.append(image)
                if batch_images.dim() == 3:
                    image = batch_images.cpu().numpy().transpose(1, 2, 0)
                    if image.max() <= 1.0:
                        image = (image * 255).astype("uint8")
                    else:
                        image = image.astype(np.uint8)
                    image = Image.fromarray(image).copy()
                    imgs.append(image)
                
                batch_images = imgs

        
            if prompt_type == "binary":
                batch_messages = []
                for img, label in zip(batch_images, batch_shuffled_labels):
                    prompt = (
                        f"You are given an image. Does the label '{label}' correspond to this image?"
                                "Answer ONLY with a valid JSON object.\n:"
                                # "Return only a JSON object formatted as: "
                                "Format: {'answer': 'YES' or 'NO', 'reason': explain your reason for choosing them}.\n"
                    )

                    # For debugging
                    # prompt = "Describe the image."

                    # messages = [
                    #     {
                    #         "role": "user",
                    #         "content": [
                    #             {"type": "image", "image": img},
                    #             {"type": "text", "text": prompt},
                    #         ],
                    #     }
                    # ]

                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]

                    batch_messages.append(messages)

                # Generate answer
                texts = [
                    self.processor.apply_chat_template(
                        msg,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for msg in batch_messages
                ]

                image_inputs, video_inputs = process_vision_info(batch_messages)

                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                # # BUG LOG
                # print(f"Line 135: {texts[0]}")

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=None, 
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                # print(f"[DEBUG] Line 151 generated_ids_trimmed: {generated_ids_trimmed}")

                output_texts = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                for i, text in enumerate(output_texts):
                    decision, reasons = extract_all_reasons(text)

                    results.append({
                        "img_idx": start_idx + batch_start_idx + i,
                        "true_label": batch_true_labels[i],
                        "shuffled_label": batch_shuffled_labels[i],
                        "answer": decision,
                        "reason": reasons,
                    })

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)
        
        return results