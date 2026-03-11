import json
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
try:
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration
    HAS_QWEN3 = True
except ImportError:
    HAS_QWEN3 = False

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


def extract_multi_label_answers(raw: str, valid_labels: set = None) -> list:
    """
    Extract predicted labels from Qwen output for multi_label prompt type.
    Expected JSON format: {'answer': [...], 'reason': '...'}
    Returns a list of valid label strings.
    """
    if not raw:
        return []

    # Try to find a JSON object in the output
    matches = re.findall(r"\{.*?\}", raw, flags=re.DOTALL)
    for m in matches:
        try:
            obj = json.loads(m)
            if "answer" in obj and isinstance(obj["answer"], list):
                labels = [str(l).strip() for l in obj["answer"]]
                if valid_labels is not None:
                    # Only keep labels that are in the valid set (case-insensitive)
                    valid_lower = {v.lower(): v for v in valid_labels}
                    labels = [valid_lower[l.lower()] for l in labels if l.lower() in valid_lower]
                return labels
        except Exception:
            continue

    # Fallback: try the whole raw string as JSON
    try:
        obj = json.loads(raw)
        if "answer" in obj and isinstance(obj["answer"], list):
            labels = [str(l).strip() for l in obj["answer"]]
            if valid_labels is not None:
                valid_lower = {v.lower(): v for v in valid_labels}
                labels = [valid_lower[l.lower()] for l in labels if l.lower() in valid_lower]
            return labels
    except Exception:
        pass

    return []


def extract_multi_label_full(raw: str, valid_labels: set = None) -> (list, str):
    """
    Extract predicted labels and reason from Qwen output for multi_label prompt type.
    Expected JSON format: {'answer': [...], 'reason': '...'}
    Returns (list of valid labels, reason string).
    Always tries to recover 'reason' even when 'answer' is empty or missing.
    """
    if not raw:
        return [], ""

    matches = re.findall(r"\{.*?\}", raw, flags=re.DOTALL)

    # Pass 1: look for a proper {"answer": [...], "reason": ...} object
    # Return even if labels is empty — reason is still useful for logging
    for m in matches:
        try:
            obj = json.loads(m)
            if "answer" in obj and isinstance(obj["answer"], list):
                labels = [str(l).strip() for l in obj["answer"]]
                if valid_labels is not None:
                    valid_lower = {v.lower(): v for v in valid_labels}
                    labels = [valid_lower[l.lower()] for l in labels if l.lower() in valid_lower]
                reason = obj.get("reason", "")
                return labels, reason
        except Exception:
            continue

    # Pass 2: no valid "answer" list found — try to recover at least "reason"
    for m in matches:
        try:
            obj = json.loads(m)
            if "reason" in obj and isinstance(obj["reason"], str):
                return [], obj["reason"].strip()
        except Exception:
            continue

    return [], ""


def load_qwen(model_path: str, device: str = None):
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        # Set left padding for decoder-only models (required for correct batch generation)
        processor.tokenizer.padding_side = 'left'
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Qwen processor.\n"
            f"Make sure transformers supports this model.\n{e}"
        )

    print(f" [DEBUG] Model path: {model_path}")
    
    # Use device_map="auto" for multi-GPU if device is not specifically set to a single one
    # This respects CUDA_VISIBLE_DEVICES
    if device in [None, "cuda", "auto"]:
        device_map = "auto"
    else:
        device_map = device

    if "Qwen2.5" in model_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            trust_remote_code=True,
            device_map=device_map,
        )

    elif "Qwen3" in model_path:
        if not HAS_QWEN3:
            raise ImportError("Qwen3 model requires 'transformers' package with Qwen3 support.")
        
        # Qwen3-VL-30B-A3B-Instruct uses MOE architecture
        if "30B-A3B" in model_path:
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                trust_remote_code=True,
                device_map=device_map,
            )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                trust_remote_code=True,
                device_map=device_map,
            )
    else:
        raise ValueError(f"Unsupported Qwen model path: {model_path}")
    
    model.eval() # Ensure model is in evaluation mode
    return processor, model
    
class QWENClassifier:
    def __init__(self, model_path="Qwen/Qwen2.5-VL-7B-Instruct", device=None):
        self.processor, self.model = load_qwen(model_path, device)
        # Use the actual device assigned to the model (handles multi-GPU device_map)
        self.device = self.model.device

    def generate_batch_results(self, data, shuffled_label_indices, true_label_indices, fine_classes, prompt_type, output_path, batch_size, start_idx=0, label_description_path=None, **kwargs):
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

                    # prompt = (
                    #     f"You are given an image.\n"
                    #     "First, identify the SINGLE main object that occupies the central visual focus "
                    #     "or is most salient in the image.\n"
                    #     "Do NOT consider background, environment, or secondary objects.\n\n"
                    #     f"Then decide whether the label '{label}' correctly describes that main object.\n"
                    #     "If the label matches only background or contextual elements, answer NO.\n\n"
                    #     "Answer ONLY with a valid JSON object formatted as: "
                    #     "{'answer': 'YES' or 'NO', 'reason': explain your reason for choosing them}."
                    # )

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]

                    # messages = [
                    #     {"role": "system", "content": "You are a helpful assistant."},
                    #     {
                    #         "role": "user",
                    #         "content": [
                    #             {"type": "image", "image": img},
                    #             {"type": "text", "text": prompt},
                    #         ],
                    #     }
                    # ]

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
                        max_new_tokens=64,
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

            elif prompt_type == "label_description":

                if label_description_path is None:
                    raise ValueError("label_description_path is required when prompt_type is 'label_description'")
                with open(label_description_path, "r", encoding="utf-8") as f:
                    label_descriptions = json.load(f)
                
                batch_messages = []
                for img, shuffled_label in zip(batch_images, batch_shuffled_labels):
                    desc = label_descriptions.get(shuffled_label)
                    if desc is None:
                        key_alt = shuffled_label.replace("_", " ").strip().lower()
                        desc = label_descriptions.get(key_alt)
                    if desc is None:
                        desc = {"visual": [], "context": []}

                    visual_list = desc.get("visual", [])
                    context_list = desc.get("context", [])
                    visual_text = "\n".join(f"- {s}" for s in visual_list) if visual_list else "(No visual description)"
                    context_text = "\n".join(f"- {s}" for s in context_list) if context_list else "(No context description)"

                    prompt = (
                        f"You are given an image and the following descriptions for the label '{shuffled_label}'.\n\n"
                        "Visual descriptions:\n" + visual_text + "\n\n"
                        "Context descriptions:\n" + context_text + "\n\n"
                        f"According to the descriptions above, does the image correctly depicts the label '{shuffled_label}'.\n"
                        "Answer ONLY with a valid JSON object formatted as: "
                        "{'answer': 'YES' or 'NO', 'reason': explain your reason}."
                    )
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    batch_messages.append(messages)

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

            elif prompt_type == "multi_label":
                # ----------------------------------------------------------------
                # Sequential batch search:
                #   For each image, iterate label_batches one-by-one.
                #   Stop as soon as the VLM returns exactly 1 confident label.
                #   Images not yet resolved are batched together each round.
                # ----------------------------------------------------------------
                label_batches = kwargs.get("label_batches", [])
                n_imgs = len(batch_images)

                # Per-image state
                per_img_answer   = [None] * n_imgs  # final label (str) or None
                per_img_reason   = [""] * n_imgs
                per_img_batches  = [0]  * n_imgs     # which label_batch each image is on
                per_img_out_len  = [0]  * n_imgs     # cumulative decoded-text char count
                resolved         = [False] * n_imgs

                for batch_round, label_batch in enumerate(label_batches):
                    # Collect indices of images that still need this batch
                    pending_indices = [
                        i for i in range(n_imgs)
                        if not resolved[i] and per_img_batches[i] == batch_round
                    ]
                    if not pending_indices:
                        continue

                    # Build messages only for pending images
                    pending_msgs = []
                    for i in pending_indices:
                        prompt = (
                            "You are given an image and a list of candidate labels. "
                            "Your task is to identify whether ANY SINGLE label in the list "
                            "clearly and confidently matches the main subject of the image.\n\n"
                            f"Candidates: {', '.join(label_batch)}\n\n"
                            "Rules:\n"
                            "- If you are CONFIDENT that exactly one label matches, return that label.\n"
                            "- If NONE of the candidates fits confidently, return an empty list.\n"
                            "- Do NOT guess. Only answer if you are sure.\n\n"
                            "Return ONLY a JSON object, no extra text:\n"
                            "{\"answer\": [\"<label>\"] or [], \"reason\": \"<brief reason>\"}"
                        )
                        pending_msgs.append([{
                            "role": "user",
                            "content": [
                                {"type": "image", "image": batch_images[i]},
                                {"type": "text", "text": prompt},
                            ],
                        }])

                    texts = [
                        self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                        for m in pending_msgs
                    ]
                    image_inputs, video_inputs = process_vision_info(pending_msgs)
                    inputs = self.processor(
                        text=texts,
                        images=image_inputs,
                        videos=video_inputs,
                        return_tensors="pt",
                        padding=True,
                    ).to(self.device)

                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                            pad_token_id=self.processor.tokenizer.pad_token_id,
                            eos_token_id=None,
                        )

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_texts = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    # --- DEBUG: Save Raw VLM Answers ---
                    debug_dir = "/tmp2/maitanha/vgu/cll_vlm/cll_vlm/ol_cll_logs/multi_label"
                    os.makedirs(debug_dir, exist_ok=True)
                    debug_file = os.path.join(debug_dir, "raw_vlm_answers.jsonl")
                    
                    with open(debug_file, "a", encoding="utf-8") as f_debug:
                        for pos, (img_i, out) in enumerate(zip(pending_indices, output_texts)):
                            debug_entry = {
                                "global_img_idx": start_idx + batch_start_idx + img_i,
                                "round": batch_round,
                                "label_batch": list(label_batch),
                                "raw_vlm_answer": out
                            }
                            f_debug.write(json.dumps(debug_entry, ensure_ascii=False) + "\n")
                    # -----------------------------------

                    batch_label_set = set(label_batch)
                    for pos, (img_i, out) in enumerate(zip(pending_indices, output_texts)):
                        predicted, reason = extract_multi_label_full(out, valid_labels=batch_label_set)

                        # Accumulate output length (chars of decoded text, not padded token count)
                        per_img_out_len[img_i] += len(out)

                        if len(predicted) == 1:
                            # VLM is confident → record and mark resolved
                            per_img_answer[img_i] = predicted[0]
                            per_img_reason[img_i] = reason
                            resolved[img_i] = True
                        else:
                            # No confident answer → advance to next label_batch
                            per_img_batches[img_i] = batch_round + 1

                for i in range(n_imgs):
                    results.append({
                        "img_idx": start_idx + batch_start_idx + i,
                        "true_label": batch_true_labels[i] if batch_true_labels else None,
                        "shuffled_label": batch_shuffled_labels[i] if batch_shuffled_labels else None,
                        "answer": per_img_answer[i],       # single label str, or None
                        "reason": per_img_reason[i],
                        "output_length": per_img_out_len[i],
                    })

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)
        
        return results
    

    def generate_text(self, prompt: str, system_content: str = None) -> str:
        """
        Generate free-form text from Qwen VL using text-only prompt (no image).
        """
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": prompt})

        # Format thành đoạn văn bản cho tokenizer
        formatted_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Không có ảnh hoặc video
        inputs = self.processor(
            text=formatted_text,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=None
            )

        output_ids_trimmed = outputs[:, inputs.input_ids.shape[1]:]
        decoded = self.processor.batch_decode(
            output_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        return decoded.strip()

    def generate_text_batch(self, prompts: List[str], system_content: str = None, batch_size: int = 8) -> List[str]:
        """
        Generate free-form text from Qwen VL using text-only prompts in batches.
        """
        all_results = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating text batch"):
            batch_prompts = prompts[i:i + batch_size]
            batch_messages = []
            
            for prompt in batch_prompts:
                messages = []
                if system_content:
                    messages.append({"role": "system", "content": system_content})
                messages.append({"role": "user", "content": prompt})
                batch_messages.append(messages)
            
            # Format and tokenize batch
            texts = [
                self.processor.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True
                )
                for msg in batch_messages
            ]
            
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=None
                )
            
            output_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]
            
            decoded_batch = self.processor.batch_decode(
                output_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            all_results.extend([d.strip() for d in decoded_batch])
            
        return all_results

    def predict_best_label_batch(self, images, label_option_list, baseprompt=None):
        """
        Batch version of predict_best_label compatible with LLaVA interface.
        Args:
            images (list[PIL.Image]): batch of input images
            label_option_list (list[list[str]]): candidate labels per image
            baseprompt (str): optional custom prompt template
        Returns:
            list[str]: chosen labels for each image
        """
        if len(images) != len(label_option_list):
            raise ValueError("images and label_option_list must have the same length")

        batch_messages = []
        for img, label_options in zip(images, label_option_list):
            if isinstance(label_options, list):
                label_text = ", ".join(label_options)
            else:
                raise ValueError("Each element of label_option_list must be a list of strings")

            # # Randonly shuffle label options to prevent position bias
            # label_options_shuffled = label_options.copy()
            # np.random.shuffle(label_options_shuffled)
            # label_text = ", ".join(label_options_shuffled)

            prompt = baseprompt.format(labels=label_text, label_text=label_text)
            # print(f"[DEBUG] Generated prompt for image:\n{prompt}\n")

            # import pdb; pdb.set_trace()
            # # Use custom baseprompt or default
            # if baseprompt:
            #     prompt = baseprompt.format(labels=label_text, label_text=label_text)
            # else:
            #     prompt = f"Select the closest label from [{label_text}]. Output strictly one label, no explanation or reasoning."

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            batch_messages.append(messages)

        # Generate answers
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

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=None,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Return raw outputs without parsing
        return [text.strip() for text in output_texts]