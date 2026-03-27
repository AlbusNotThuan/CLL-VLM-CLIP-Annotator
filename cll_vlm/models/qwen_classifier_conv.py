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


def extract_multi_label_full(raw: str) -> tuple:
    """
    Extract the first valid predicted label(s) and reason from Qwen output.
    """
    if not raw:
        return [], ""

    raw_text = raw.strip()

    # Try parsing the structure {"answer": [...], "reason": "..."} using regex first for robustness
    # This also helps if the JSON is surrounded by other text
    
    # Try finding JSON block
    start = raw_text.find('{')
    if start != -1:
        depth = 0
        end = -1
        for i in range(start, len(raw_text)):
            if raw_text[i] == '{':
                depth += 1
            elif raw_text[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            try:
                obj = json.loads(raw_text[start:end+1])
                if "answer" in obj:
                    ans = obj["answer"]
                    reason = obj.get("reason", "").strip()
                    if isinstance(ans, list):
                        labels = [str(l).strip() for l in ans]
                    elif isinstance(ans, str):
                        labels = [ans.strip()]
                    else:
                        labels = []
                    # Check if 'NO' is the answer
                    labels = [l for l in labels if l]
                    if len(labels) == 1 and labels[0].upper() == "NO":
                        return [], reason
                    elif "NO" in [l.upper() for l in labels]:
                        labels = [l for l in labels if l.upper() != "NO"]
                    return labels, reason
            except:
                pass

    # Regex fallback. We make the closing quote optional because it could be truncated!
    reason_m = re.search(r'"reason"\s*:\s*"([^"]*)', raw_text)
    reason = reason_m.group(1).strip() if reason_m else ""

    ans_list_m = re.search(r'"answer"\s*:\s*\[([^\]]*)\]', raw_text)
    if ans_list_m:
        raw_items = ans_list_m.group(1)
        labels = [l.strip().strip('"').strip("'") for l in raw_items.split(',')]
        labels = [l for l in labels if l]
        if len(labels) == 1 and labels[0].upper() == "NO":
            return [], reason
        labels = [l for l in labels if l.upper() != "NO"]
        if labels:
            return labels, reason

    ans_str_m = re.search(r'"answer"\s*:\s*"([^"]+)"', raw_text)
    if ans_str_m:
        val = ans_str_m.group(1).strip()
        if val.upper() == "NO":
            return [], reason
        return [val], reason

    # If all fails, treat the whole raw text as reason, returning empty array for answers
    return [], raw_text


def load_qwen(model_path: str, device: str = None):
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        # Set left padding for decoder-only models (required for correct batch generation)
        if hasattr(processor, 'tokenizer'):
            processor.tokenizer.padding_side = 'left'
        elif hasattr(processor, 'padding_side'):
            processor.padding_side = 'left'
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
        self.temperature = 1.0  # Default temperature, can be overridden in generation kwargs

    def set_temperature(self, temp: float):
        """Set the temperature for generation (if applicable)."""
        self.temperature = temp

    def generate_batch_results(self, data, shuffled_label_indices, true_label_indices, fine_classes, prompt_type, output_path, batch_size, start_idx=0, label_description_path=None, **kwargs):
        total_images = len(data)
        num_batches = (total_images + batch_size - 1) // batch_size
        results = []

        # Determine whether to collect raw VLM answers (only in test mode)
        raw_output_path = kwargs.get("raw_output_path")
        save_raw = bool(raw_output_path)
        raw_answers_by_img_idx = {} if save_raw else None

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


            elif prompt_type == "multi_label":
                label_batches = kwargs.get("label_batches", [])
                topk          = kwargs.get("topk", 1)
                n_imgs        = len(batch_images)

                # Per-image state: accumulate matched labels / reasons across all batches
                per_img_answers    = [[] for _ in range(n_imgs)]   # list[str]
                per_img_reasons    = [[] for _ in range(n_imgs)]   # list[str]
                per_img_conversations = [[] for _ in range(n_imgs)] # list[list[dict]] for conversation history
                # Only allocate raw rounds buffer when saving is needed
                per_img_raw_rounds = [[] for _ in range(n_imgs)] if save_raw else None

                for batch_round, label_batch in enumerate(label_batches):
                    
                    # Build one message per image
                    batch_msgs = []

                    for i, img in enumerate(batch_images):    
                        if topk == 1:
                            prompt = (
                                "You are given an image. "
                                "Examine the image carefully and identify which objects from the candidate list are present.\n"
                                f"Candidates: ({', '.join(label_batch)}).\n"
                                "From this list, choose only ONE label that is most likely present in the image. "
                                "Do not include any label that is not in the candidate list. "
                                "If none of the candidates are present, return \"NO\".\n"
                                "Return ONLY a JSON object and nothing else, formatted as: "
                                "{\"answer\": [one label from the given candidate list or \"NO\"], \"reason\": \"short explanation\"}"
                            )
                        else:
                            # prompt topk
                            prompt = (
                                "You are given an image. "
                                "Examine the image carefully and identify which objects from the candidate list are present.\n"
                                f"Candidates: ({', '.join(label_batch)}).\n"
                                f"From this list, choose up to {topk} label(s) that are likely present in the image. "
                                "Do not include any label that is not in the candidate list. "
                                "If none of the candidates are present, return an empty list.\n"
                                "Return ONLY a JSON object and nothing else, formatted as: "
                                "{\"answer\": [...], \"reason\": \"...\"}"
                            )
                        
                        if batch_round == 0:
                            user_msg = {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": img},
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        else:
                            user_msg = {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        
                        per_img_conversations[i].append(user_msg)
                        batch_msgs.append(list(per_img_conversations[i]))

                    texts = [
                        self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                        for m in batch_msgs
                    ]
                    image_inputs, video_inputs = process_vision_info(batch_msgs)
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
                            max_new_tokens=64,
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

                    batch_label_set = set(label_batch)
                    for img_i, out in enumerate(output_texts):
                        # Add assistant response to conversation history
                        assistant_msg = {
                            "role": "assistant",
                            "content": out
                        }
                        per_img_conversations[img_i].append(assistant_msg)

                        if save_raw:
                            per_img_raw_rounds[img_i].append({
                                "round": batch_round,
                                "label_batch": list(label_batch),
                                "raw_vlm_answer": out,
                            })
                        predicted, reason = extract_multi_label_full(out)
                        if predicted:
                            per_img_answers[img_i].extend(predicted)
                        if reason:
                            per_img_reasons[img_i].append(reason)

                # FINAL PASS: if an image has >1 candidate label, ask VLM again.
                per_img_final_answers = [list(per_img_answers[i]) for i in range(n_imgs)]
                per_img_final_reasons = ["" for _ in range(n_imgs)]
                second_pass_indices = []
                second_pass_msgs = []
                for i in range(n_imgs):
                    if len(per_img_answers[i]) > 1:
                        second_pass_indices.append(i)
                        cands = per_img_answers[i]
                        if topk == 1:
                            prompt = (
                                "You are given an image. "
                                "Examine the image carefully and identify which objects from the candidate list are present.\n"
                                f"Candidates: ({', '.join(cands)}).\n"
                                "From this list, choose only ONE label that is most likely present in the image. "
                                "Do not include any label that is not in the candidate list. "
                                "If none of the candidates are present, return an empty list.\n"
                                "Return ONLY a JSON object and nothing else, formatted as: "
                                "{\"answer\": [...], \"reason\": \"...\"}"
                            )
                        else:
                            prompt = (
                                "You are given an image. "
                                "Examine the image carefully and identify which objects from the candidate list are present.\n"
                                f"Candidates: ({', '.join(cands)}).\n"
                                f"From this list, choose up to {topk} label(s) that are likely present in the image. "
                                "Do not include any label that is not in the candidate list. "
                                "If none of the candidates are present, return an empty list.\n"
                                "Return ONLY a JSON object and nothing else, formatted as: "
                                "{\"answer\": [...], \"reason\": \"...\"}"
                            )
                        second_pass_msgs.append([{
                            "role": "user",
                            "content": [
                                {"type": "image", "image": batch_images[i]},
                                {"type": "text", "text": prompt},
                            ],
                        }])
                
                if second_pass_msgs:
                    texts = [
                        self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                        for m in second_pass_msgs
                    ]
                    image_inputs, video_inputs = process_vision_info(second_pass_msgs)
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
                            max_new_tokens=64,
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

                    for out_idx, img_i in enumerate(second_pass_indices):
                        out = output_texts[out_idx]
                        if save_raw:
                            per_img_raw_rounds[img_i].append({
                                "round": "final_pass",
                                "label_batch": list(per_img_answers[img_i]),
                                "raw_vlm_answer": out,
                            })
                        predicted, reason = extract_multi_label_full(out)
                        per_img_final_answers[img_i] = predicted
                        if reason:
                            per_img_final_reasons[img_i] = reason

                for i in range(n_imgs):
                    img_idx = start_idx + batch_start_idx + i
                    results.append({
                        "img_idx": img_idx,
                        "true_label": batch_true_labels[i] if batch_true_labels else None,
                        "shuffled_label": batch_shuffled_labels[i] if batch_shuffled_labels else None,
                        "candidate_answer": per_img_answers[i],
                        "candidate_reason": per_img_reasons[i],
                        "answer": per_img_final_answers[i],
                        "answer_reason": per_img_final_reasons[i],
                    })

                    # [DEBUG]
                    if save_raw:
                        raw_answers_by_img_idx[str(img_idx)] = {
                            "img_idx": img_idx,
                            "true_label": batch_true_labels[i] if batch_true_labels else None,
                            "shuffled_label": batch_shuffled_labels[i] if batch_shuffled_labels else None,
                            "candidate_answer": per_img_answers[i],
                            "answer": per_img_final_answers[i],
                            "rounds": per_img_raw_rounds[i],
                        }

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)

        # Save raw VLM answers to a separate JSON file (only in test mode)
        if save_raw and raw_answers_by_img_idx:
            os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
            with open(raw_output_path, "w", encoding="utf-8") as f:
                json.dump(raw_answers_by_img_idx, f, indent=4, ensure_ascii=False)
        
        return results
    

    def generate_text(self, prompt: str, system_content: str = None) -> str:
        """
        Generate free-form text from Qwen VL using text-only prompt (no image).
        """
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": prompt})

        formatted_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

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

        # print("[DEBUG] Temperature for generation:", self.temperature)

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
                do_sample=False if self.temperature == 1.0 else True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=None,
                temperature=self.temperature,
                # top_p=0.95,
                # top_k=50,
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