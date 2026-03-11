import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from typing import List, Dict, Union
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import json
import re

def strip_prompt(text: str) -> str:
    # Loại bỏ các tag phổ biến của các dòng LLaVA khác nhau
    if "[/INST]" in text:
        text = text.split("[/INST]", 1)[1]
    elif "ASSISTANT:" in text:
        text = text.split("ASSISTANT:", 1)[1]
    elif "assistant\n" in text.lower():
        # Một số bản Llama-3 LLaVA dùng format này
        parts = re.split(r'assistant\n', text, flags=re.IGNORECASE)
        text = parts[-1]
    
    return text.strip()

def parse_llava_output(raw: str):
    text = strip_prompt(raw)
    # print(f"[DEBUG] LLaVA output text: {text}")

    # 1. Thử tìm trong định dạng JSON (ưu tiên)
    m = re.search(r'["\']answer["\']\s*:\s*["\'](YES|NO)["\']', text, re.IGNORECASE)
    if m:
        answer = m.group(1).upper()
    else:
        # Fallback 1: Tìm chữ YES/NO đứng cô lập hoặc ở đầu dòng
        m2 = re.search(r'\b(YES|NO)\b', text, re.IGNORECASE)
        if m2:
            answer = m2.group(1).upper()
        else:
            answer = "UNKNOWN"

    # 2. Parse reason
    r = re.search(r'["\']reason["\']\s*:\s*["\'](.*)', text, re.DOTALL)
    if r:
        reason = r.group(1).rstrip('"} \n')
    else:
        # Nếu không có JSON, lấy toàn bộ text làm lý do (nhưng bỏ kết quả YES/NO ở đầu)
        reason = text
    
    # print(f"[DEBUG] Extracted answer: {answer}")

    # 3. Final cleanup
    reason = reason.strip()
    if "explain your reason for choosing" in reason.lower():
        reason = ""

    return answer, reason


def extract_multi_label_full_llava(raw: str, valid_labels: set = None) -> (list, str):
    """
    Extract predicted labels and reason from LLaVA output for multi_label prompt type.
    Strips the echoed prompt first (LLaVA returns the full prompt+answer string),
    then delegates to the same JSON-parsing logic used by the Qwen version.
    Expected JSON format: {'answer': [...], 'reason': '...'}
    Returns (list of valid labels, reason string).
    """
    text = strip_prompt(raw)
    if not text:
        return [], ""

    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)

    # Pass 1: look for a proper {"answer": [...], "reason": ...} object
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


class LLaVAClassifier:
    def __init__(self, model_path="llava-hf/llava-v1.6-mistral-7b-hf", baseprompt=None, device=None, device_map=None):
        self.processor = LlavaNextProcessor.from_pretrained(model_path, use_fast=False)
        self.is_vicuna = "vicuna" in model_path.lower()
        
        # Use device_map="auto" for multi-GPU if device is not specifically set to a single one
        # This respects CUDA_VISIBLE_DEVICES
        if device_map is None:
            if device in [None, "cuda", "auto"]:
                device_map = "auto"
            else:
                device_map = device

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device_map
        )
        
        # Use the actual device assigned to the model (handles multi-GPU device_map)
        self.device = self.model.device
        self.baseprompt = baseprompt

    @classmethod
    def build_model(cls, args):
        model_url = getattr(args, 'model_url', "llava-hf/llava-v1.6-mistral-7b-hf")
        return cls(model_path=model_url, baseprompt=args.prompt)

    def create_prompt(self, label: str, baseprompt: str) -> str:
        # Format the baseprompt with the label
        if baseprompt is None:
            baseprompt = "Does the label '{label}' match this image? Answer with only a single word: YES or NO."
        formatted_prompt = baseprompt.format(label=label)
        return f"[INST]<image>\n{formatted_prompt} Do not repeat the question or provide any explanation.[/INST]"

    def predict(self, images, labels):
        """Predict YES/NO for a batch of images + labels"""
        prompts = [self.create_prompt(lab, self.baseprompt) for lab in labels]
        inputs = self.processor(
            images=images, text=prompts, padding=True, return_tensors="pt"
        ).to(self.model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            pad_token_id=self.processor.tokenizer.eos_token_id
        )

        answers = self.processor.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        processed = []
        for ans in answers:
            # Extract answer from the response
            ans = self._extract_answer(ans)
            print(f"Raw answer: {ans}")
            if "yes" in ans.lower():
                processed.append("YES")
            elif "no" in ans.lower():
                processed.append("NO")
            else:
                # fallback: treat as NO if uncertain
                processed.append("NO")
        return processed
    
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
                    # prompt = (
                    #     "Forget you previous answer. "
                    #     f"You are given an image. Does the label '{label}' correspond to this image?"
                    #         "Answer ONLY with a valid JSON object formatted as:"
                    #         # "Return only a JSON object formatted as: "
                    #         "{'answer': 'YES' or 'NO', 'reason': explain your reason for choosing them}."
                    # )

                    prompt = (
                        f"You are given an image.\n"
                        "First, identify the SINGLE main object that occupies the central visual focus "
                        "or is most salient in the image.\n"
                        "Do NOT consider background, environment, or secondary objects.\n\n"
                        f"Then decide whether the label '{label}' correctly describes that main object.\n"
                        "If the label matches only background or contextual elements, answer NO.\n\n"
                        "Answer ONLY with a valid JSON object formatted as: "
                        "{'answer': 'YES' or 'NO', 'reason': explain your reason for choosing them}."
                    )

                    if self.is_vicuna:
                        prompt = f"USER: <image>\n{prompt} ASSISTANT:"
                    else:
                        prompt = f"[INST] <image>\n{prompt} [/INST]"
                    batch_messages.append(prompt)

                inputs = self.processor(
                    images=batch_images, text=batch_messages, padding=True, return_tensors="pt"
                ).to(self.model.device)
                inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

                answers = self.processor.batch_decode(
                    outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                # print(f"Answers sample: {answers}")
                
                for idx_in_batch, raw in enumerate(answers):
                    # Robust strip for Vicuna if ASSISTANT: format is used
                    if self.is_vicuna and "ASSISTANT:" in raw:
                        raw = raw.split("ASSISTANT:", 1)[1]
                    decision, reason = parse_llava_output(raw)

                    global_idx = start_idx + batch_start_idx + idx_in_batch

                    results.append({
                        "img_idx": global_idx,
                        "shuffled_label": batch_shuffled_labels[idx_in_batch],
                        "true_label": batch_true_labels[idx_in_batch],
                        "answer": decision,
                        "reason": reason,
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
                        "First, identify the SINGLE main object that occupies the central visual focus or is most salient in the image.\n"
                        f"Then decide whether, according to the descriptions above, this image correctly depicts the label '{shuffled_label}'.\n"
                        "Answer ONLY with a valid JSON object formatted as: "
                        "{'answer': 'YES' or 'NO', 'reason': explain your reason}."
                    )
                    if self.is_vicuna:
                        prompt = f"USER: <image>\n{prompt} ASSISTANT:"
                    else:
                        prompt = f"[INST] <image>\n{prompt} [/INST]"
                    batch_messages.append(prompt)

                inputs = self.processor(
                    images=batch_images, text=batch_messages, padding=True, return_tensors="pt"
                ).to(self.model.device)
                inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

                answers = self.processor.batch_decode(
                    outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                for idx_in_batch, raw in enumerate(answers):
                    # Robust strip for Vicuna if ASSISTANT: format is used
                    if self.is_vicuna and "ASSISTANT:" in raw:
                        raw = raw.split("ASSISTANT:", 1)[1]
                    decision, reason = parse_llava_output(raw)
                    global_idx = start_idx + batch_start_idx + idx_in_batch
                    results.append({
                        "img_idx": global_idx,
                        "true_label": batch_true_labels[idx_in_batch],
                        "shuffled_label": batch_shuffled_labels[idx_in_batch],
                        "answer": decision,
                        "reason": [reason] if reason else [],
                    })

            elif prompt_type == "multi_label":
                # ----------------------------------------------------------------
                # Sequential batch search (mirrors QWENClassifier logic):
                #   For each image, iterate label_batches one-by-one.
                #   Stop as soon as the VLM returns exactly 1 confident label.
                # ----------------------------------------------------------------
                label_batches = kwargs.get("label_batches", [])
                n_imgs = len(batch_images)

                # Per-image state
                per_img_answer  = [None] * n_imgs
                per_img_reason  = [""] * n_imgs
                per_img_batches = [0] * n_imgs   # which label_batch each image is on
                per_img_out_len = [0] * n_imgs   # cumulative decoded-text char count
                resolved        = [False] * n_imgs

                for batch_round, label_batch in enumerate(label_batches):
                    # Collect indices of images that still need this round
                    pending_indices = [
                        i for i in range(n_imgs)
                        if not resolved[i] and per_img_batches[i] == batch_round
                    ]
                    if not pending_indices:
                        continue

                    pending_prompts = []
                    pending_imgs    = []
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
                            '{"answer": ["<label>"] or [], "reason": "<brief reason>"}'
                        )
                        if self.is_vicuna:
                            prompt = f"USER: <image>\n{prompt} ASSISTANT:"
                        else:
                            prompt = f"[INST] <image>\n{prompt} [/INST]"
                        pending_prompts.append(prompt)
                        pending_imgs.append(batch_images[i])

                    inputs = self.processor(
                        images=pending_imgs, text=pending_prompts,
                        padding=True, return_tensors="pt"
                    ).to(self.model.device)
                    inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )

                    answers = self.processor.batch_decode(
                        outputs, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )

                    # --- DEBUG: Save Raw VLM Answers ---
                    debug_dir = "/tmp2/maitanha/vgu/cll_vlm/cll_vlm/ol_cll_logs/multi_label_llava"
                    os.makedirs(debug_dir, exist_ok=True)
                    debug_file = os.path.join(debug_dir, "raw_vlm_answers.jsonl")
                    with open(debug_file, "a", encoding="utf-8") as f_debug:
                        for img_i, out in zip(pending_indices, answers):
                            f_debug.write(json.dumps({
                                "global_img_idx": start_idx + batch_start_idx + img_i,
                                "round": batch_round,
                                "label_batch": list(label_batch),
                                "raw_vlm_answer": strip_prompt(out)
                            }, ensure_ascii=False) + "\n")
                    # -----------------------------------

                    batch_label_set = set(label_batch)
                    for img_i, out in zip(pending_indices, answers):
                        cleaned = strip_prompt(out)
                        per_img_out_len[img_i] += len(cleaned)
                        predicted, reason = extract_multi_label_full_llava(
                            out, valid_labels=batch_label_set
                        )

                        if len(predicted) == 1:
                            per_img_answer[img_i] = predicted[0]
                            per_img_reason[img_i] = reason
                            resolved[img_i] = True
                        else:
                            per_img_batches[img_i] = batch_round + 1

                for i in range(n_imgs):
                    results.append({
                        "img_idx": start_idx + batch_start_idx + i,
                        "true_label":     batch_true_labels[i] if batch_true_labels else None,
                        "shuffled_label": batch_shuffled_labels[i] if batch_shuffled_labels else None,
                        "answer":         per_img_answer[i],
                        "reason":         per_img_reason[i],
                        "output_length":  per_img_out_len[i],
                    })

            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=4)

        return results
    
    def _extract_answer(self, response: str) -> str:
        """Extract just the answer from the model response"""
        # Split by common separators and take the last part
        if "[/INST]" in response:
            answer = response.split("[/INST]")[-1].strip()
        elif "ASSISTANT:" in response:
            answer = response.split("ASSISTANT:")[-1].strip()
        else:
            answer = response.strip()
        
        # # Remove any repeated question text
        # if "does the label" in answer.lower():
        #     # If it repeated the question, try to find the actual answer
        #     lines = answer.split('\n')
        #     for line in lines:
        #         line = line.strip().lower()
        #         if line in ['yes', 'no'] or line.startswith('yes') or line.startswith('no'):
        #             return line
        
        # Extract first word that looks like yes/no
        words = answer.split()
        for word in words:
            clean_word = word.lower().strip('.,!?:')
            if clean_word in ['yes', 'no']:
                return clean_word
        
        # If still no clear answer, return the cleaned response
        return answer

    def predict_best_label(self, images, label_options):
        """
        Given one image and a list of label options (e.g., 4 strings),
        ask LLaVA which label best matches the image.
        Returns the single chosen label.
        """
        if isinstance(label_options, list):
            label_text = ", ".join(label_options)
        else:
            raise ValueError("label_options must be a list of strings")

        prompt = f"Which of the following labels best describes this image? Answer the question with a single word from [{label_text}]."

        prompt = f"[INST]<image>\n{prompt}[/INST]"
        inputs = self.processor(
            images=images, text=[prompt], padding=True, return_tensors="pt"
        ).to(self.model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=15,
            pad_token_id=self.processor.tokenizer.eos_token_id
        )

        answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        answer = self._extract_answer(answer)

        # pick the first label name that appears in response
        for lab in label_options:
            if lab.lower() in answer.lower():
                return lab
        # fallback: return raw answer
        return answer

    def predict_best_label_batch(self, images, label_option_list, baseprompt=None):
        """
        Batch version of predict_best_label.
        Args:
            images (list[PIL.Image]): batch of input images
            label_option_list (list[list[str]]): candidate labels per image
            baseprompt (str): optional custom prompt template from CLI
                e.g. "<image>\nWhich of the following labels best describes this image: {labels}?"
        Returns:
            list[str]: chosen labels for each image
        """
        if len(images) != len(label_option_list):
            raise ValueError("images and label_option_list must have the same length")

        # ========== BUILD PROMPTS ==========
        prompts = []
        for label_options in label_option_list:
            if isinstance(label_options, list):
                label_text = ", ".join(label_options)
            else:
                raise ValueError("Each element of label_option_list must be a list of strings")

            # choose baseprompt (from arg or model default)
            prompt_template = baseprompt or self.baseprompt
            if prompt_template:
                # allow {labels} placeholder in prompt
                prompt = prompt_template.format(labels=label_text, label_text=label_text)
            else:
                prompt = f"Which of the following labels best describes this image? Answer the question with a single word from [{label_text}]."

            # ensure <image> token exists
            if "<image>" not in prompt:
                prompt = f"<image>\n{prompt}"

            # wrap in instruction tags if not already formatted
            if "[INST]" not in prompt:
                prompt = f"[INST]{prompt}[/INST]"

            prompts.append(prompt)

        # ========== PROCESS BATCH ==========
        inputs = self.processor(
            images=images, text=prompts, padding=True, return_tensors="pt"
        ).to(self.model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=15,
            pad_token_id=self.processor.tokenizer.eos_token_id
        )

        # ========== DECODE & PARSE ANSWERS ==========
        answers = self.processor.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        results = []
        for ans, label_options in zip(answers, label_option_list):
            ans_clean = self._extract_answer(ans)

            # find first candidate label mentioned in answer
            chosen = None
            for lab in label_options:
                if lab.lower() in ans_clean.lower():
                    chosen = lab
                    break

            # fallback — nếu model trả câu khác (VD: “cat” khi không có cat)
            if chosen is None:
                chosen = ans_clean.strip()

            results.append(chosen)

        return results
