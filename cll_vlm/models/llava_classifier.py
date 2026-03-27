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


def extract_multi_label_full_llava(raw: str, valid_labels: set = None) -> tuple:
    """
    Extract the first valid predicted label(s) and reason from LLaVA output.

    Handles:
      - {"answer": [...], "reason": "..."}
      - {"answer": "cattle", "reason": "..."}
      - {"answer": "NO", ...} -> []
      - Bare "NO" -> []
      - Repeated prompt/assistant echoes
      - Truncated JSON fallback
    """
    text = strip_prompt(raw)
    if not text:
        return [], ""

    stripped = text.strip()
    if stripped.upper() == "NO":
        return [], ""

    valid_lower = None
    if valid_labels is not None:
        valid_lower = {v.lower(): v for v in valid_labels}

    segments = re.split(r'\bassistant\b', text, flags=re.IGNORECASE)

    def normalize_labels(labels):
        if valid_lower is None:
            return labels
        out = []
        for l in labels:
            key = l.lower()
            if key in valid_lower:
                out.append(valid_lower[key])
        return out

    def _try_parse_segment(seg: str):
        start = seg.find('{')
        if start == -1:
            return None

        depth = 0
        end = -1
        for i in range(start, len(seg)):
            if seg[i] == '{':
                depth += 1
            elif seg[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break

        obj = None
        if end != -1:
            try:
                obj = json.loads(seg[start:end + 1])
            except json.JSONDecodeError:
                pass

        if obj is not None:
            if "answer" not in obj:
                if "reason" in obj and isinstance(obj["reason"], str):
                    return [], obj["reason"].strip()
                return None

            answer_val = obj["answer"]
            reason = obj.get("reason", "")

            if isinstance(answer_val, list):
                labels = [
                    str(x).strip() for x in answer_val
                    if str(x).strip().upper() != "NO"
                ]
                labels = normalize_labels(labels)
                return labels, reason

            if isinstance(answer_val, str):
                val = answer_val.strip()
                if val.upper() == "NO":
                    return [], reason
                labels = normalize_labels([val])
                return labels, reason

            return None

        partial = seg[start:]

        m_list = re.search(r'"answer"\s*:\s*\[([^\]]*)\]', partial)
        if m_list:
            raw_items = m_list.group(1)
            labels = [
                x.strip().strip('"').strip("'")
                for x in raw_items.split(',')
                if x.strip().strip('"').strip("'").upper() not in ("", "NO")
            ]
            labels = normalize_labels(labels)
            if labels:
                return labels, ""

        m_str = re.search(r'"answer"\s*:\s*"([^"]+)"', partial)
        if m_str:
            val = m_str.group(1).strip()
            if val.upper() != "NO":
                labels = normalize_labels([val])
                if labels:
                    return labels, ""

        return None

    for seg in segments:
        result = _try_parse_segment(seg)
        if result is not None:
            return result

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

            elif prompt_type == "multi_label":
                label_batches = kwargs.get("label_batches", [])
                topk = kwargs.get("topk", 1)
                n_imgs = len(batch_images)

                raw_output_path = kwargs.get("raw_output_path")
                save_raw = bool(raw_output_path)
                raw_answers_by_img_idx = {} if save_raw else None

                # Per-image state
                per_img_answers = [[] for _ in range(n_imgs)]   # all matched labels across rounds
                per_img_reasons = [[] for _ in range(n_imgs)]   # reasons across rounds
                per_img_raw_rounds = [[] for _ in range(n_imgs)] if save_raw else None

                for batch_round, label_batch in enumerate(label_batches):
                    batch_messages = []

                    for img in batch_images:
                        if topk == 1:
                            prompt = (
                                "You are given an image. "
                                "Examine the image carefully and identify which objects from the candidate list are present.\n"
                                f"Candidates: ({', '.join(label_batch)}).\n"
                                "From this list, choose only ONE label that is most likely present in the image. "
                                "Do not include any label that is not in the candidate list. "
                                "If you think none of the candidates are present, reply with exactly \"NO\".\n"
                                "Provide a short reason for your answer.\n"
                                "Before you make the final response, carefully review if your answer ONLY contains labels in the candidates. "
                                "Your answer should be ONLY a JSON dict and nothing else, formatted as: "
                                "{\"answer\": \"your chosen label\" or \"NO\", \"reason\": \"short explanation\"}\n"
                                "Please don't reply in other formats."
                            )
                        else:
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

                        if self.is_vicuna:
                            prompt = f"USER: <image>\n{prompt} ASSISTANT:"
                        else:
                            prompt = f"[INST] <image>\n{prompt} [/INST]"

                        batch_messages.append(prompt)

                    inputs = self.processor(
                        images=batch_images,
                        text=batch_messages,
                        padding=True,
                        return_tensors="pt"
                    ).to(self.model.device)
                    inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )

                    answers = self.processor.batch_decode(
                        outputs,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )

                    batch_label_set = set(label_batch)

                    for img_i, out in enumerate(answers):
                        cleaned_out = strip_prompt(out)

                        if save_raw:
                            per_img_raw_rounds[img_i].append({
                                "round": batch_round,
                                "label_batch": list(label_batch),
                                "raw_vlm_answer": cleaned_out,
                            })

                        predicted, reason = extract_multi_label_full_llava(
                            cleaned_out,
                            valid_labels=batch_label_set
                        )

                        if predicted:
                            per_img_answers[img_i].extend(predicted)
                        if reason:
                            per_img_reasons[img_i].append(reason)

                # FINAL PASS: if an image has >1 candidate label, ask again
                per_img_final_answers = [list(per_img_answers[i]) for i in range(n_imgs)]
                second_pass_indices = []
                second_pass_prompts = []
                second_pass_images = []

                for i in range(n_imgs):
                    # deduplicate while preserving order
                    seen = set()
                    dedup_candidates = []
                    for lab in per_img_answers[i]:
                        if lab not in seen:
                            seen.add(lab)
                            dedup_candidates.append(lab)

                    per_img_answers[i] = dedup_candidates
                    per_img_final_answers[i] = list(dedup_candidates)

                    if len(dedup_candidates) > 1:
                        second_pass_indices.append(i)
                        second_pass_images.append(batch_images[i])

                        prompt = (
                            "You are given an image. "
                            "Examine the image carefully and identify which object from the candidate list is present.\n"
                            f"Candidates: ({', '.join(dedup_candidates)}).\n"
                            "From this list, choose only ONE label that is most likely present in the image. "
                            "Do not include any label that is not in the candidate list. "
                            "If you think none of the candidates are present, reply with exactly \"NO\".\n"
                            "Provide a short reason for your answer.\n"
                            "Before you make the final response, carefully review if your answer ONLY contains a label in the candidates.\n"
                            "Your answer should be ONLY a JSON dict and nothing else, formatted as: "
                            "{\"answer\": \"your chosen label\" or \"NO\", \"reason\": \"short explanation\"}\n"
                            "Please don't reply in other formats."
                        )

                        if self.is_vicuna:
                            prompt = f"USER: <image>\n{prompt} ASSISTANT:"
                        else:
                            prompt = f"[INST] <image>\n{prompt} [/INST]"

                        second_pass_prompts.append(prompt)

                if second_pass_prompts:
                    inputs = self.processor(
                        images=second_pass_images,
                        text=second_pass_prompts,
                        padding=True,
                        return_tensors="pt"
                    ).to(self.model.device)
                    inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )

                    answers = self.processor.batch_decode(
                        outputs,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )

                    for out_idx, img_i in enumerate(second_pass_indices):
                        out = strip_prompt(answers[out_idx])

                        if save_raw:
                            per_img_raw_rounds[img_i].append({
                                "round": "final_pass",
                                "label_batch": list(per_img_answers[img_i]),
                                "raw_vlm_answer": out,
                            })

                        predicted, reason = extract_multi_label_full_llava(
                            out,
                            valid_labels=set(per_img_answers[img_i])
                        )

                        per_img_final_answers[img_i] = predicted
                        if reason:
                            per_img_reasons[img_i].append(reason)

                for i in range(n_imgs):
                    img_idx = start_idx + batch_start_idx + i

                    results.append({
                        "img_idx": img_idx,
                        "true_label": batch_true_labels[i] if batch_true_labels else None,
                        "shuffled_label": batch_shuffled_labels[i] if batch_shuffled_labels else None,
                        "candidate_answer": per_img_answers[i],
                        "answer": per_img_final_answers[i],
                        "reason": per_img_reasons[i],
                    })

                    if save_raw:
                        raw_answers_by_img_idx[str(img_idx)] = {
                            "img_idx": img_idx,
                            "true_label": batch_true_labels[i] if batch_true_labels else None,
                            "shuffled_label": batch_shuffled_labels[i] if batch_shuffled_labels else None,
                            "candidate_answer": per_img_answers[i],
                            "answer": per_img_final_answers[i],
                            "rounds": per_img_raw_rounds[i],
                        }

                if save_raw and raw_answers_by_img_idx:
                    os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
                    with open(raw_output_path, "w", encoding="utf-8") as f:
                        json.dump(raw_answers_by_img_idx, f, indent=4, ensure_ascii=False)

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
