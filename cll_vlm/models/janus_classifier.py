import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.janus import JanusForConditionalGeneration, JanusProcessor
import numpy as np
from PIL import Image
import os
import json
import re
from collections import Counter
from tqdm import tqdm
from typing import List, Dict, Union

def extract_all_reasons(raw: str):
    """
    Extract all JSON objects from Janus output.
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

def load_janus(model_path: str, device: str = None):
    try:
        from transformers import AutoTokenizer, AutoImageProcessor
        from transformers.models.janus import JanusForConditionalGeneration, JanusProcessor, JanusImageProcessor
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, legacy=False)
        
        # Standard special tokens for Janus Pro. 
        # Janus-Pro-1B uses these strings. We must make sure they are strings for the processor.
        # But we only add those that are REALLY missing from the vocabulary to avoid ID shifts.
        special_tokens_to_add = []
        for t in ["<image_placeholder>", "<|vision_start|>", "<|vision_end|>"]:
            if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id or tokenizer.convert_tokens_to_ids(t) is None:
                # If it's None or unk, and it's not a standard token, we might need to add it.
                # However, for Janus-Pro, <image_placeholder> should be 100581.
                pass
        
        # Manually ensure attributes are set for the processor as strings
        tokenizer.image_token = "<image_placeholder>"
        # Janus-Pro-1B often doesn't use boi/eoi in the same way as older Janus. 
        # If they are not in vocab, setting them to empty string prevents count errors in Processor.
        tokenizer.boi_token = "<|vision_start|>" if tokenizer.convert_tokens_to_ids("<|vision_start|>") is not None else ""
        tokenizer.eoi_token = "<|vision_end|>" if tokenizer.convert_tokens_to_ids("<|vision_end|>") is not None else ""
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        try:
            image_processor = JanusImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
            
        processor = JanusProcessor(image_processor=image_processor, tokenizer=tokenizer)
        
        # Use device_map="auto" for multi-GPU if device is not specifically set to a single one
        # This respects CUDA_VISIBLE_DEVICES
        if device in [None, "cuda", "auto"]:
            device_map = "auto"
        else:
            device_map = device

        # Use bfloat16 for DeepSeek models if on CUDA
        dtype = torch.bfloat16 if (device_map == "auto" or "cuda" in str(device_map)) else torch.float32
        
        model = JanusForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map
        )
            
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Janus model or processor from {model_path}.\n{e}"
        )
    return processor, model

class JanusClassifier:
    def __init__(self, model_path="deepseek-ai/Janus-Pro-1B", device=None):
        self.processor, self.model = load_janus(model_path, device)
        # Use the actual device assigned to the model (handles multi-GPU device_map)
        self.device = self.model.device

    def generate_batch_results(self, data, shuffled_label_indices, true_label_indices, fine_classes, prompt_type, output_path, batch_size, start_idx=0, label_description_path=None):
        total_images = len(data)
        num_batches = (total_images + batch_size - 1) // batch_size
        results = []

        for batch_idx in tqdm(range(num_batches)):
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = min((batch_idx + 1) * batch_size, total_images)

            batch_images_raw = data[batch_start_idx:batch_end_idx]

            batch_shuffled_labels = [
                fine_classes[idx] for idx in shuffled_label_indices[batch_start_idx:batch_end_idx]
            ]
            batch_true_labels = [
                fine_classes[idx] for idx in true_label_indices[batch_start_idx:batch_end_idx]
            ]
            
            # Process Images to PIL
            pil_images = []
            if isinstance(batch_images_raw, torch.Tensor):
                # Handle Torch Tensors
                if batch_images_raw.dim() == 4: # Batch of images
                    for i in range(batch_images_raw.shape[0]):
                        img_tensor = batch_images_raw[i, :]
                        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0) # C H W -> H W C
                        if img_np.max() <= 1.0:
                             img_np = (img_np * 255).astype("uint8")
                        else:
                             img_np = img_np.astype(np.uint8)
                        pil_images.append(Image.fromarray(img_np))
                elif batch_images_raw.dim() == 3: # Single image tensor, but wrapped in list/loop logic should handle batch
                     # Assuming list of 3D tensors passed as batch_images_raw if not handled above, 
                     # but here 'data' slice is usually a list or a concatenated tensor.
                     # If data is a list of tensors:
                     pass # handled below if data is list
            
            if not pil_images: # If not processed from tensor batch above (e.g. data is a list of tensors or list of arrays)
                 for item in batch_images_raw:
                    if isinstance(item, torch.Tensor):
                         img_np = item.cpu().numpy().transpose(1, 2, 0)
                         if img_np.max() <= 1.0:
                             img_np = (img_np * 255).astype("uint8")
                         else:
                             img_np = img_np.astype(np.uint8)
                         pil_images.append(Image.fromarray(img_np))
                    elif isinstance(item, np.ndarray):
                        pil_images.append(Image.fromarray(item))
                    elif isinstance(item, Image.Image):
                        pil_images.append(item)
            
            batch_images = pil_images

            # Prepare prompts and inputs
            batch_conversations = []
            batch_pil_imgs = [] # Flattened list of images for processor if needed, but Janus usually takes list of images per prompt
            
            # Note: JanusProcessor typically processes text and images. 
            # We need to construct the prompt string formatted for Janus.
            # Janus often uses: User: <image_placeholder>\n{text}<eos_token>Assistant:
            
            for img, label in zip(batch_images, batch_shuffled_labels):
                
                # Construct Prompt
                if prompt_type == "binary":
                    prompt_text = (
                        f"You are given an image.\n"
                        "First, identify the SINGLE main object that occupies the central visual focus "
                        "or is most salient in the image.\n"
                        "Do NOT consider background, environment, or secondary objects.\n\n"
                        f"Then decide whether the label '{label}' correctly describes that main object.\n"
                        "If the label matches only background or contextual elements, answer NO.\n\n"
                        "Answer ONLY with a valid JSON object formatted as: "
                        "{'answer': 'YES' or 'NO', 'reason': explain your reason for choosing them}."
                    )
                elif prompt_type == "label_description":
                    if label_description_path is None:
                        raise ValueError("label_description_path is required when prompt_type is 'label_description'")
                    # Load descriptions (optimization: load once outside loop if possible, but keeping consistent with request)
                    # For efficiency, we should load outside. But sticking to structure.
                    # Warning: Loading file inside loop is inefficient. 
                    # Assuming caller handles path correctly. Ideally load once in init or passed as dict.
                    # QwenClassifier loaded it inside the loop condition block but outside the batch item loop.
                    # We'll mimic what QwenClassifier likely intended or rewrite for better flow.
                    
                    if not hasattr(self, 'label_descriptions_cache'):
                         with open(label_description_path, "r", encoding="utf-8") as f:
                            self.label_descriptions_cache = json.load(f)
                    
                    label_descriptions = self.label_descriptions_cache
                    
                    desc = label_descriptions.get(label)
                    if desc is None:
                        key_alt = label.replace("_", " ").strip().lower()
                        desc = label_descriptions.get(key_alt)
                    if desc is None:
                        desc = {"visual": [], "context": []}

                    visual_list = desc.get("visual", [])
                    context_list = desc.get("context", [])
                    visual_text = "\n".join(f"- {s}" for s in visual_list) if visual_list else "(No visual description)"
                    context_text = "\n".join(f"- {s}" for s in context_list) if context_list else "(No context description)"
                    
                    prompt_text = (
                        f"You are given an image and the following descriptions for the label '{label}'.\n\n"
                        "Visual descriptions:\n" + visual_text + "\n\n"
                        "Context descriptions:\n" + context_text + "\n\n"
                        f"According to the descriptions above, does the image correctly depicts the label '{label}'.\n"
                        "Answer ONLY with a valid JSON object formatted as: "
                        "{'answer': 'YES' or 'NO', 'reason': explain your reason}."
                    )
                else:
                    prompt_text = f"Does the image contain {label}? Answer YES or NO."

                
                conversation = [
                    {
                        "role": "<|User|>", # Janus specific role
                        "content": f"<image_placeholder>\n{prompt_text}",
                        "images": [img],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                batch_conversations.append(conversation)

            # Batched processing might be tricky with custom conversation format depending on implementation.
            # We will process one by one in the batch if the processor doesn't support list of conversations well for Janus custom format
            # Or use apply_chat_template if available.
            # The standard Janus usage often involves preparing inputs manually or via specific processor calls.
            # Let's adapt to the JanusProcessor usage seen in 'processing_janus.py' or standard usage.
            
            # Standard Janus often uses:
            # inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(device)
            
            # Prepare inputs
            texts = []
            images_flat = []
            
            for conv in batch_conversations:
                # Assuming conversation structure map to text prompt
                # User: ...
                # Assistant: 
                user_content = conv[0]["content"]
                # JanusProcessor usually expects <image_placeholder> in text matching image count.
                texts.append(user_content)
                images_flat.append(conv[0]["images"][0])
            
            inputs = self.processor(
                text=texts,
                images=images_flat,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Decode
            # We need to trim input tokens from output
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_texts = self.processor.tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
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
