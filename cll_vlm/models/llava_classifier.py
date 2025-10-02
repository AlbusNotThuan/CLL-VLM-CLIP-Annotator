# llava_classifier.py
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


class LLaVAClassifier:
    def __init__(self, model_path="llava-hf/llava-v1.6-mistral-7b-hf"):
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

    @classmethod
    def build_model(cls, args):
        return cls(args.model_path)

    def create_prompt(self, label: str) -> str:
        return f"[INST] <image>\nQuestion: Does the label '{label}' match this image? Answer only YES or NO. [/INST]"

    def predict(self, images, labels):
        """Predict YES/NO for a batch of images + labels"""
        prompts = [self.create_prompt(lab) for lab in labels]
        inputs = self.processor(
            images=images, text=prompts, padding=True, return_tensors="pt"
        ).to(self.model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=10,
            pad_token_id=self.processor.tokenizer.eos_token_id
        )

        answers = self.processor.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        processed = []
        for ans in answers:
            # Take last part after [/INST], clean text
            ans = ans.split("[/INST]")[-1].strip().lower()
            if "yes" in ans:
                processed.append("YES")
            elif "no" in ans:
                processed.append("NO")
            else:
                # fallback: treat as NO if uncertain
                processed.append("NO")
        return processed
