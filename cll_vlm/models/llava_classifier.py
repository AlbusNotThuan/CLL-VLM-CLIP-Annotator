import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from typing import List, Dict, Union
from PIL import Image

class LLaVAClassifier:
    def __init__(self, model_path="llava-hf/llava-v1.6-mistral-7b-hf", baseprompt=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = LlavaNextProcessor.from_pretrained(model_path, use_fast=True)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=None
        ).to(self.device)
        self.baseprompt = baseprompt

    @classmethod
    def build_model(cls, args):
        return cls(model_path="llava-hf/llava-v1.6-mistral-7b-hf", baseprompt=args.prompt)

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
            max_new_tokens=10,
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
