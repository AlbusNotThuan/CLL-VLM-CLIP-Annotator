from typing import List, Sequence
import numpy as np
import torch
from PIL import Image
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


try:
    from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
    from transformers import AutoModelForCausalLM
    HAS_DEEPSEEK_VL2 = True
except ImportError:
    HAS_DEEPSEEK_VL2 = False
    DeepseekVLV2Processor = object
    DeepseekVLV2ForCausalLM = object


class DeepSeekVL2Classifier:
    def __init__(
        self,
        model_path: str = "deepseek-ai/deepseek-vl2-small",
        device: str = None,
    ):
        if not HAS_DEEPSEEK_VL2:
            raise ImportError(
                "DeepSeek VL2 dependencies are missing. Install with: "
                "pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git"
            )

        self.model_path = model_path
        self._requested_device = device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = 1.0

        self.processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

        if device in [None, "auto", "cuda"]:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            self.model = self.model.to(device)
            self.device = torch.device(device)

        self.model.eval()

    def set_temperature(self, temp: float):
        """Set the temperature for generation."""
        self.temperature = temp

    @staticmethod
    def _to_pil_image(image) -> Image.Image:
        """Convert various image formats to PIL Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if not isinstance(image, torch.Tensor):
            raise TypeError("Each image must be a PIL image or a torch.Tensor")

        tensor = image.detach().cpu()
        if tensor.dim() == 4:
            if tensor.size(0) != 1:
                raise ValueError("4D image tensors must have shape [1, C, H, W]")
            tensor = tensor.squeeze(0)

        if tensor.dim() != 3:
            raise ValueError("Image tensor must have shape [C, H, W] or [H, W, C]")

        if tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)

        arr = tensor.numpy()
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        return Image.fromarray(arr).convert("RGB")

    def predict_best_label_batch(self, images, label_option_list, baseprompt=None):
        """
        Batch interface for label prediction.
        Args:
            images (list[PIL.Image] | list[torch.Tensor]): input images
            label_option_list (list[list[str]]): candidate labels per image
            baseprompt (str): optional prompt template with {labels} or {label_text}
        Returns:
            list[str]: raw decoded model outputs
        """
        if len(images) != len(label_option_list):
            raise ValueError("images and label_option_list must have the same length")

        results = []

        for image, label_options in zip(images, label_option_list):
            if not isinstance(label_options, list):
                raise ValueError("Each element of label_option_list must be a list of strings")

            pil_image = self._to_pil_image(image)
            label_text = ", ".join(label_options)

            if baseprompt:
                prompt = baseprompt.format(labels=label_text, label_text=label_text)
            else:
                prompt = (
                    "You are given an image and a list of candidate labels.\n"
                    f"Candidates: [{label_text}].\n"
                    "Choose exactly ONE label from the candidates that best matches the main object in the image.\n"
                    "If none match, answer with only: None.\n"
                    "Return only the final answer label (or None), with no explanation."
                )

            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n{prompt}",
                    "images": [pil_image],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            prepare_inputs = self.processor(
                conversations=conversation,
                images=[pil_image],
                force_batchify=True,
            ).to(self.device)

            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

            gen_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": prepare_inputs.attention_mask,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "bos_token_id": self.tokenizer.bos_token_id,
                "max_new_tokens": 16,
            }
            if self.temperature == 1.0:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = self.temperature

            with torch.no_grad():
                outputs = self.model.generate(**gen_kwargs)

            decoded = self.tokenizer.decode(outputs[0].detach().cpu().tolist(), skip_special_tokens=True)
            results.append(decoded.strip())

        return results

    def generate_text(self, prompt: str, system_content: str = None) -> str:
        """
        Generate text using text-only prompt (no image).
        Args:
            prompt (str): the input prompt
            system_content (str): optional system message (currently unused for compatibility)
        Returns:
            str: generated text
        """
        formatted_prompt = prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        gen_kwargs = {
            "max_new_tokens": 128,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
        }
        if self.temperature == 1.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        decoded = self.tokenizer.decode(outputs[0].detach().cpu().tolist(), skip_special_tokens=True)
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):]
        return decoded.strip()
