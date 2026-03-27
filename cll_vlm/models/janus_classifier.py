from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

try:
    from janus.models import MultiModalityCausalLM, VLChatProcessor

    HAS_JANUS = True
except ImportError:
    HAS_JANUS = False
    MultiModalityCausalLM = object
    VLChatProcessor = object


class JanusClassifier:
    def __init__(
        self,
        model_path: str = "deepseek-ai/Janus-Pro-7B",
        baseprompt: str = None,
        device: str = None,
        use_fast: bool = True,
        lazy_load: bool = True,
    ):
        if not HAS_JANUS:
            raise ImportError(
                "Janus dependencies are missing. Install with: "
                "pip install git+https://github.com/deepseek-ai/Janus.git"
            )

        self.processor = VLChatProcessor.from_pretrained(
            model_path,
            use_fast=use_fast,
            legacy=True,
        )
        self.tokenizer = self.processor.tokenizer

        self.model_path = model_path
        self._requested_device = device
        self._dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._compat_patched = False
        self.baseprompt = baseprompt
        self.temperature = 1.0

        if not lazy_load:
            self._ensure_model_loaded()

    def _from_pretrained_compat(self, **kwargs):
        """Load with modern dtype arg, but stay compatible with older Transformers versions."""
        common_kwargs = {
            "trust_remote_code": True,
            **kwargs,
        }
        try:
            return AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=self._dtype,
                **common_kwargs,
            )
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            return AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self._dtype,
                **common_kwargs,
            )

    def _target_device(self) -> str:
        if self._requested_device in [None, "auto", "cuda"]:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self._requested_device

    def _apply_runtime_compat_patches(self):
        """Patch known Janus/Transformers interoperability issues at runtime."""
        if self._compat_patched:
            return

        # Newer Transformers expects a mapping-like all_tied_weights_keys attribute.
        if not hasattr(MultiModalityCausalLM, "all_tied_weights_keys"):
            MultiModalityCausalLM.all_tied_weights_keys = {}

        # Janus SigLIP init calls x.item() over torch.linspace values. Under some
        # meta-init paths this becomes a meta tensor; force linspace on CPU.
        import janus.models.siglip_vit as siglip_vit

        if not getattr(siglip_vit, "_cll_safe_linspace_patch", False):
            original_linspace = siglip_vit.torch.linspace

            def _safe_cpu_linspace(*args, **kwargs):
                kwargs = dict(kwargs)
                kwargs.setdefault("device", "cpu")
                return original_linspace(*args, **kwargs)

            siglip_vit.torch.linspace = _safe_cpu_linspace
            siglip_vit._cll_safe_linspace_patch = True

        self._compat_patched = True

    @staticmethod
    def _infer_model_device(model) -> torch.device:
        hf_device_map = getattr(model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for location in hf_device_map.values():
                if isinstance(location, str) and location not in {"cpu", "disk", "meta"}:
                    return torch.device(location)
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _ensure_model_loaded(self):
        if self.model is not None:
            return

        self._apply_runtime_compat_patches()

        try:
            if self._requested_device in [None, "auto", "cuda"]:
                model = self._from_pretrained_compat(device_map="auto")
            else:
                model = self._from_pretrained_compat()
                model = model.to(self._target_device())
        except RuntimeError as exc:
            if "meta tensors" not in str(exc):
                raise

            print(
                "Janus auto device_map hit meta-tensor init error; "
                "retrying with eager loading (no device_map) as workaround."
            )
            model = self._from_pretrained_compat(device_map=None, low_cpu_mem_usage=False)
            target_device = self._target_device()
            if target_device != "cpu":
                model = model.to(target_device)

        model.eval()
        self.model = model
        self.device = self._infer_model_device(model)

    def set_temperature(self, temp: float):
        """Set generation temperature. Uses greedy decoding when temperature == 1.0."""
        self.temperature = temp

    @staticmethod
    def _to_pil_image(image) -> Image.Image:
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

    def _format_label_prompt(self, label_options: Sequence[str], baseprompt: str = None) -> str:
        label_text = ", ".join(label_options)
        prompt_template = baseprompt or self.baseprompt

        if prompt_template:
            return prompt_template.format(labels=label_text, label_text=label_text)

        return (
            "You are given an image and a list of candidate labels.\n"
            f"Candidates: [{label_text}].\n"
            "Choose exactly ONE label from the candidates that best matches the main object in the image.\n"
            "If none match, answer with only: None.\n"
            "Return only the final answer label (or None), with no explanation."
        )

    def _generate_multimodal_answer(self, image: Image.Image, prompt: str, max_new_tokens: int = 32) -> str:
        self._ensure_model_loaded()

        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [image],
            },
            {"role": "Assistant", "content": ""},
        ]

        prepared = self.processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
        ).to(self.device)

        inputs_embeds = self.model.prepare_inputs_embeds(**prepared)

        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": prepared.attention_mask,
            "pad_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        }
        if self.temperature == 1.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = self.model.language_model.generate(**gen_kwargs)

        return self.tokenizer.decode(outputs[0].detach().cpu().tolist(), skip_special_tokens=True)

    def generate_text(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Text-only generation helper."""
        self._ensure_model_loaded()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if self.temperature == 1.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = self.model.language_model.generate(**inputs, **gen_kwargs)

        decoded = self.tokenizer.decode(outputs[0].detach().cpu().tolist(), skip_special_tokens=True)
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):]
        return decoded.strip()

    def generate_text_batch(self, prompts: List[str], max_new_tokens: int = 128) -> List[str]:
        """Batch text generation helper for compatibility with other wrappers."""
        return [self.generate_text(prompt, max_new_tokens=max_new_tokens) for prompt in prompts]

    def predict_best_label_batch(self, images, label_option_list, baseprompt=None):
        """
        Batch interface compatible with existing classifiers.
        Args:
            images (list[PIL.Image] | list[torch.Tensor]): input images
            label_option_list (list[list[str]]): candidate labels per image
            baseprompt (str): optional prompt template with {labels} or {label_text}
        Returns:
            list[str]: raw decoded model outputs (no parsing at this stage)
        """
        if len(images) != len(label_option_list):
            raise ValueError("images and label_option_list must have the same length")

        results = []
        for image, label_options in zip(images, label_option_list):
            if not isinstance(label_options, list):
                raise ValueError("Each element of label_option_list must be a list of strings")

            pil_image = self._to_pil_image(image)
            prompt = self._format_label_prompt(label_options, baseprompt=baseprompt)
            raw_output = self._generate_multimodal_answer(pil_image, prompt, max_new_tokens=16)
            results.append(raw_output.strip())

        return results

    def predict_best_label_patch(self, images, label_option_list, baseprompt=None):
        """Compatibility alias for typoed interface name used by some scripts."""
        return self.predict_best_label_batch(images, label_option_list, baseprompt=baseprompt)
