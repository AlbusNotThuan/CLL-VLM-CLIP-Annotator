import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from utils.path_manager import get_root_path
import os, sys

# ========== IMPORT OFFICIAL LLAVA REPO ==========
ROOT_PATH = get_root_path()
LLAVA_PATH = os.path.join(ROOT_PATH, "vlm/LLaVA")
if LLAVA_PATH not in sys.path:
    sys.path.append(LLAVA_PATH)

from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, \
                            DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init

from PIL import Image
from io import BytesIO
import requests
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

class LLaVAClassifier:
    def __init__(self,
                 model_path: str = "liuhaotian/llava-v1.5-7b",
                 device_map: str = "auto",
                 load_in_8_bit: bool = True,
                 **quant_kwargs) -> None:
        """Initialize the LLaVA model for classification (OL vs CL)."""
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.img_tensor = None
        self.conv = None
        self.roles = None
        self.stop_key = None

        self.load_models(model_path,
                         device_map=device_map,
                         load_in_8_bit=load_in_8_bit,
                         **quant_kwargs)

    def load_models(self,
                    model_path: str,
                    device_map: str,
                    load_in_8_bit: bool,
                    **quant_kwargs) -> None:
        """Load model, tokenizer, and vision tower."""
        quant_cfg = BitsAndBytesConfig(**quant_kwargs)
        self.model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            device_map=device_map,
            load_in_8bit=load_in_8_bit,
            quantization_config=quant_cfg
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        vision_tower = self.model.get_vision_tower()
        vision_tower.load_model()
        vision_tower.to(device="cuda")
        self.image_processor = vision_tower.image_processor

        disable_torch_init()

    def setup_image(self, img) -> None:
        """Load and process the image from path, URL, or direct PIL.Image/np.array."""
        if isinstance(img, str):
            if img.startswith("http"):
                response = requests.get(img)
                conv_img = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                conv_img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):  # PIL Image
            conv_img = img.convert("RGB")
        elif isinstance(img, np.ndarray):   # numpy array
            conv_img = Image.fromarray(img).convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(img)}")

        self.conv_img = conv_img
        self.img_tensor = self.image_processor.preprocess(
            self.conv_img,
            return_tensors="pt"
        )["pixel_values"].to(dtype=torch.float32, device="cuda")


    def generate_answer(self, do_sample=True, temperature=0.2,
                        max_new_tokens=64, use_cache=True, **kwargs) -> str:
        """Generate an answer from the current conversation."""
        raw_prompt = self.conv.get_prompt()
        input_ids = tokenizer_image_token(raw_prompt,
                                        self.tokenizer,
                                        IMAGE_TOKEN_INDEX,
                                        return_tensors='pt').unsqueeze(0).cuda()
        stopping = KeywordsStoppingCriteria([self.stop_key],
                                            self.tokenizer,
                                            input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=self.img_tensor,
                # stopping_criteria=[stopping],
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                **kwargs
            )
            # print(f"[DEBUG output_ids length]: {output_ids.shape}")
            # print(f"[DEBUG decoded full]: {self.tokenizer.decode(output_ids[0], skip_special_tokens=False)}")         
        # decoded = self.tokenizer.decode(
        #     output_ids[0, input_ids.shape[1]:]
        # ).strip()
        decoded = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()
        return decoded

    def classify_pair(self, img, label,
                    do_sample=True, temperature=0.2,
                    max_new_tokens=64, use_cache=True, **kwargs) -> tuple:
        """
        Classify if (image, label) is OL (ordinary label) or CL (complementary label).
        Returns:
            (predicted, raw_answer)
        """
        self.setup_image(img)

        # setup conversation
        conv_mode = "v1"
        self.conv = conv_templates[conv_mode].copy()
        self.roles = self.conv.roles
        if self.conv.sep_style == SeparatorStyle.TWO:
            self.stop_key = self.conv.sep2
        else:
            self.stop_key = self.conv.sep

        # prompt
        first_input = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN +
                    DEFAULT_IM_END_TOKEN +
                    f"\nQuestion: Does the label '{label}' match this image? "
                    "Answer only 'YES' or 'NO'.")
        self.conv.append_message(self.roles[0], first_input)
        self.conv.append_message(self.roles[1], None)

        # generate
        raw_answer = self.generate_answer(
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            **kwargs
        )
        # print(f"[DEBUG raw_answer]: {repr(raw_answer)}")
        # Map to OL/CL
        if "yes" in raw_answer.lower():
            predicted = "OL"
        elif "no" in raw_answer.lower():
            predicted = "CL"
        else:
            predicted = "UNKNOWN"

        return predicted, raw_answer

    def classify(self,
                    original_dataset,
                    shuffled_dataset,
                    indices=None,
                    save_path: str = None,
                    num_samples: int = None) -> pd.DataFrame:
        results = []
        if indices is None:
            indices = list(range(len(original_dataset)))
        if num_samples:
            indices = indices[:num_samples]

        for idx in tqdm(indices, desc="Classifying"):
            img, true_label_idx = original_dataset[idx]
            _, rand_label_idx = shuffled_dataset[idx]

            true_label = original_dataset.classes[true_label_idx]
            rand_label = shuffled_dataset.classes[rand_label_idx]

            predicted, raw_answer = self.classify_pair(img, rand_label)

            results.append({
                "index": idx,
                "true_label": true_label,
                "random_label": rand_label,
                "predicted": predicted,
                "raw_answer": raw_answer
            })

        df = pd.DataFrame(results)
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
        return df
