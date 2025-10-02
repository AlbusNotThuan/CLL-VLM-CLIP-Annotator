import torch
from PIL import Image
import os
import sys
from models.base_vlm import VLMModel
from utils.path_manager import get_root_path

# ========== IMPORT CLIP ==========
ROOT_PATH = get_root_path()
CLIP_PATH = os.path.join(ROOT_PATH, "vlm/CLIP")
if CLIP_PATH not in sys.path:
    sys.path.append(CLIP_PATH)
print(CLIP_PATH)
import clip

class CLIPModel(VLMModel):
    """
    CLIP wrapper implementation of VLMModel
    """
    def __init__(self, model_name: str="ViT-B/32", device: str=None, jit: bool=False):
        super().__init__(device)
        self.model, self.preprocess = clip.load(model_name, device=self.device, jit=jit)

    def encode_image(self, images):
        """
        Encode image(s)
        Args:
            images: PIL.Image | list[PIL.Image] | torch.Tensor
        Returns:
            torch.Tensor (B, D)
        """
        if isinstance(images, Image.Image):
            # single image
            images = self.preprocess(images).unsqueze(0)
        elif isinstance(images, list):
            # list of PIL.Image
            images = torch.stack([self.preprocess(img) for img in images])
        elif isinstance(images, torch.Tensor):
            # assume already preprocessed
            pass
        else:
            raise TypeError("images must be PIL.Image, list[PIL.Image], or torch.Tensor")
        
        images = images.to(self.device)
        
        with torch.no_grad():
            feats = self.model.encode_image(images)
        return feats
    
    def encode_text(self, texts):
        """
        Encode text(s)
        Args:
            texts: str | list[str]
        Returns:
            torch.Tensor (B, D)
        """
        if isinstance(texts, str):
            texts = [texts]
        tokens = clip.tokenize(texts).to(self.device)
        
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
        return feats