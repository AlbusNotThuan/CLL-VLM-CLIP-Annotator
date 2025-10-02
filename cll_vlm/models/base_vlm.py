import abc
import torch

class VLMModel(abc.ABC):
    """
    Abstract base class for VLM models
    ALl models (CLIP, ...) should inherit from this
    """
    def __init__(self, device: str=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @abc.abstractmethod
    def encode_image(self, images):
        """
        Encode an image or a batch of images
        Args:
            images: PIL.Image, list[PIL.Image], or torch.Tensor (B,3,H,W)
        Returns:
            torch.Tensor (B, D)
        """
        pass

    @abc.abstractmethod
    def encode_text(self, texts):
        """
        Encode text or batch of texts
        Args:
            texts: str or list[str]
        Returns:
            torch.Tensor (B, D)
        """
        pass

    def similarity(self, image_features, text_features, normalize=True):
        """
        Compute similarity between image and text embeddings
        Default: cosine similarity
        Args:
            image_features: torch.Tensor (N, D)
            text_features: torch.Tensor (M, D)
        Returns:
            torch.Tensor (N, M) similarity maatrixx
        """
        if normalize:
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return image_features @ text_features.T