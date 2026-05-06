"""
CLIP (Contrastive Language–Image Pretraining) benchmarker.

Default model: openai/clip-vit-base-patch32
Preprocessing handled by HuggingFace CLIPImageProcessor (runs in DataLoader workers).
Output: L2-normalised feature vectors.
"""
import numpy as np
import torch
import torch.nn.functional as F

from transformers import CLIPModel, CLIPImageProcessor

from .base import BaseBenchmarker, make_data_parallel
from .registry import register_model
from ..config import get_device


class _CLIPTransform:
    """Wraps CLIPImageProcessor as a picklable torchvision-style transform."""

    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        inputs = self.image_processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)  # (C, H, W)


@register_model("clip")
class CLIPBenchmarker(BaseBenchmarker):
    """
    CLIP image encoder for hotel retrieval.

    Args:
        checkpoint: HuggingFace model ID or local path
                    (default: ``openai/clip-vit-base-patch32``).
        device:     Device override (``cuda``, ``mps``, ``cpu``).
    """

    def __init__(self, checkpoint="openai/clip-vit-base-patch32", device=None):
        self._device = get_device(device)

        print(f"[CLIP] Loading from: {checkpoint}")
        self.model = CLIPModel.from_pretrained(checkpoint).to(self._device)
        self._image_processor = CLIPImageProcessor.from_pretrained(checkpoint)
        self.model.eval()
        self.model = make_data_parallel(self.model, self._device)
        print(f"[CLIP] ✅ Ready.  Device: {self._device}")

    @property
    def device(self):
        return self._device

    def get_transform(self):
        """HF image processor, wrapped for DataLoader workers."""
        return _CLIPTransform(self._image_processor)

    def extract_features(self, batch):
        """
        Args:
            batch: Tensor (N, C, H, W) on CPU.
        Returns:
            np.ndarray (N, D), L2-normalised.
        """
        batch = batch.to(self._device)

        with torch.no_grad():
            if self._device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    features = self.model.get_image_features(pixel_values=batch)
            else:
                features = self.model.get_image_features(pixel_values=batch)

        features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy()
