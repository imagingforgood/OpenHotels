"""
ViT (Vision Transformer) benchmarker — ImageNet-supervised baseline.

Default model: google/vit-base-patch16-224
Preprocessing handled by HuggingFace AutoImageProcessor (runs in DataLoader workers).
Output: L2-normalised CLS token feature vectors.
"""
import numpy as np
import torch
import torch.nn.functional as F

from transformers import ViTModel, AutoImageProcessor

from .base import BaseBenchmarker, make_data_parallel
from .registry import register_model
from ..config import get_device


class _ViTTransform:
    """Wraps AutoImageProcessor as a picklable torchvision-style transform."""

    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        inputs = self.image_processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)  # (C, H, W)


@register_model("vit")
class ViTBenchmarker(BaseBenchmarker):
    """
    Supervised ViT (ImageNet) image encoder for hotel retrieval.
    Extracts the CLS token feature and L2-normalises it.

    Args:
        checkpoint: HuggingFace model ID or local path
                    (default: ``google/vit-base-patch16-224``).
        device:     Device override (``cuda``, ``mps``, ``cpu``).
    """

    def __init__(self, checkpoint="google/vit-base-patch16-224", device=None):
        self._device = get_device(device)

        print(f"[ViT] Loading from: {checkpoint}")
        self.model = ViTModel.from_pretrained(checkpoint).to(self._device)
        self._image_processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model.eval()
        self.model = make_data_parallel(self.model, self._device)
        print(f"[ViT] ✅ Ready.  Device: {self._device}")

    @property
    def device(self):
        return self._device

    def get_transform(self):
        """HF image processor, wrapped for DataLoader workers."""
        return _ViTTransform(self._image_processor)

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
                    outputs = self.model(pixel_values=batch)
            else:
                outputs = self.model(pixel_values=batch)

            # Use raw CLS token (avoid the pooler's Dense+Tanh)
            if isinstance(outputs, dict):
                features = outputs["last_hidden_state"][:, 0, :]
            else:
                features = outputs.last_hidden_state[:, 0, :]

        features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy()
