"""
EPSHN benchmarker.

@InProceedings{Xuan_2020_WACV,
author = {Xuan, Hong and Stylianou, Abby and Pless, Robert},
title = {Improved Embeddings with Easy Positive Triplet Mining},
booktitle = {The IEEE Winter Conference on Applications of Computer Vision (WACV)},
month = {March},
year = {2020}
}

Architecture : ResNet-50 backbone → 256-dim FC head
Preprocessing: Resize(256) → CenterCrop(224) → Normalize (Hotels-50K stats)
Output       : L2-normalised 256-d feature vector per image
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from .base import BaseBenchmarker, make_data_parallel
from .registry import register_model
from ..config import CHECKPOINT_DIR, get_device

# ── EPSHN-specific constants ─────────────────────────────────────────────────
OUTPUT_DIM = 256
IMG_SIZE = 256
CROP_SIZE = 224

# Hotels-50K training-set statistics
_NORMALIZE = transforms.Normalize(
    mean=[0.5838, 0.5146, 0.4470],
    std=[0.6298, 0.6112, 0.4445],
)


@register_model("epshn")
class EPSHNBenchmarker(BaseBenchmarker):
    """
    ResNet-50 trained with EPSHN on Hotels-50K.

    Args:
        checkpoint: Local ``.pth`` path or HuggingFace repo ID.
                    Falls back to ``benchmark/checkpoints/epshn_model.pth``.
        device:     Device override (``cuda``, ``mps``, ``cpu``).
    """

    def __init__(self, checkpoint=None, device=None):
        self._device = get_device(device)
        print(f"[EPSHN] Using device: {self._device}")

        # ── Architecture ─────────────────────────────────────────────────────
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, OUTPUT_DIM)

        # ── Checkpoint ───────────────────────────────────────────────────────
        if checkpoint is None:
            checkpoint = os.path.join(CHECKPOINT_DIR, "epshn_model.pth")

        if os.path.isfile(checkpoint):
            ckpt_path = checkpoint
        else:
            # Treat as a HuggingFace repo ID and download the .pth file
            from huggingface_hub import hf_hub_download
            print(f"[EPSHN] Downloading from HuggingFace: {checkpoint}")
            ckpt_path = hf_hub_download(
                repo_id=checkpoint, filename="epshn_model.pth"
            )

        print(f"[EPSHN] Loading checkpoint: {ckpt_path}")
        state_dict = torch.load(
            ckpt_path, map_location=self._device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        print("[EPSHN] ✅ Checkpoint loaded.")

        self.model.to(self._device).eval()
        self.model = make_data_parallel(self.model, self._device)

    @property
    def device(self):
        return self._device

    def get_transform(self):
        """Resize → CenterCrop → Tensor → Normalize.  Runs in DataLoader workers."""
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            _NORMALIZE,
        ])

    def extract_features(self, batch):
        """
        Args:
            batch: Tensor (N, 3, 224, 224) on CPU.
        Returns:
            np.ndarray (N, 256), L2-normalised.
        """
        batch = batch.to(self._device)

        with torch.no_grad():
            if self._device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    features = self.model(batch)
            else:
                features = self.model(batch)

        features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy()
