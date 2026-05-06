"""
Minimal base class for all benchmark models.
"""
import numpy as np
import torch


def make_data_parallel(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Wrap *model* in ``DataParallel`` when multiple CUDA GPUs are available."""
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


class BaseBenchmarker:
    """
    Every benchmarker must provide:
        device          →  torch device
        get_transform() →  callable  PIL.Image → Tensor
        extract_features(batch: Tensor) → np.ndarray  (N, D), L2-normalised
    """

    @property
    def device(self):
        raise NotImplementedError

    def get_transform(self):
        """
        Return a callable ``PIL.Image → Tensor`` used by the DataLoader
        workers.  This runs in subprocesses, so it must be picklable
        (use ``transforms.Compose``, not lambdas).
        """
        raise NotImplementedError

    def extract_features(self, batch: torch.Tensor) -> np.ndarray:
        """
        Forward-pass a pre-transformed batch through the model.

        Args:
            batch: Tensor of shape ``(N, C, H, W)`` on **CPU**.
                   The runner handles ``.to(device)``.

        Returns:
            np.ndarray of shape ``(N, D)``, L2-normalised.
        """
        raise NotImplementedError
