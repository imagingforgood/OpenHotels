"""
Generic dataset classes for the hotel benchmark.

Transforms are applied inside __getitem__ so DataLoader workers
do the heavy lifting (resize, crop, normalise) in parallel.
"""
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Loads images from disk using JSON metadata and applies a transform.

    Each metadata entry must have:
        - ``path``:      relative image path from *image_root*
                         (e.g. ``images/gallery/00/000000.jpg``)
        - ``hotel_id``:  hotel identifier (int or str)

    Args:
        metadata_path: Path to a JSON file (list of dicts).
        image_root:    Root directory joined with ``path``.
        transform:     Callable ``PIL.Image → Tensor``.  Applied in
                       DataLoader worker processes for maximum throughput.
    """

    def __init__(self, metadata_path: str, image_root: str, transform):
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        self.image_root = image_root
        self.transform = transform

        # Pre-compute full file paths and IDs once for fast __getitem__
        self.paths = [
            os.path.join(image_root, entry["path"])
            for entry in self.metadata
        ]
        self.hotel_ids = [str(entry["hotel_id"]) for entry in self.metadata]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        hotel_id = self.hotel_ids[idx]
        try:
            image = Image.open(self.paths[idx]).convert("RGB")
        except (FileNotFoundError, OSError, Image.UnidentifiedImageError):
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        image = self.transform(image)  # PIL → Tensor (runs in worker)
        return image, hotel_id


def tensor_collate(batch):
    """Stack image tensors; keep hotel-ID strings as a list."""
    images, hotel_ids = zip(*batch)
    return torch.stack(images), list(hotel_ids)
