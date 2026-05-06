"""
Central configuration for the NeurIPS Hotel Benchmark.
"""
import os
import torch

# ── Repository root (one level up from benchmark/) ──────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── Dataset paths ────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join(_REPO_ROOT, "data")
IMAGE_ROOT = os.path.join(DATASET_ROOT, "full")

GALLERY_METADATA = os.path.join(DATASET_ROOT, "full", "metadata_gallery.json")
TEST_OBJECT_METADATA = os.path.join(DATASET_ROOT, "full", "metadata_test_object.json")
TEST_NON_OBJECT_METADATA = os.path.join(DATASET_ROOT, "full", "metadata_test_non_object.json")

# Split name → metadata path mapping
SPLITS = {
    "test_object": TEST_OBJECT_METADATA,
    "test_non_object": TEST_NON_OBJECT_METADATA,
}

# ── Benchmark paths ─────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ── Defaults ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 256
NUM_WORKERS = 8
TOP_K_METRICS = [1, 5, 10, 100]


def get_device(requested=None):
    """Auto-detect the best available device, or use a user override."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
