"""
CLI entry-point for running hotel retrieval benchmarks.

Usage:
python -m benchmark.run_benchmark --model vit --checkpoint imagingforgood/vit-base-patch16-224-OpenHotels --splits test_non_object --batch-size 1024 --num-workers 16 
"""
import argparse
import json
import os
import time

import numpy as np
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader

from .config import (
    GALLERY_METADATA,
    IMAGE_ROOT,
    RESULTS_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    TOP_K_METRICS,
    SPLITS,
)
from .datasets.datasets import ImageDataset, tensor_collate
from .models import get_model
from .evaluation import build_faiss_index, search_index
from .evaluation.metrics import compute_recalls, format_results_table


# ═════════════════════════════════════════════════════════════════════════════
# Feature Extraction
# ═════════════════════════════════════════════════════════════════════════════

def extract_all_features(model, metadata_path, image_root, batch_size, num_workers, desc="Extracting"):
    """
    Run feature extraction over an entire dataset.

    Transforms run inside DataLoader workers (parallel).
    Tensors are pin-memory'd for fast GPU transfer.

    Returns:
        features: np.ndarray  (N, D)
        ids:      np.ndarray  (N,) of hotel-id strings
    """
    transform = model.get_transform()
    dataset = ImageDataset(metadata_path, image_root, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=tensor_collate,
    )

    all_features = []
    all_ids = []

    for batch_images, hotel_ids in tqdm(loader, desc=desc):
        feats = model.extract_features(batch_images)
        all_features.append(feats)
        all_ids.extend(hotel_ids)

    return np.vstack(all_features), np.array(all_ids)


# ═════════════════════════════════════════════════════════════════════════════
# Results I/O
# ═════════════════════════════════════════════════════════════════════════════

def save_results(model_name, split_name, recalls, gallery_size, query_size):
    """Persist results as a timestamped JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    result = {
        "model": model_name,
        "split": split_name,
        "timestamp": datetime.now().isoformat(),
        "gallery_size": gallery_size,
        "query_size": query_size,
        "recalls": {str(k): v for k, v in recalls.items()},
    }

    filename = f"{model_name}_{split_name}.json"
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"📁 Results saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NeurIPS Hotel Benchmark — Model Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name (e.g. epshn, clip, siglip)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=list(SPLITS.keys()),
        choices=list(SPLITS.keys()),
        help="Test splits to evaluate",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to local model weights or HuggingFace model ID",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (cuda, mps, cpu)",
    )
    parser.add_argument(
        "--top-k", type=int, nargs="+", default=TOP_K_METRICS,
        help="Top-K values for recall computation",
    )
    args = parser.parse_args()

    # ── 1. Create Model ──────────────────────────────────────────────────────
    model_kwargs = {}
    if args.checkpoint:
        model_kwargs["checkpoint"] = args.checkpoint
    if args.device:
        model_kwargs["device"] = args.device

    print(f"\n🚀 Initialising model: {args.model}")
    model = get_model(args.model, **model_kwargs)

    # ── 2. Extract Gallery Features (once, reused across splits) ─────────────
    print(f"\n📸 Extracting gallery features ...")
    t0 = time.time()
    gallery_features, gallery_ids = extract_all_features(
        model, GALLERY_METADATA, IMAGE_ROOT,
        args.batch_size, args.num_workers,
        desc="Gallery",
    )
    gallery_time = time.time() - t0
    print(
        f"   Gallery: {len(gallery_ids):,} images  |  "
        f"{gallery_features.shape[1]}-dim  |  {gallery_time:.1f}s"
    )

    # ── 3. Build FAISS Index (once) ──────────────────────────────────────────
    print("\n🔍 Building FAISS index ...")
    index = build_faiss_index(gallery_features)

    # ── 4. Evaluate Each Split ───────────────────────────────────────────────
    max_k = max(args.top_k)
    all_results = {}

    for split_name in args.splits:
        metadata_path = SPLITS[split_name]
        print(f"\n{'─' * 50}")
        print(f"📋 Evaluating split: {split_name}")
        print(f"{'─' * 50}")

        # Extract query features
        t0 = time.time()
        query_features, query_ids = extract_all_features(
            model, metadata_path, IMAGE_ROOT,
            args.batch_size, args.num_workers,
            desc=f"Queries [{split_name}]",
        )
        query_time = time.time() - t0
        print(f"   Queries: {len(query_ids):,} images  |  {query_time:.1f}s")

        # Search
        distances, indices = search_index(index, query_features, max_k)

        # Compute Recalls
        recalls = compute_recalls(query_ids, gallery_ids, indices, args.top_k)

        # Display
        table = format_results_table(args.model, split_name, recalls)
        print(f"\n{table}")

        # Persist
        save_results(args.model, split_name, recalls, len(gallery_ids), len(query_ids))
        all_results[split_name] = recalls

    # ── 5. Summary (when evaluating multiple splits) ─────────────────────────
    if len(args.splits) > 1:
        print(f"\n{'═' * 55}")
        print(f"  SUMMARY — {args.model.upper()}")
        print(f"{'═' * 55}")
        header = f"  {'Split':<22}" + "  ".join(f"R@{k:<3}" for k in args.top_k)
        print(header)
        print("  " + "─" * (len(header) - 2))
        for split_name, recalls in all_results.items():
            vals = "  ".join(f"{recalls[k]:>5.1f}" for k in args.top_k)
            print(f"  {split_name:<22} {vals}")
        print(f"{'═' * 55}")


if __name__ == "__main__":
    main()
