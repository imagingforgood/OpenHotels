#!/usr/bin/env python3
"""
Extract all tar shards into flat image directories.

Usage:
    python scripts/extract_shards.py                        # default: data/full
    python scripts/extract_shards.py --data-dir data/full   # explicit

After extraction the directory tree looks like:

    data/full/
    ├── images/
    │   ├── gallery/00/000000.jpg
    │   ├── test_non_object/00/000000.jpg
    │   └── ...
    ├── shards/          (kept as-is)
    └── metadata_*.json
"""
from __future__ import annotations

import argparse
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path


def _extract_one(shard_path: str, output_dir: str) -> str:
    """Extract a single shard into *output_dir*. Returns a status string."""
    basename = os.path.basename(shard_path)
    with tarfile.open(shard_path, "r:") as tar:
        tar.extractall(path=output_dir)
    return f"  ✓ {basename}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract all tar shards into image directories."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/full"),
        help="Root of the dataset (contains shards/ and metadata JSON files).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel extraction workers.",
    )
    args = parser.parse_args()

    shard_dir = args.data_dir / "shards"
    shards = sorted(glob(str(shard_dir / "*.tar")))

    if not shards:
        print(f"No .tar files found in {shard_dir}")
        return 1

    output_dir = str(args.data_dir)
    print(f"Extracting {len(shards)} shards → {output_dir}/images/")
    print(f"Using {args.workers} workers\n")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_extract_one, s, output_dir): s for s in shards
        }
        for fut in as_completed(futures):
            try:
                print(fut.result())
            except Exception as exc:
                print(f"  ✗ {os.path.basename(futures[fut])}: {exc}")

    print("\nDone. You can now run benchmarks with plain file reads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
