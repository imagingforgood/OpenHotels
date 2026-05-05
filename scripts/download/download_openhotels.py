#!/usr/bin/env python3
"""Download OpenHotels releases from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DATASETS = {
    "full": "imagingforgood/OpenHotels",
    "sample": "imagingforgood/OpenHotelsSample",
}

METADATA_PATTERNS = [
    "README.md",
    "croissant.json",
    "metadata_*.json",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the OpenHotels full dataset or representative sample."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="sample",
        help="Which OpenHotels release to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where the dataset snapshot should be stored.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only README, Croissant, and metadata JSON files.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help=(
            "Optional Hugging Face allow patterns. Overrides the default patterns "
            "used by --metadata-only."
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional dataset repo revision, branch, tag, or commit hash.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_id = DATASETS[args.dataset]
    target_dir = args.output_dir / args.dataset

    allow_patterns = args.include
    if allow_patterns is None and args.metadata_only:
        allow_patterns = METADATA_PATTERNS

    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=target_dir,
        revision=args.revision,
        allow_patterns=allow_patterns,
    )

    print(f"Downloaded {repo_id} to {snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
