# OpenHotels

OpenHotels is a large-scale hotel image retrieval benchmark built from hotel-room imagery and associated hotel metadata. The task is hotel-scale retrieval: given a query image, retrieve the matching hotel from a large gallery containing both matching classes and many distractor hotel classes.

This repository will contain the code used to reproduce the OpenHotels experiments. The dataset itself is hosted separately on Hugging Face.

## Dataset

Full dataset:

- https://huggingface.co/datasets/imagingforgood/OpenHotels

Representative sample for inspection:

- https://huggingface.co/datasets/imagingforgood/OpenHotelsSample

The full release uses tar-sharded image files and JSON metadata. Each image metadata row includes:

- `path`: image member name inside the tar shard.
- `shard`: relative path to the tar shard containing the image.
- `hotel_id`: anonymized hotel class identifier.
- `room`: room identifier associated with the upload when available.
- `timestamp`: upload timestamp.

Gallery rows also include `is_object` plus either `view_type` for non-object room views or `object_type` for object-centric images. Test Non-Object rows include `view_type`; Test Object rows include `object_type`.

## Downloading the Dataset

Install the Hugging Face Hub client:

```bash
pip install huggingface_hub
```

Download the representative sample:

```bash
python scripts/download/download_openhotels.py --dataset sample --output-dir data
```

Download the full dataset:

```bash
python scripts/download/download_openhotels.py --dataset full --output-dir data
```

Download only metadata:

```bash
python scripts/download/download_openhotels.py --dataset full --metadata-only --output-dir data
```

The downloaded folder keeps the Hugging Face release structure, including `metadata_*.json` files and tar shards under `shards/`.

## Reproduction Scope

This repository is intended to support the following reproduction paths as implementation code is added.

### 1. Zero-Shot Results

This repository should include code to reproduce all zero-shot results reported in the paper using the current sharded OpenHotels dataset layout.

The zero-shot code should:

- Load gallery and query metadata from the Hugging Face release.
- Read images from tar shards using each row's `shard` and `path`.
- Extract embeddings for gallery and query images.
- Compute retrieval metrics for Test Non-Object and Test Object.
- Write machine-readable outputs that can be used to reproduce the reported tables.

### 2. Trained Model Weights

Any trained model weights used for reported paper results should be made available with clear download instructions. If weights are too large for GitHub, they should be hosted through a release, Hugging Face model repository, or another stable artifact store.

This repository should include:

- A manifest listing each trained checkpoint.
- The model architecture/backbone associated with each checkpoint.
- The table/result rows each checkpoint reproduces.
- Checksums where practical.

### 3. Evaluation with Trained Weights

This repository should include code to produce the relevant paper table entries from the released trained weights.

The evaluation code should:

- Load the checkpoint.
- Load OpenHotels gallery and query metadata.
- Read images from tar shards.
- Extract or load features.
- Compute Recall@K and any additional reported metrics.
- Write machine-readable metrics and paper-ready table rows.

### 4. Training Code

Training code is desirable but may be deferred if it cannot be completed before the submission deadline. If included, it should document:

- Training split construction.
- Data augmentations.
- Model/backbone configuration.
- Optimization settings.
- Hardware assumptions.
- Expected runtime.

## Status

This repository currently includes dataset download utilities. Zero-shot evaluation, trained-weight evaluation, and optional training code will be added as those components are finalized.
