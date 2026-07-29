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

## Environment Setup

To set up the required environment, run:

```bash
conda create -n OpenHotels python=3.12  -c conda-forge -y
conda activate OpenHotels

pip install torch==2.8.0 torchvision==0.23.0 --extra-index-url https://download.pytorch.org/whl/cu129
pip install transformers==4.56.2 torchmetrics==1.8.2 numpy==1.26.4 tqdm pandas==2.3.2 faiss-cpu==1.13.2 sympy==1.13.3 huggingface_hub==0.34.4 requests==2.32.5 pytorch_lightning==2.6.1 pytorch_metric_learning==2.9.0
```

For VPR training and evaluation, we forked [serizba/salad](https://github.com/serizba/salad), keeping the original codebase intact while adding support for the OpenHotels dataset and introducing our multi-vector SALAD aggregation approach. Clone the fork and switch to the `OpenHotels` branch:

```bash
git clone -b OpenHotels https://github.com/GWUvision/salad.git
```

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

### Extracting Images

After downloading the dataset, you must extract the tar shards into flat image directories for faster data loading during benchmarks. You can do this by running:

```bash
python scripts/extract_shards.py --data-dir data/full
```

## Model Checkpoints

All model checkpoints can be found in our Hugging Face collection:
[OpenHotels Models Collection](https://huggingface.co/collections/imagingforgood/openhotel-models)

### Foundation Models

- `imagingforgood/clip-vit-base-patch32-OpenHotels`
- `imagingforgood/dinov2-base-OpenHotels`
- `imagingforgood/vit-base-patch16-224-OpenHotels`
- `imagingforgood/siglip-base-patch16-224-OpenHotels`

### VPR Models

- `imagingforgood/salad-OpenHotels`
- `imagingforgood/CosPlace-OpenHotels`
- `imagingforgood/GeM-OpenHotels`
- `imagingforgood/epshn-resnet50-Hotels50k`
- `imagingforgood/ConvAP-OpenHotels`
- `imagingforgood/MixVPR-OpenHotels`
- `imagingforgood/salad_multivector-OpenHotels`

## Running Benchmarks

### Foundation Models

Users can run foundation model benchmarks using the following command:

```bash
python -m benchmark.run_benchmark --model vit --checkpoint imagingforgood/vit-base-patch16-224-OpenHotels --splits test_non_object
```

### VPR Models

Users can run VPR model benchmarks using the following command:

```bash
python eval_OpenHotels.py --hf_repo imagingforgood/CosPlace-OpenHotels
```

## Training VPR Models

We provide training code to fine-tune VPR models on OpenHotels. This is done through our fork of [serizba/salad](https://github.com/serizba/salad), which extends the original repository with:

- An **OpenHotels dataloader** adapted to our dataset format.
- A **multi-vector aggregation** approach built on top of the SALAD optimal transport framework.

The aggregation strategy and all other training hyperparameters are controlled via the `model_config` dict in `salad/main.py` (or by supplying a JSON override with `--model_config_path`). To start training, run:

```bash
cd salad/
python main.py
```

Or with a custom config:

```bash
cd salad/
python main.py --model_config_path path/to/model_config.json
```

Checkpoints are saved under `logs/` after each epoch.

## Results

### Foundation Models

| Model | Room R@1 | Room R@5 | Room R@10 | Room R@100 | Object R@1 | Object R@5 | Object R@10 | Object R@100 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `openai/clip-vit-base-patch32` | 11.80 | 17.09 | 19.70 | 31.29 | 5.82 | 8.78 | 10.09 | 15.55 |
| `facebook/dinov2-base` | 14.89 | 21.77 | 24.96 | 38.02 | 6.53 | 9.59 | 10.92 | 16.59 |
| `google/vit-base-patch16-224` | 13.58 | 19.89 | 22.93 | 35.08 | 7.13 | 10.40 | 11.97 | 18.10 |
| `google/siglip-base-patch16-224` | 15.44 | 22.04 | 25.19 | 39.01 | 9.40 | 13.39 | 15.11 | 22.29 |

*Retrieval performance for frozen general-purpose foundation models and after LoRA fine-tuning. Additional zero-shot results can be found in the Appendix.*

### VPR Models

| Method | Descriptor Size | Room R@1 | Room R@5 | Room R@10 | Room R@100 | Object R@1 | Object R@5 | Object R@10 | Object R@100 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| epshn_model (Baseline) | 256 | 22.51 | 32.35 | 36.61 | 51.88 | 5.32 | 8.11 | 9.54 | 15.99 |
| GeM | 1024 | 14.72 | 23.03 | 26.92 | 43.36 | 6.44 | 10.16 | 11.97 | 20.04 |
| MixVPR | 4096 | 18.47 | 27.16 | 31.31 | 48.23 | 9.37 | 13.78 | 15.76 | 23.53 |
| CosPlace | 2048 | 26.37 | 37.13 | 41.77 | 58.38 | 12.85 | 17.91 | 20.11 | 28.92 |
| ConvAP | 8192 | 26.50 | 37.19 | 41.76 | 58.29 | 12.10 | 16.68 | 18.72 | 26.38 |
| SALAD | 8448 (256+8192) | 31.60 | 42.64 | 47.20 | 62.59 | 14.41 | 19.50 | 21.73 | 29.80 |
| **Multi-Vector SALAD** | 8320 ((64+1)*128) | **34.11** | **45.24** | **49.58** | **64.32** | **15.64** | **20.83** | **23.19** | **31.64** |

*Performance using DINOv2-ViTB14 as a backbone across various state-of-the-art visual place recognition pooling and aggregation strategies.*

## Usage
OpenHotels and its associated models are provided to support research and educational work on visual recognition, retrieval, representation learning, and related areas. Because visual datasets and models can be used in contexts beyond those anticipated by their creators, access to OpenHotels and its associated models is gated to support thoughtful, responsible use and to make these usage expectations visible to prospective users.

These resources should not be used:
- to identify, locate, track, profile, or target specific individuals;
- in ways that facilitate surveillance, harassment, discrimination, exploitation, or other harm;
- to infer sensitive personal information about individuals;
- in law-enforcement, immigration-enforcement, military, intelligence, or commercial-surveillance applications without first consulting the maintainers;
- to reconstruct, re-identify, or expose information that has been removed, obscured, or anonymized;
- to redistribute the dataset, model weights, or access credentials to individuals who have not separately received access; or
- to create or release derivative resources that materially increase the risk of misuse without appropriate ethical and safety review.

Proposed uses in law-enforcement, immigration-enforcement, military, intelligence, or commercial-surveillance contexts warrant particular scrutiny and should undergo appropriate independent ethical, legal, and institutional review before proceeding.

Users are responsible for determining whether their proposed work requires review by an institutional review board, ethics committee, data-protection office, or other relevant authority. Access to these resources should not be understood as ethical, legal, institutional, or regulatory approval for a particular project.

We encourage users to describe the limitations and foreseeable risks of these resources in resulting publications and other outputs, consider the people and communities who could be affected by their use, and avoid claims or demonstrations that encourage harmful or inappropriate deployment.
