# Make Data Count Citation Pipeline

This repository contains a Python pipeline for the Make Data Count finding-data-references task. It extracts dataset citation candidates from article PDFs/XML, enriches them with metadata, applies rule-based labeling, and optionally classifies unresolved candidates with a DeBERTa ensemble.

## What is included

- `src/`: retrieval, classification, evaluation, training, and utility modules.
- `scripts/`: command-line entry points for candidate generation, training, inference, evaluation, and ensembling.
- `configs/`: YAML configuration files for training and inference.
- `SPEC.md`: repository specification and expected behavior.

## What is not included

The public repository intentionally excludes private or heavy local assets:

- raw competition data under `data/`
- generated artifacts under `artifacts/`
- trained model checkpoints and optimizer states
- logs, notebooks, zip archives, and local environment files

Place those assets locally using the paths configured in `configs/*.yaml`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU inference or training, install a PyTorch build compatible with your CUDA version before running the pipeline.

## Typical workflow

Build labeled training candidates:

```bash
python scripts/build_train_candidates.py --config configs/train.yaml
```

Train the DeBERTa classifier:

```bash
python scripts/train_classifier.py --config configs/train.yaml
```

Run inference:

```bash
python scripts/run_inference.py --config configs/inference.yaml
```

Run fold ensemble on retrieval predictions:

```bash
python scripts/run_ensemble.py \
  --config configs/inference.yaml \
  --input-csv artifacts/outputs/predictions.csv
```

## Repository Safety

This repo is prepared for public GitHub hosting. The `.gitignore` blocks datasets, model weights, generated artifacts, logs, notebooks, and environment files by default.
