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

## Model results

The current local DeBERTa ensemble was trained with 6 folds. The training summary reports:

| Metric | Value |
| --- | ---: |
| Folds trained | 6 |
| Mean fold F1 | 0.7200 |
| Overall OOF F1 | 0.6886 |

Held-out evaluation was run on `artifacts/splits/train_candidates_test.csv` with 74 rows from 21 articles. The best documented test variants are:

| Variant | Threshold | Accuracy | Primary F1 | Macro F1 | Weighted F1 | Competition F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Tuned global | `0.546` | 0.8243 | 0.7111 | 0.7924 | 0.8122 | 0.8243 |
| Tuned by identifier kind | `doi=0.545`, `acc=0.538` | 0.8108 | 0.7308 | 0.7925 | 0.8075 | 0.8108 |

For the tuned-global run, the confusion matrix over `[Secondary, Primary]` was:

```text
[[45,  1],
 [12, 16]]
```

For the tuned-by-kind run, the confusion matrix over `[Secondary, Primary]` was:

```text
[[41,  5],
 [ 9, 19]]
```

## Repository Safety

This repo is prepared for public GitHub hosting. The `.gitignore` blocks datasets, model weights, generated artifacts, logs, notebooks, and environment files by default.
