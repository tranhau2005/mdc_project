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

The latest documented evaluation uses a transparent final split that excludes the
current `test/PDF` debug articles from classifier training and threshold tuning.
The debug set contained 30 PDF articles, 12 of which overlapped with the original
candidate-training CSV, so the final report uses a separate held-out article list.

Final split summary:

| Split | Rows | Articles | Primary | Secondary |
| --- | ---: | ---: | ---: | ---: |
| Classifier train | 600 | 172 | 220 | 380 |
| Final held-out | 106 | 30 | 39 | 67 |

The final local DeBERTa ensemble was retrained with 6 folds on the classifier
train split:

| Metric | Value |
| --- | ---: |
| Folds trained | 6 |
| Mean fold F1 | 0.5907 |
| Overall OOF F1 | 0.5944 |

Thresholds were tuned only from OOF validation predictions. The selected frozen
threshold for final reporting is the global primary threshold `0.543`.

Classifier-only evaluation on `artifacts/final_eval/train_candidates_final_heldout.csv`:

| Variant | Threshold | Accuracy | Primary Precision | Primary Recall | Primary F1 | Macro F1 | Weighted F1 | Competition F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tuned global | `0.543` | 0.9057 | 1.0000 | 0.7436 | 0.8529 | 0.8917 | 0.9020 | 0.9057 |
| Tuned by identifier kind | `doi=0.543`, `acc=0.558` | 0.9057 | 1.0000 | 0.7436 | 0.8529 | 0.8917 | 0.9020 | 0.9057 |

Classifier-only confusion matrix over `[Secondary, Primary]`:

```text
[[67,  0],
 [10, 29]]
```

Full pipeline evaluation on the same 30 held-out articles, starting from PDF/XML:

| Metric | Value |
| --- | ---: |
| Label rows | 106 |
| Prediction rows | 92 |
| Precision | 0.9783 |
| Recall | 0.8491 |
| F1 | 0.9091 |
| TP / FP / FN | 90 / 2 / 16 |

Retrieval-only metrics for the full pipeline:

| Metric | Value |
| --- | ---: |
| Precision | 0.9891 |
| Recall | 0.8585 |
| F1 | 0.9192 |
| TP / FP / FN | 91 / 1 / 15 |

The generated local report files are under `artifacts/final_eval/`, which is not
committed because generated artifacts and model weights are intentionally ignored.

## Repository Safety

This repo is prepared for public GitHub hosting. The `.gitignore` blocks datasets, model weights, generated artifacts, logs, notebooks, and environment files by default.
