# Repository Specification

## Purpose

This project implements an end-to-end citation discovery and classification pipeline for identifying dataset references in scientific articles. The pipeline is designed around the Make Data Count task format, where each output row links an `article_id` to a `dataset_id` and assigns the reference type as `Primary` or `Secondary`.

## Scope

The repository contains source code, configuration, and command-line scripts only. It does not version raw PDFs/XML files, metadata pickles, trained model weights, intermediate CSVs, logs, or notebook outputs. Those files are expected to exist in local runtime directories described by the YAML configuration files.

## High-Level Architecture

The system has five main layers:

1. Data loading
   - Loads article PDFs and optional XML files.
   - Extracts normalized article text from PDFs using PyMuPDF.
   - Loads external metadata dictionaries and training labels from local data paths.

2. Candidate retrieval
   - Detects DOI-style dataset references with regex-based matching and metadata lookup.
   - Detects accession-style dataset references using precomputed article-to-dataset mappings.
   - Merges duplicate candidates by `article_id` and `dataset_id`.
   - Filters noisy identifiers before downstream classification.

3. Rule-based classification
   - Applies heuristics for obvious `Primary` and `Secondary` references.
   - Uses title similarity, author overlap, DOI/accession metadata, and source-specific rules.

4. Model-based classification
   - Builds natural-language prompts from unresolved citation candidates.
   - Uses DeBERTa sequence-classification checkpoints to estimate `Primary` and `Secondary` probabilities.
   - Supports fold-based ensemble inference over multiple model directories.

5. Postprocessing and evaluation
   - Applies threshold and quantile rules to convert probabilities into labels.
   - Supports majority-style postprocessing by article.
   - Computes classification metrics when labels are available.
   - Writes prediction and submission CSV files.

## Key Modules

- `src/config.py`: YAML configuration loader and typed config dataclasses.
- `src/data/loaders.py`: PDF/XML article loader.
- `src/data/repositories.py`: metadata and label repository loaders.
- `src/retrieval/doi_extractor.py`: DOI candidate extraction.
- `src/retrieval/accession_extractor.py`: accession candidate extraction.
- `src/retrieval/merger.py`: duplicate candidate consolidation.
- `src/pipeline/retrieval_pipeline.py`: combined retrieval orchestration.
- `src/classification/heuristics.py`: rule-based label assignment.
- `src/classification/deberta_classifier.py`: DeBERTa model wrapper.
- `src/classification/ensemble.py`: model ensemble interface.
- `src/classification/postprocess.py`: probability-to-label postprocessing.
- `src/training/ema.py`: EMA-enabled trainer for model training.
- `src/evaluation/metrics.py`: scoring utilities.

## Command-Line Entry Points

- `scripts/build_train_candidates.py`
  - Builds a supervised training CSV from known labels and local article text.

- `scripts/train_classifier.py`
  - Trains one fold or multiple folds of the DeBERTa classifier using prompt-formatted candidates.

- `scripts/run_inference.py`
  - Runs retrieval, classification, postprocessing, and optional evaluation from a YAML config.

- `scripts/run_ensemble.py`
  - Runs multi-fold ensemble classification over unresolved retrieval predictions.

- `scripts/run_evaluation.py`
  - Evaluates prediction outputs against labels.

- `scripts/evaluate_classifier_test.py`
  - Evaluates trained classifier checkpoints on held-out/test-style candidate data.

- `scripts/split_train_val_test.py`
  - Creates train/validation/test splits for experimentation.

## Inputs

Expected local inputs are configured in `configs/*.yaml`:

- PDF directory containing article files named by article id.
- Optional XML directory containing article files named by article id.
- `train_labels.csv` with at least `article_id`, `dataset_id`, and `type`.
- External metadata directory containing the pickled mappings used by retrieval.
- Optional DeBERTa tokenizer/model directories for training or inference.

## Outputs

Generated files are written under `artifacts/` by default and are intentionally ignored by git:

- retrieved candidate CSVs
- prediction CSVs
- submission-format CSVs
- model checkpoints
- fold metrics
- logs

## Configuration

Configuration is YAML-based:

- `configs/base.yaml`: shared default paths and runtime settings.
- `configs/train.yaml`: training paths, model settings, fold setup, optimizer settings, and EMA behavior.
- `configs/inference.yaml`: test/inference paths, model checkpoint list, thresholds, and runtime settings.

Relative paths are resolved from the configured `project_root`.

## Public Repository Policy

The repository must remain safe for public GitHub hosting:

- Do not commit `.env` files or credentials.
- Do not commit raw competition data, external metadata pickles, PDFs, XML files, or generated artifacts.
- Do not commit model checkpoints, optimizer states, tokenizer caches, or zip archives.
- Keep reproducible code and configuration in git; keep large/private runtime assets local or in a dedicated model/data registry.

## Known Operational Requirements

- Python 3.10 or newer is recommended.
- CUDA-capable GPU is recommended for DeBERTa training and ensemble inference.
- PyTorch, Transformers, pandas, NumPy, scikit-learn, PyMuPDF, PyYAML, RapidFuzz, and nameparser are required.
- Local data and model directories must match the configured paths before running the scripts.
