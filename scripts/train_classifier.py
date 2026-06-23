
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

from src.classification.prompts import build_prompt
from src.config import load_config
from src.training.datasets import PromptClassificationDataset, PromptDataCollator
from src.training.ema import EMATrainer
from src.utils.gpu import configure_torch, seed_everything
from src.utils.io import ensure_dir
from src.utils.logging import setup_logging
from src.utils.text import get_title_dist


LABEL_MAP = {"Secondary": 0, "Primary": 1}
INV_LABEL_MAP = {0: "Secondary", 1: "Primary"}
REQUIRED_COLUMNS = [
    "article_id",
    "dataset_id",
    "type",
    "article_title",
    "dataset_title",
    "start_of_text",
    "text_chunk",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--fold", type=int, default=None)
    return parser.parse_args()


def load_training_frame(config, input_csv: str | None) -> pd.DataFrame:
    csv_path = input_csv or config.train.input_csv
    if csv_path is None:
        raise ValueError("Training requires --input-csv or train.input_csv in config.")
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Training CSV is missing columns: {missing}")
    if config.train.drop_missing:
        df = df[df["type"].isin(["Primary", "Secondary"])].copy()
    df = df.reset_index(drop=True)
    return df


def add_prompts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prompt"] = [build_prompt(row) for row in df.to_dict(orient="records")]
    df["label_id"] = df["type"].map(LABEL_MAP)
    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def build_model_and_tokenizer(config):
    base_model_path = config.train.base_model_path or config.model.tokenizer_path
    tokenizer_path = config.train.tokenizer_path or config.model.tokenizer_path or base_model_path
    if not base_model_path:
        raise ValueError("Set train.base_model_path in train.yaml.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model_cfg = AutoConfig.from_pretrained(
        base_model_path,
        num_labels=2,
        problem_type="single_label_classification",
    )
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path, config=model_cfg)
    model.config.use_cache = False
    # model.gradient_checkpointing_enable()
    return model, tokenizer


def apply_title_heuristic(df: pd.DataFrame, pred_ids: np.ndarray) -> np.ndarray:
    adjusted = pred_ids.copy()
    for idx, row in enumerate(df.to_dict(orient="records")):
        dataset_id = str(row.get("dataset_id", ""))
        if not dataset_id.startswith("https://doi.org/"):
            continue

        dataset_title = str(row.get("dataset_title", "") or "")
        article_title = str(row.get("article_title", "") or "")
        title_dist = get_title_dist(dataset_title, article_title)
        if dataset_title.lower().startswith("data from:") or title_dist >= 85:
            adjusted[idx] = LABEL_MAP["Primary"]
    return adjusted


def train_one_fold(config, df: pd.DataFrame, train_idx: np.ndarray, valid_idx: np.ndarray, fold: int, logger):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)

    model, tokenizer = build_model_and_tokenizer(config)
    collator = PromptDataCollator(tokenizer=tokenizer, max_length=config.model.max_length)
    train_ds = PromptClassificationDataset(train_df["prompt"].tolist(), train_df["label_id"].tolist())
    valid_ds = PromptClassificationDataset(valid_df["prompt"].tolist(), valid_df["label_id"].tolist())

    fold_dir = Path(config.train.output_dir).resolve() / f"fold_{fold}"
    run_name = f"{config.train.run_name_prefix}-fold{fold}"
    output_dir = fold_dir / run_name
    ensure_dir(output_dir)

    logger.info("Training fold %s | train=%s valid=%s | output=%s", fold, len(train_df), len(valid_df), output_dir)

    class_weights = None
    if config.train.use_class_weights:
        class_counts = train_df["label_id"].value_counts().reindex([0, 1], fill_value=0).astype(float)
        class_weights = torch.tensor(
            [
                len(train_df) / (2.0 * max(class_counts[0], 1.0)),
                len(train_df) / (2.0 * max(class_counts[1], 1.0)),
            ],
            dtype=torch.float32,
        )
        logger.info("Fold %s class weights: secondary=%.4f primary=%.4f", fold, class_weights[0], class_weights[1])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        learning_rate=config.train.learning_rate,
        per_device_train_batch_size=config.model.batch_size,
        per_device_eval_batch_size=config.model.batch_size,
        num_train_epochs=config.train.epochs,
        weight_decay=config.train.weight_decay,
        warmup_ratio=config.train.warmup_ratio,
        eval_strategy=config.train.eval_strategy,
        save_strategy=config.train.save_strategy,
        logging_steps=config.train.logging_steps,
        save_total_limit=config.train.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=config.train.metric_for_best_model,
        greater_is_better=config.train.greater_is_better,
        fp16=config.model.use_fp16 and torch.cuda.is_available(),
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        max_grad_norm=config.train.gradient_clipping,
        dataloader_num_workers=config.train.dataloader_num_workers,
        report_to=config.train.report_to,
        seed=config.runtime.seed,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        

    )

    trainer = EMATrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        ema_decay=config.train.ema_decay,
        ema_update_every=config.train.ema_update_every,
        ema_update_after_step=config.train.ema_update_after_step,
        class_weights=class_weights,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    trainer.save_state()

    valid_probs = trainer.predict(valid_ds).predictions
    valid_pred = np.argmax(valid_probs, axis=-1)
    if config.train.apply_title_heuristic:
        valid_pred = apply_title_heuristic(valid_df, valid_pred)

    adjusted_metrics = {
        "eval_accuracy": accuracy_score(valid_df["label_id"], valid_pred),
        "eval_precision": precision_score(valid_df["label_id"], valid_pred, zero_division=0),
        "eval_recall": recall_score(valid_df["label_id"], valid_pred, zero_division=0),
        "eval_f1": f1_score(valid_df["label_id"], valid_pred, zero_division=0),
    }
    metrics.update(adjusted_metrics)

    oof = valid_df[["article_id", "dataset_id", "type"]].copy()
    oof["pred_label_id"] = valid_pred
    oof["pred_type"] = oof["pred_label_id"].map(INV_LABEL_MAP)
    oof["fold"] = fold

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    valid_df.to_csv(output_dir / "valid_split.csv", index=False)
    oof.to_csv(output_dir / "oof_valid_predictions.csv", index=False)
    return metrics, oof


def main():
    args = parse_args()
    config = load_config(args.config)
    ensure_dir(Path(config.paths.log_dir))
    ensure_dir(Path(config.train.output_dir).resolve())
    logger = setup_logging(Path(config.paths.log_dir), debug=config.runtime.debug)

    seed_everything(config.runtime.seed)
    configure_torch()

    df = load_training_frame(config, args.input_csv)
    df = add_prompts(df)

    fold_override = args.fold if args.fold is not None else config.train.fold_index
    splitter = StratifiedGroupKFold(
        n_splits=config.train.num_folds,
        shuffle=True,
        random_state=config.runtime.seed,
    )

    all_metrics = []
    all_oof = []
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(df, y=df["label_id"], groups=df["article_id"])
    ):
        if fold_override is not None and fold != fold_override:
            continue
        metrics, oof = train_one_fold(config, df, train_idx, valid_idx, fold, logger)
        all_metrics.append({"fold": fold, **metrics})
        all_oof.append(oof)

    if not all_metrics:
        raise ValueError("No folds were trained. Check fold override / num_folds.")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = Path(config.train.output_dir).resolve() / "fold_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Saved fold metrics to %s", metrics_path)

    if all_oof:
        oof_df = pd.concat(all_oof, ignore_index=True)
        oof_path = Path(config.train.output_dir).resolve() / "oof_predictions.csv"
        oof_df.to_csv(oof_path, index=False)
        logger.info("Saved OOF predictions to %s", oof_path)

        overall_f1 = f1_score(
            oof_df["type"].map(LABEL_MAP),
            oof_df["pred_label_id"],
            zero_division=0,
        )
        summary = {
            "folds_trained": len(all_metrics),
            "mean_f1": float(metrics_df["eval_f1"].mean()) if "eval_f1" in metrics_df else None,
            "overall_oof_f1": float(overall_f1),
        }
        summary_path = Path(config.train.output_dir).resolve() / "training_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info("Training summary: %s", summary)


if __name__ == "__main__":
    main()
