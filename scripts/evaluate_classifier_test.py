from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.deberta_classifier import DebertaClassifier
from src.classification.prompts import build_prompt
from src.config import load_config
from src.evaluation.metrics import MetricEvaluator
from src.utils.text import get_title_dist


LABEL_MAP = {"Secondary": 0, "Primary": 1}
INV_LABEL_MAP = {0: "Secondary", 1: "Primary"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--model-dir", default="artifacts/models")
    parser.add_argument("--output-dir", default="artifacts/test_eval")
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-title-heuristic", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune-threshold-from-oof", action="store_true")
    parser.add_argument("--threshold-mode", choices=["global", "by-kind"], default="global")
    return parser.parse_args()


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


def dataset_kind(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["dataset_id"].astype(str).str.startswith("https://doi.org/"), "doi", "acc")


def predict_with_threshold(df: pd.DataFrame, probs: np.ndarray, threshold: float | dict[str, float]) -> np.ndarray:
    primary_probs = probs[:, LABEL_MAP["Primary"]]
    if isinstance(threshold, dict):
        kinds = dataset_kind(df)
        pred = np.zeros(len(df), dtype=int)
        for kind, cutoff in threshold.items():
            mask = kinds == kind
            pred[mask] = (primary_probs[mask] >= cutoff).astype(int)
        return pred
    return (primary_probs >= threshold).astype(int)


def threshold_grid() -> np.ndarray:
    return np.round(np.arange(0.0, 1.0001, 0.001), 3)


def tune_global_threshold(y_true: np.ndarray, probs: np.ndarray, metric: str = "macro_f1") -> tuple[float, dict]:
    best_threshold = 0.5
    best_score = -1.0
    rows = []
    for threshold in threshold_grid():
        pred = (probs[:, LABEL_MAP["Primary"]] >= threshold).astype(int)
        score = f1_score(y_true, pred, average="macro" if metric == "macro_f1" else "binary", zero_division=0)
        rows.append({"threshold": float(threshold), "score": float(score), "pred_primary": int(pred.sum())})
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, {"best_score": float(best_score), "grid": rows}


def tune_kind_thresholds(df: pd.DataFrame, y_true: np.ndarray, probs: np.ndarray) -> tuple[dict[str, float], dict]:
    kinds = dataset_kind(df)
    thresholds = {}
    details = {}
    for kind in ("doi", "acc"):
        mask = kinds == kind
        if not mask.any():
            thresholds[kind] = 0.5
            details[kind] = {"best_score": None, "rows": 0}
            continue
        threshold, detail = tune_global_threshold(y_true[mask], probs[mask], metric="macro_f1")
        thresholds[kind] = threshold
        details[kind] = {"best_score": detail["best_score"], "rows": int(mask.sum())}
    pred = predict_with_threshold(df, probs, thresholds)
    details["combined_macro_f1"] = float(f1_score(y_true, pred, average="macro", zero_division=0))
    return thresholds, details


def load_classifier(config, model_dir: Path, fold: int, device: str, batch_size: int) -> DebertaClassifier:
    model_path = model_dir / f"fold_{fold}" / f"{config.train.run_name_prefix}-fold{fold}"
    if not (model_path / "ema_model").exists():
        raise FileNotFoundError(f"Missing EMA model for fold {fold}: {model_path / 'ema_model'}")

    return DebertaClassifier(
        model_path=str(model_path),
        tokenizer_path=config.train.tokenizer_path or config.model.tokenizer_path or str(model_path / "ema_model"),
        max_length=config.model.max_length,
        batch_size=batch_size,
        device=device,
        use_fp16=config.model.use_fp16,
    ).load()


def tune_threshold_from_oof(config, model_dir: Path, folds: list[int], device: str, batch_size: int, mode: str) -> tuple[
    float | dict[str, float], dict
]:
    frames = []
    prob_parts = []
    for fold in folds:
        valid_path = model_dir / f"fold_{fold}" / f"{config.train.run_name_prefix}-fold{fold}" / "valid_split.csv"
        if not valid_path.exists():
            raise FileNotFoundError(f"Missing valid split for fold {fold}: {valid_path}")
        valid_df = pd.read_csv(valid_path)
        valid_df = valid_df[valid_df["type"].isin(LABEL_MAP)].reset_index(drop=True)
        valid_df["prompt"] = [build_prompt(row) for row in valid_df.to_dict(orient="records")]

        classifier = load_classifier(config, model_dir, fold, device, batch_size)
        prob_parts.append(classifier.predict_proba(valid_df["prompt"].tolist()))
        frames.append(valid_df.drop(columns=["prompt"]))
        del classifier
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    oof_df = pd.concat(frames, ignore_index=True)
    oof_probs = np.concatenate(prob_parts, axis=0)
    y_oof = oof_df["type"].map(LABEL_MAP).to_numpy()
    if mode == "by-kind":
        threshold, tuning_details = tune_kind_thresholds(oof_df, y_oof, oof_probs)
    else:
        threshold, tuning_details = tune_global_threshold(y_oof, oof_probs, metric="macro_f1")

    pred = predict_with_threshold(oof_df, oof_probs, threshold)
    tuning_details.update(
        {
            "threshold_mode": mode,
            "threshold": threshold,
            "oof_rows": int(len(oof_df)),
            "oof_label_counts": oof_df["type"].value_counts().to_dict(),
            "oof_accuracy": float(accuracy_score(y_oof, pred)),
            "oof_f1_primary": float(f1_score(y_oof, pred, zero_division=0)),
            "oof_macro_f1": float(f1_score(y_oof, pred, average="macro", zero_division=0)),
            "oof_confusion_matrix": confusion_matrix(y_oof, pred, labels=[0, 1]).tolist(),
        }
    )
    return threshold, tuning_details


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = df[df["type"].isin(LABEL_MAP)].reset_index(drop=True)
    df["prompt"] = [build_prompt(row) for row in df.to_dict(orient="records")]
    y_true = df["type"].map(LABEL_MAP).to_numpy()

    folds = [int(fold.strip()) for fold in args.folds.split(",") if fold.strip()]
    device = args.device or config.model.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    batch_size = args.batch_size or config.model.batch_size

    threshold: float | dict[str, float] = args.threshold
    tuning_details = None
    if args.tune_threshold_from_oof:
        threshold, tuning_details = tune_threshold_from_oof(
            config=config,
            model_dir=Path(args.model_dir),
            folds=folds,
            device=device,
            batch_size=batch_size,
            mode=args.threshold_mode,
        )

    all_probs = []
    for fold in folds:
        classifier = load_classifier(config, Path(args.model_dir), fold, device, batch_size)
        probs = classifier.predict_proba(df["prompt"].tolist())
        all_probs.append(probs)

        del classifier
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    pred_ids_raw = probs.argmax(axis=1)
    pred_ids = predict_with_threshold(df, probs, threshold)
    if not args.no_title_heuristic and config.train.apply_title_heuristic:
        pred_ids = apply_title_heuristic(df, pred_ids)

    pred_df = df.drop(columns=["prompt"]).copy()
    pred_df["prob_secondary"] = probs[:, LABEL_MAP["Secondary"]]
    pred_df["prob_primary"] = probs[:, LABEL_MAP["Primary"]]
    pred_df["pred_label_id_raw"] = pred_ids_raw
    pred_df["pred_type_raw"] = pd.Series(pred_ids_raw).map(INV_LABEL_MAP)
    pred_df["pred_label_id"] = pred_ids
    pred_df["pred_type"] = pd.Series(pred_ids).map(INV_LABEL_MAP)
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    submission_df = pred_df[["article_id", "dataset_id", "pred_type"]].rename(columns={"pred_type": "type"})
    label_df = df.drop(columns=["prompt"])
    submission_df.to_csv(output_dir / "test_submission_format.csv", index=False)

    metrics = {
        "input_csv": args.input_csv,
        "model_dir": args.model_dir,
        "folds": folds,
        "device": device,
        "batch_size": batch_size,
        "threshold": threshold,
        "tuning_details": tuning_details,
        "rows": int(len(df)),
        "articles": int(df["article_id"].nunique()),
        "label_counts": df["type"].value_counts().to_dict(),
        "accuracy": float(accuracy_score(y_true, pred_ids)),
        "precision_primary": float(precision_score(y_true, pred_ids, zero_division=0)),
        "recall_primary": float(recall_score(y_true, pred_ids, zero_division=0)),
        "f1_primary": float(f1_score(y_true, pred_ids, zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred_ids, average="weighted", zero_division=0)),
        "confusion_matrix_labels": ["Secondary", "Primary"],
        "confusion_matrix": confusion_matrix(y_true, pred_ids, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            pred_ids,
            labels=[0, 1],
            target_names=["Secondary", "Primary"],
            zero_division=0,
            output_dict=True,
        ),
        "competition_metric": MetricEvaluator().evaluate(submission_df, label_df),
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
