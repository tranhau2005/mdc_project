from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.classification.deberta_classifier import DebertaClassifier
from src.classification.prompts import build_prompt
from src.classification.postprocess import PredictionPostprocessor
from src.config import load_config
from src.data.repositories import LabelRepository
from src.evaluation.metrics import MetricEvaluator
from src.utils.io import save_dataframe
from src.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Run 6-fold ensemble in the same style as the original notebook.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-csv", required=True, help="Raw retrieval predictions CSV, e.g. artifacts/outputs/raw_df.csv")
    parser.add_argument("--output-csv", default=None, help="Final ensembled predictions CSV")
    parser.add_argument("--submission-csv", default=None, help="Optional submission CSV path")
    parser.add_argument("--parts-dir", default=None, help="Directory to store per-fold part CSVs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=None, help="GPU ids to use, e.g. --gpu-ids 0 1")
    parser.add_argument("--process-doi", action="store_true", help="Also classify unresolved DOI rows. Original notebook left these as default Secondary.")
    parser.add_argument("--skip-postprocess", action="store_true", help="Skip majority-vote postprocess.")

    # hidden worker mode, used by the parent process to mimic PART=... runs
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--part", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    return parser.parse_args()


def _normalize_type_column(df: pd.DataFrame) -> pd.Series:
    if "type" not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df["type"].fillna("").astype(str)


def _load_prediction_splits(predictions: pd.DataFrame, process_doi: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    type_col = _normalize_type_column(predictions)
    already_set = predictions[type_col != ""].copy().reset_index(drop=True)
    unresolved = predictions[type_col == ""].copy().reset_index(drop=True)

    doi_df = unresolved[unresolved["dataset_id"].astype(str).str.startswith("https://doi.org/")].copy().reset_index(drop=True)
    acc_df = unresolved[~unresolved["dataset_id"].astype(str).str.startswith("https://doi.org/")].copy().reset_index(drop=True)

    if process_doi:
        to_predict = pd.concat([doi_df, acc_df], ignore_index=True)
    else:
        to_predict = acc_df.copy()
    return already_set, doi_df, to_predict


def _build_prompts(df: pd.DataFrame) -> list[str]:
    return [build_prompt(row) for row in df.to_dict(orient="records")]


def run_worker(cfg_path: str, input_csv: str, model_path: str, output_csv: str, device: str = "cuda") -> None:
    cfg = load_config(cfg_path)
    df = pd.read_csv(input_csv)
    prompts = _build_prompts(df)

    clf = DebertaClassifier(
        model_path=model_path,
        tokenizer_path=cfg.model.tokenizer_path,
        max_length=cfg.model.max_length,
        batch_size=cfg.model.batch_size,
        device=device,
        use_fp16=cfg.model.use_fp16,
    ).load()

    probs = clf.predict_proba(prompts)
    out_df = df.copy()
    out_df["secondary_prob"] = probs[:, 0] if len(probs) else []
    out_df["primary_prob"] = probs[:, 1] if len(probs) else []
    out_df.to_csv(output_csv, index=False)


def _spawn_worker(
    script_path: Path,
    cfg_path: Path,
    input_csv: Path,
    model_path: str,
    output_csv: Path,
    gpu_id: int,
    part: int,
    logger,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable,
        str(script_path),
        "--worker",
        "--config",
        str(cfg_path),
        "--input-csv",
        str(input_csv),
        "--model-path",
        model_path,
        "--worker-output",
        str(output_csv),
        "--part",
        str(part),
        "--device",
        "cuda",
    ]
    logger.info("Launching part %s on GPU %s", part, gpu_id)
    return subprocess.Popen(cmd, env=env)


def _chunk_parts(parts: Iterable[int], chunk_size: int) -> list[list[int]]:
    parts = list(parts)
    return [parts[i : i + chunk_size] for i in range(0, len(parts), chunk_size)]


def _read_part_outputs(part_paths: list[Path]) -> tuple[list[np.ndarray], list[np.ndarray], pd.DataFrame]:
    primary_probs = []
    secondary_probs = []
    base_df = None
    for path in part_paths:
        part_df = pd.read_csv(path).reset_index(drop=True)
        if base_df is None:
            base_df = part_df.copy()
        primary_probs.append(part_df["primary_prob"].values)
        secondary_probs.append(part_df["secondary_prob"].values)
    assert base_df is not None
    return primary_probs, secondary_probs, base_df


def _combine_like_notebook(
    predictions: pd.DataFrame,
    to_predict: pd.DataFrame,
    part_paths: list[Path],
    process_doi: bool,
    postprocessor: PredictionPostprocessor,
    skip_postprocess: bool,
) -> pd.DataFrame:
    already_set_preds, doi_df_unresolved, _ = _load_prediction_splits(predictions, process_doi=process_doi)
    primary_probs, secondary_probs, part0 = _read_part_outputs(part_paths)

    primary_probs = np.stack(primary_probs).mean(axis=0)
    secondary_probs = np.stack(secondary_probs).mean(axis=0)

    if process_doi:
        doi_idx = part0[part0["dataset_id"].astype(str).str.startswith("https://doi.org/")].index.values
        acc_idx = part0[~part0["dataset_id"].astype(str).str.startswith("https://doi.org/")].index.values

        doi_df = part0.iloc[doi_idx].copy()
        doi_df["primary_prob"] = primary_probs[doi_idx]
        doi_df["secondary_prob"] = secondary_probs[doi_idx]
        doi_df.loc[doi_df["primary_prob"] >= postprocessor.doi_primary_threshold, "type"] = "Primary"
        doi_df.loc[doi_df["primary_prob"] < postprocessor.doi_primary_threshold, "type"] = "Secondary"
    else:
        doi_df = doi_df_unresolved.copy()

    acc_idx = part0[~part0["dataset_id"].astype(str).str.startswith("https://doi.org/")].index.values
    acc_df = part0.iloc[acc_idx].copy()
    acc_df["primary_prob"] = primary_probs[acc_idx]
    acc_df["secondary_prob"] = secondary_probs[acc_idx]
    acc_threshold = acc_df["secondary_prob"].quantile(postprocessor.acc_secondary_quantile)
    acc_df.loc[acc_df["secondary_prob"] >= acc_threshold, "type"] = "Secondary"
    acc_df.loc[acc_df["secondary_prob"] < acc_threshold, "type"] = "Primary"

    predictions_llm = pd.concat([already_set_preds, doi_df, acc_df]).reset_index(drop=True)
    predictions_llm.loc[_normalize_type_column(predictions_llm) == "", "type"] = "Secondary"

    if not skip_postprocess:
        predictions_llm = postprocessor.majority_vote_by_article(predictions_llm)
    return predictions_llm


def main_parent(args) -> None:
    cfg = load_config(args.config)
    logger = setup_logging(cfg.paths.log_dir, cfg.runtime.debug)

    predictions = pd.read_csv(args.input_csv)
    if args.limit is not None:
        predictions = predictions.head(args.limit).copy()

    already_set, doi_df, to_predict = _load_prediction_splits(predictions, process_doi=args.process_doi)
    logger.info(
        "Rows: total=%s, already_set=%s, unresolved_doi=%s, to_predict=%s",
        len(predictions), len(already_set), len(doi_df), len(to_predict),
    )

    if to_predict.empty:
        logger.warning("No unresolved rows to classify. Writing fallback outputs.")
        postprocessor = PredictionPostprocessor(
            doi_primary_threshold=cfg.thresholds.doi_primary_threshold,
            acc_secondary_quantile=cfg.thresholds.acc_secondary_quantile,
        )
        final_df = predictions.copy()
        final_df.loc[_normalize_type_column(final_df) == "", "type"] = "Secondary"
        if not args.skip_postprocess:
            final_df = postprocessor.majority_vote_by_article(final_df)
        output_csv = Path(args.output_csv or (cfg.paths.output_dir / "predictions_ensemble.csv"))
        save_dataframe(final_df, output_csv)
        return

    parts_dir = Path(args.parts_dir or (cfg.paths.output_dir / "ensemble_parts"))
    parts_dir.mkdir(parents=True, exist_ok=True)
    to_predict_csv = parts_dir / "to_predict.csv"
    to_predict.to_csv(to_predict_csv, index=False)

    output_csv = Path(args.output_csv or (cfg.paths.output_dir / "predictions_ensemble.csv"))
    submission_csv = Path(args.submission_csv or (cfg.paths.output_dir / "submission_ensemble.csv"))

    gpu_ids = args.gpu_ids or cfg.runtime.gpu_ids or [0]
    if len(gpu_ids) == 0:
        gpu_ids = [0]

    model_paths = cfg.model.model_paths
    if not model_paths:
        raise ValueError("No model_paths configured.")

    script_path = Path(__file__).resolve()
    cfg_path = Path(args.config).resolve()

    part_paths: list[Path] = []
    chunk_size = max(1, len(gpu_ids))
    for part_group in _chunk_parts(range(len(model_paths)), chunk_size):
        processes: list[subprocess.Popen] = []
        group_paths: list[Path] = []
        for offset, part in enumerate(part_group):
            gpu_id = gpu_ids[offset % len(gpu_ids)]
            part_output = parts_dir / f"llm_df_part_{part}.csv"
            group_paths.append(part_output)
            part_paths.append(part_output)
            proc = _spawn_worker(
                script_path=script_path,
                cfg_path=cfg_path,
                input_csv=to_predict_csv,
                model_path=model_paths[part],
                output_csv=part_output,
                gpu_id=gpu_id,
                part=part,
                logger=logger,
            )
            processes.append(proc)

        for proc, part in zip(processes, part_group):
            return_code = proc.wait()
            if return_code != 0:
                raise RuntimeError(f"Worker for part {part} failed with exit code {return_code}")

    postprocessor = PredictionPostprocessor(
        doi_primary_threshold=cfg.thresholds.doi_primary_threshold,
        acc_secondary_quantile=cfg.thresholds.acc_secondary_quantile,
    )
    final_df = _combine_like_notebook(
        predictions=predictions,
        to_predict=to_predict,
        part_paths=sorted(part_paths, key=lambda p: int(p.stem.split("_")[-1])),
        process_doi=args.process_doi,
        postprocessor=postprocessor,
        skip_postprocess=args.skip_postprocess,
    )

    save_dataframe(final_df, output_csv)
    submission = final_df[["article_id", "dataset_id", "type"]].copy()
    submission.insert(0, "row_id", range(len(submission)))
    save_dataframe(submission, submission_csv)

    labels = LabelRepository(cfg.paths.train_labels_path).load()
    if labels is not None:
        metrics = MetricEvaluator().evaluate(final_df, labels)
        logger.info("Final ensemble metrics: %s", metrics)


def main():
    args = parse_args()
    if args.worker:
        if args.model_path is None or args.worker_output is None:
            raise ValueError("Worker mode requires --model-path and --worker-output")
        run_worker(
            cfg_path=args.config,
            input_csv=args.input_csv,
            model_path=args.model_path,
            output_csv=args.worker_output,
            device=args.device,
        )
    else:
        main_parent(args)


if __name__ == "__main__":
    main()
