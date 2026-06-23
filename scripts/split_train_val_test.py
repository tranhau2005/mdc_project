from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = ("Primary", "Secondary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="artifacts/outputs/train_candidates.csv")
    parser.add_argument("--output-dir", default="artifacts/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--search-steps", type=int, default=20000)
    return parser.parse_args()


def split_score(
    counts: np.ndarray,
    group_counts: np.ndarray,
    selected: np.ndarray,
    target_ratio: float,
    total_counts: np.ndarray,
    total_groups: int,
) -> float:
    split_counts = counts[selected].sum(axis=0)
    split_groups = int(selected.sum())
    split_rows = split_counts.sum()
    total_rows = total_counts.sum()

    if split_rows == 0:
        return 1_000_000.0

    full_primary_rate = total_counts[0] / total_rows
    split_primary_rate = split_counts[0] / split_rows
    missing_class_penalty = 100.0 if (split_counts == 0).any() else 0.0

    return (
        abs(split_rows / total_rows - target_ratio) * 4.0
        + abs(split_groups / total_groups - target_ratio) * 2.0
        + abs(split_primary_rate - full_primary_rate) * 3.0
        + missing_class_penalty
    )


def find_split(
    counts: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    search_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_groups = len(counts)
    indices = np.arange(n_groups)
    n_test = max(1, round(n_groups * test_ratio))
    n_val = max(1, round(n_groups * val_ratio))
    total_counts = counts.sum(axis=0)
    group_counts = np.ones(n_groups, dtype=np.int64)
    train_ratio = 1.0 - val_ratio - test_ratio

    best_score = float("inf")
    best_masks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for _ in range(search_steps):
        test_idx = rng.choice(indices, size=n_test, replace=False)
        remaining = np.setdiff1d(indices, test_idx, assume_unique=False)
        val_idx = rng.choice(remaining, size=n_val, replace=False)

        test_mask = np.zeros(n_groups, dtype=bool)
        val_mask = np.zeros(n_groups, dtype=bool)
        test_mask[test_idx] = True
        val_mask[val_idx] = True
        train_mask = ~(test_mask | val_mask)

        score = (
            split_score(counts, group_counts, train_mask, train_ratio, total_counts, n_groups)
            + split_score(counts, group_counts, val_mask, val_ratio, total_counts, n_groups)
            + split_score(counts, group_counts, test_mask, test_ratio, total_counts, n_groups)
        )
        if score < best_score:
            best_score = score
            best_masks = (train_mask.copy(), val_mask.copy(), test_mask.copy())

    if best_masks is None:
        raise RuntimeError("Could not find a split.")
    return best_masks


def summarize(df: pd.DataFrame, split_name: str) -> dict:
    part = df[df["split"] == split_name]
    label_counts = {label: int((part["type"] == label).sum()) for label in LABELS}
    return {
        "rows": int(len(part)),
        "articles": int(part["article_id"].nunique()),
        "labels": label_counts,
        "primary_rate": float(label_counts["Primary"] / len(part)) if len(part) else None,
    }


def main() -> None:
    args = parse_args()
    if args.val_ratio <= 0 or args.test_ratio <= 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("Expected 0 < val_ratio, 0 < test_ratio, and val_ratio + test_ratio < 1.")

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    missing = {"article_id", "type"} - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    df = df[df["type"].isin(LABELS)].reset_index(drop=True)
    grouped = df.groupby("article_id", sort=True)["type"].value_counts().unstack(fill_value=0)
    grouped = grouped.reindex(columns=LABELS, fill_value=0)
    article_ids = grouped.index.to_numpy()
    counts = grouped.to_numpy(dtype=np.int64)

    train_mask, val_mask, test_mask = find_split(
        counts=counts,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        search_steps=args.search_steps,
    )

    article_to_split = {}
    for split_name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        for article_id in article_ids[mask]:
            article_to_split[article_id] = split_name

    split_df = df.copy()
    split_df["split"] = split_df["article_id"].map(article_to_split)
    if split_df["split"].isna().any():
        raise RuntimeError("Some rows did not receive a split.")

    split_df.to_csv(output_dir / "train_candidates_with_split.csv", index=False)
    split_df[split_df["split"] == "train"].drop(columns=["split"]).to_csv(
        output_dir / "train_candidates_train.csv", index=False
    )
    split_df[split_df["split"] == "val"].drop(columns=["split"]).to_csv(
        output_dir / "train_candidates_val.csv", index=False
    )
    split_df[split_df["split"] == "test"].drop(columns=["split"]).to_csv(
        output_dir / "train_candidates_test.csv", index=False
    )
    split_df[split_df["split"].isin(["train", "val"])].drop(columns=["split"]).to_csv(
        output_dir / "train_candidates_trainval.csv", index=False
    )

    summary = {
        "input_csv": str(input_csv),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "group_column": "article_id",
        "splits": {name: summarize(split_df, name) for name in ("train", "val", "test")},
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
