from __future__ import annotations

import argparse
import pandas as pd

from src.evaluation.metrics import MetricEvaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    pred_df = pd.read_csv(args.predictions)
    label_df = pd.read_csv(args.labels)
    evaluator = MetricEvaluator()
    print(evaluator.evaluate(pred_df, label_df))


if __name__ == "__main__":
    main()
