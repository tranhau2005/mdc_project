from __future__ import annotations

import copy

import pandas as pd


def f1_score(tp, fp, fn):
    num = 2 * tp
    den = 2 * tp + fp + fn
    return (num / den) if den > 1e-8 else 0.0


class MetricEvaluator:
    def _metric(self, pred_df, label_df, with_type=True):
        if with_type:
            fields = ["article_id", "dataset_id", "type"]
        else:
            fields = ["article_id", "dataset_id"]

        missing_article_ids = label_df[label_df["type"] == "Missing"]["article_id"].unique()
        label_df = label_df[~label_df["article_id"].isin(missing_article_ids)]
        pred_df = pred_df[~pred_df["article_id"].isin(missing_article_ids)]

        hits_df = label_df.merge(pred_df, on=fields)
        tp = hits_df.shape[0]
        fp = pred_df.shape[0] - tp
        fn = label_df.shape[0] - tp
        f1 = f1_score(tp, fp, fn)
        return f1, tp, fp, fn

    def _classification_metric(self, pred_df, label_df, dataset_type):
        if dataset_type == "DOI":
            pred_subset = pred_df[pred_df["dataset_id"].str.startswith("https://doi.org/")]
            label_subset = label_df[label_df["dataset_id"].str.startswith("https://doi.org/")]
        else:
            pred_subset = pred_df[~pred_df["dataset_id"].str.startswith("https://doi.org/")]
            label_subset = label_df[~label_df["dataset_id"].str.startswith("https://doi.org/")]

        if len(pred_subset) == 0 or len(label_subset) == 0:
            return 0.0, 0, 0, 0, 0.0, 0, 0, 0

        matched_df = label_subset.merge(pred_subset, on=["article_id", "dataset_id"])
        if len(matched_df) == 0:
            return 0.0, 0, 0, 0, 0.0, 0, 0, 0

        primary_tp = len(matched_df[(matched_df["type_x"] == "Primary") & (matched_df["type_y"] == "Primary")])
        primary_fp = len(matched_df[(matched_df["type_x"] != "Primary") & (matched_df["type_y"] == "Primary")])
        primary_fn = len(matched_df[(matched_df["type_x"] == "Primary") & (matched_df["type_y"] != "Primary")])
        primary_f1 = f1_score(primary_tp, primary_fp, primary_fn)

        secondary_tp = len(matched_df[(matched_df["type_x"] == "Secondary") & (matched_df["type_y"] == "Secondary")])
        secondary_fp = len(matched_df[(matched_df["type_x"] != "Secondary") & (matched_df["type_y"] == "Secondary")])
        secondary_fn = len(matched_df[(matched_df["type_x"] == "Secondary") & (matched_df["type_y"] != "Secondary")])
        secondary_f1 = f1_score(secondary_tp, secondary_fp, secondary_fn)

        return primary_f1, primary_tp, primary_fp, primary_fn, secondary_f1, secondary_tp, secondary_fp, secondary_fn

    def evaluate(self, pred_df: pd.DataFrame, label_df: pd.DataFrame) -> dict:
        pred_df = copy.deepcopy(pred_df)
        label_df = copy.deepcopy(label_df)

        pred_type_doi = copy.deepcopy(pred_df)
        pred_type_acc = copy.deepcopy(pred_df)

        for idx, row in pred_df.iterrows():
            sel_df = label_df[label_df["dataset_id"] == row["dataset_id"]]
            if sel_df.shape[0] == 0:
                continue
            if str(row["dataset_id"]).startswith("https://doi.org/"):
                pred_type_doi.loc[idx, "type"] = sel_df["type"].iloc[0]
            else:
                pred_type_acc.loc[idx, "type"] = sel_df["type"].iloc[0]

        default_f1, default_tp, default_fp, default_fn = self._metric(pred_df, label_df, with_type=True)
        doi_gt_f1, doi_gt_tp, doi_gt_fp, doi_gt_fn = self._metric(pred_type_doi, label_df, with_type=True)
        acc_gt_f1, acc_gt_tp, acc_gt_fp, acc_gt_fn = self._metric(pred_type_acc, label_df, with_type=True)
        all_gt_f1, all_gt_tp, all_gt_fp, all_gt_fn = self._metric(pred_df, label_df, with_type=False)

        doi_cls = self._classification_metric(pred_df, label_df, "DOI")
        acc_cls = self._classification_metric(pred_df, label_df, "ACC")

        return {
            "default": {"f1": default_f1, "tp": default_tp, "fp": default_fp, "fn": default_fn},
            "doi_gt_type": {"f1": doi_gt_f1, "tp": doi_gt_tp, "fp": doi_gt_fp, "fn": doi_gt_fn},
            "acc_gt_type": {"f1": acc_gt_f1, "tp": acc_gt_tp, "fp": acc_gt_fp, "fn": acc_gt_fn},
            "retrieval_only": {"f1": all_gt_f1, "tp": all_gt_tp, "fp": all_gt_fp, "fn": all_gt_fn},
            "doi_classification": {
                "primary": {"f1": doi_cls[0], "tp": doi_cls[1], "fp": doi_cls[2], "fn": doi_cls[3]},
                "secondary": {"f1": doi_cls[4], "tp": doi_cls[5], "fp": doi_cls[6], "fn": doi_cls[7]},
            },
            "acc_classification": {
                "primary": {"f1": acc_cls[0], "tp": acc_cls[1], "fp": acc_cls[2], "fn": acc_cls[3]},
                "secondary": {"f1": acc_cls[4], "tp": acc_cls[5], "fp": acc_cls[6], "fn": acc_cls[7]},
            },
        }
