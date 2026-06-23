from __future__ import annotations

import copy

import pandas as pd

from src.constants import PRIMARY, SECONDARY


class PredictionPostprocessor:
    def __init__(self, doi_primary_threshold: float = 0.5, acc_secondary_quantile: float = 0.1):
        self.doi_primary_threshold = doi_primary_threshold
        self.acc_secondary_quantile = acc_secondary_quantile

    def apply_thresholds(self, predictions: pd.DataFrame) -> pd.DataFrame:
        predictions = copy.deepcopy(predictions)
        if predictions.empty:
            return predictions

        already_set_preds = predictions[predictions["type"].fillna("") != ""].copy()
        doi_df = predictions[(predictions["dataset_id"].astype(str).str.startswith("https://doi.org/")) & (predictions["type"].fillna("") == "")].copy()
        acc_df = predictions[(~predictions["dataset_id"].astype(str).str.startswith("https://doi.org/")) & (predictions["type"].fillna("") == "")].copy()

        if not doi_df.empty:
            doi_df.loc[doi_df["primary_prob"] >= self.doi_primary_threshold, "type"] = PRIMARY
            doi_df.loc[doi_df["primary_prob"] < self.doi_primary_threshold, "type"] = SECONDARY

        if not acc_df.empty:
            acc_threshold = acc_df["secondary_prob"].quantile(self.acc_secondary_quantile)
            acc_df.loc[acc_df["secondary_prob"] >= acc_threshold, "type"] = SECONDARY
            acc_df.loc[acc_df["secondary_prob"] < acc_threshold, "type"] = PRIMARY

        predictions_llm = pd.concat([already_set_preds, doi_df, acc_df]).reset_index(drop=True)
        predictions_llm.loc[predictions_llm["type"].fillna("") == "", "type"] = SECONDARY
        return predictions_llm

    def majority_vote_by_article(self, predictions: pd.DataFrame) -> pd.DataFrame:
        predictions = copy.deepcopy(predictions)
        for article_id in predictions["article_id"].unique():
            article_preds = predictions[predictions["article_id"] == article_id]
            if article_preds["dataset_id"].astype(str).str.startswith("https://doi.org/").any():
                continue
            type_counts = article_preds["type"].value_counts()
            if len(type_counts) > 0:
                if len(type_counts) > 1 and type_counts.iloc[0] == type_counts.iloc[1]:
                    for idx, row in article_preds.iterrows():
                        if str(row["dataset_id"]).startswith("https://doi.org/"):
                            predictions.loc[idx, "type"] = PRIMARY
                        else:
                            predictions.loc[idx, "type"] = SECONDARY
                else:
                    most_common_type = type_counts.index[0]
                    predictions.loc[predictions["article_id"] == article_id, "type"] = most_common_type
        return predictions

    def run(self, predictions: pd.DataFrame) -> pd.DataFrame:
        predictions = self.apply_thresholds(predictions)
        predictions = self.majority_vote_by_article(predictions)
        return predictions
