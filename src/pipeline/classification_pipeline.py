from __future__ import annotations

import pandas as pd

from src.classification.prompts import build_prompt


class ClassificationPipeline:
    def __init__(self, heuristic_labeler, ensemble_classifier=None, postprocessor=None, logger=None):
        self.heuristic_labeler = heuristic_labeler
        self.ensemble_classifier = ensemble_classifier
        self.postprocessor = postprocessor
        self.logger = logger

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = self.heuristic_labeler.apply(df)
        unresolved_mask = df["type"].isna() | (df["type"].fillna("") == "")

        if unresolved_mask.any() and self.ensemble_classifier is not None:
            unresolved_df = df.loc[unresolved_mask].copy().reset_index(drop=True)
            prompts = [build_prompt(row) for row in unresolved_df.to_dict(orient="records")]
            probs = self.ensemble_classifier.predict_proba(prompts)
            if len(probs) == len(unresolved_df):
                unresolved_df["secondary_prob"] = probs[:, 0]
                unresolved_df["primary_prob"] = probs[:, 1]
                df.loc[unresolved_mask, "secondary_prob"] = unresolved_df["secondary_prob"].values
                df.loc[unresolved_mask, "primary_prob"] = unresolved_df["primary_prob"].values

        if self.postprocessor is not None:
            df = self.postprocessor.run(df)

        df.loc[df["type"].fillna("") == "", "type"] = "Secondary"
        return df
