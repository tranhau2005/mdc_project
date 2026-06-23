from __future__ import annotations

import pandas as pd

from src.constants import PRIMARY, SECONDARY


class HeuristicLabeler:
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.empty:
            return df

        for idx, row in df.iterrows():
            current = row.get("type")
            if isinstance(current, str) and current != "":
                continue
            if pd.notna(row.get("type")) and row.get("type") is not None:
                continue

            dataset_id = str(row.get("dataset_id", ""))
            if dataset_id.startswith("https://doi.org/"):
                df.loc[idx, "type"] = None
            else:
                df.loc[idx, "type"] = None

        return df
