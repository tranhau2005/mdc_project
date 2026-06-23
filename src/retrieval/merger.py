from __future__ import annotations

import pandas as pd
from rapidfuzz import fuzz


class PredictionMerger:
    def deduplicate_similar_texts(self, texts, similarity_threshold=0.85, jaccard_threshold=0.7):
        if not texts or len(texts) <= 1:
            return texts

        def jaccard_similarity(text1, text2):
            words1 = set(str(text1).lower().split())
            words2 = set(str(text2).lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0

        def is_similar(text1, text2):
            len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
            if len_ratio < 0.5:
                return False
            jaccard_sim = jaccard_similarity(text1, text2)
            if jaccard_sim >= jaccard_threshold:
                return True
            if jaccard_sim >= 0.3:
                fuzzy_sim = fuzz.ratio(text1, text2) / 100.0
                return fuzzy_sim >= similarity_threshold
            return False

        unique_texts = []
        for text in texts:
            is_duplicate = False
            for existing_text in unique_texts:
                if is_similar(text, existing_text):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_texts.append(text)
        return unique_texts

    def group_unique(self, df: pd.DataFrame, max_nb_papers: int = 10) -> pd.Series:
        out = df.iloc[0].copy()
        all_text_chunks = df["text_chunk"].tolist()
        deduplicated_chunks = self.deduplicate_similar_texts(all_text_chunks)
        out["text_chunk"] = "\n...\n".join(deduplicated_chunks[:max_nb_papers])
        return out
