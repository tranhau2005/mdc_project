from __future__ import annotations

import gc

import pandas as pd
import torch

from src.retrieval.validators import CandidateValidator


class CitationRetriever:
    def __init__(self, doi_extractor, accession_extractor, merger, retrieval_config, logger=None):
        self.doi_extractor = doi_extractor
        self.accession_extractor = accession_extractor
        self.merger = merger
        self.cfg = retrieval_config
        self.logger = logger

    def retrieve_for_article(self, article):
        doi_predictions = self.doi_extractor.extract(article, article.pdf_text)
        if len(doi_predictions) == 0 and article.xml_text:
            doi_predictions = self.doi_extractor.extract(article, article.xml_text)

        acc_predictions = self.accession_extractor.extract(article, article.pdf_text)
        acc_predictions_dataset_ids = set([p.dataset_id for p in acc_predictions])

        acc_predictions_xml = self.accession_extractor.extract(article, article.xml_text) if article.xml_text else []
        acc_predictions_xml = [p for p in acc_predictions_xml if p.dataset_id not in acc_predictions_dataset_ids]

        if len(acc_predictions) == 0 or len(acc_predictions_xml) <= 10:
            acc_predictions += acc_predictions_xml

        if any(pred.type_label for pred in doi_predictions):
            preds = doi_predictions
        elif len(doi_predictions) <= 1 and len(acc_predictions) > 1:
            preds = acc_predictions
        else:
            if len(doi_predictions) == 0:
                preds = acc_predictions
            elif len(acc_predictions) == 0:
                preds = doi_predictions
            else:
                if len(doi_predictions) > 1:
                    preds = doi_predictions
                else:
                    if len(acc_predictions) >= 7:
                        preds = acc_predictions
                    else:
                        preds = doi_predictions

        preds = [p for p in preds if not CandidateValidator.is_noisy_identifier(p.dataset_id)]

        for pred in preds:
            if pred.type_label == "Primary":
                pred.primary_prob = 1.0
                pred.secondary_prob = 0.0
            elif pred.type_label == "Secondary":
                pred.primary_prob = 0.0
                pred.secondary_prob = 1.0
            else:
                pred.primary_prob = 0.5
                pred.secondary_prob = 0.5

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return preds

    def to_dataframe(self, candidates: list) -> pd.DataFrame:
        columns = [
            "article_id", "dataset_id", "article_title", "dataset_title", "text_chunk",
            "start_of_text", "source", "type", "primary_prob", "secondary_prob", "metadata",
        ]
        if not candidates:
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame([c.to_dict() for c in candidates])
        df = (
            df.sort_values(by=["article_id"], ascending=[False])
            .reset_index(drop=True)
        )
        df = (
            df.groupby(["article_id", "dataset_id"])
            .apply(self.merger.group_unique)
            .reset_index(drop=True)
        )
        return df
