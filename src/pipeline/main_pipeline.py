from __future__ import annotations

import pandas as pd


class MDCPipeline:
    def __init__(self, loader, retriever, classifier_pipeline, evaluator=None, logger=None):
        self.loader = loader
        self.retriever = retriever
        self.classifier_pipeline = classifier_pipeline
        self.evaluator = evaluator
        self.logger = logger

    def run(self, article_ids=None, label_df: pd.DataFrame | None = None):
        article_ids = article_ids or self.loader.list_article_ids()
        all_candidates = []

        for idx, article_id in enumerate(article_ids, start=1):
            article = self.loader.load_article(article_id)
            candidates = self.retriever.retrieve_for_article(article)
            all_candidates.extend(candidates)
            if self.logger and idx % 50 == 0:
                self.logger.info("Processed %s articles", idx)

        pred_df = self.retriever.to_dataframe(all_candidates)
        pred_df = self.classifier_pipeline.run(pred_df)

        metrics = None
        if self.evaluator is not None and label_df is not None:
            metrics = self.evaluator.evaluate(pred_df, label_df)
            if self.logger:
                self.logger.info("Evaluation: %s", metrics)

        return pred_df, metrics
