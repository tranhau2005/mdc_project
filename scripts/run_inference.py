from __future__ import annotations

import argparse
from pathlib import Path

from src.classification.deberta_classifier import DebertaClassifier
from src.classification.ensemble import EnsembleClassifier
from src.classification.heuristics import HeuristicLabeler
from src.classification.postprocess import PredictionPostprocessor
from src.config import load_config
from src.data.loaders import PDFXMLLoader
from src.data.repositories import DatabaseRepository, LabelRepository
from src.evaluation.metrics import MetricEvaluator
from src.pipeline.classification_pipeline import ClassificationPipeline
from src.pipeline.main_pipeline import MDCPipeline
from src.pipeline.retrieval_pipeline import CitationRetriever
from src.retrieval.accession_extractor import AccessionExtractor
from src.retrieval.doi_extractor import DOIExtractor
from src.retrieval.merger import PredictionMerger
from src.retrieval.regex import RegexRegistry
from src.utils.gpu import configure_torch, seed_everything
from src.utils.io import save_dataframe
from src.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logging(cfg.paths.log_dir, cfg.runtime.debug)
    seed_everything(cfg.runtime.seed)
    configure_torch()

    repo = DatabaseRepository(cfg.paths.database_dir).load()
    labels = LabelRepository(cfg.paths.train_labels_path).load()
    loader = PDFXMLLoader(cfg.paths.pdf_dir, cfg.paths.xml_dir)

    regex = RegexRegistry()
    doi_extractor = DOIExtractor(repo, regex, cfg.retrieval)
    accession_extractor = AccessionExtractor(repo, cfg.retrieval)
    merger = PredictionMerger()
    retriever = CitationRetriever(doi_extractor, accession_extractor, merger, cfg.retrieval, logger=logger)

    classifiers = []
    for model_path in cfg.model.model_paths:
        model_dir = Path(model_path)
        if model_dir.exists():
            clf = DebertaClassifier(
                model_path=str(model_dir),
                tokenizer_path=cfg.model.tokenizer_path,
                max_length=cfg.model.max_length,
                batch_size=cfg.model.batch_size,
                device=cfg.model.device,
                use_fp16=cfg.model.use_fp16,
            ).load()
            classifiers.append(clf)
        else:
            logger.warning("Model path not found: %s", model_dir)

    ensemble = EnsembleClassifier(classifiers) if classifiers else None
    postprocessor = PredictionPostprocessor(
        doi_primary_threshold=cfg.thresholds.doi_primary_threshold,
        acc_secondary_quantile=cfg.thresholds.acc_secondary_quantile,
    )
    classification_pipeline = ClassificationPipeline(
        heuristic_labeler=HeuristicLabeler(),
        ensemble_classifier=ensemble,
        postprocessor=postprocessor,
        logger=logger,
    )

    evaluator = MetricEvaluator() if labels is not None else None
    pipeline = MDCPipeline(loader, retriever, classification_pipeline, evaluator=evaluator, logger=logger)

    article_ids = loader.list_article_ids()
    if args.limit is not None:
        article_ids = article_ids[: args.limit]

    predictions, metrics = pipeline.run(article_ids=article_ids, label_df=labels)
    save_dataframe(predictions, cfg.paths.output_dir / "predictions.csv")

    submission = predictions[["article_id", "dataset_id", "type"]].copy()
    submission.insert(0, "row_id", range(len(submission)))
    save_dataframe(submission, cfg.paths.output_dir / "submission.csv")

    if metrics is not None:
        logger.info("Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
