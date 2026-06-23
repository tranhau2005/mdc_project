import argparse
import csv
import re
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data.loaders import PDFXMLLoader
from src.data.repositories import DatabaseRepository
from src.utils.text import clean_article_id, clean_doi

def build_exact_pattern(dataset_id: str):
    if dataset_id.startswith("https://doi.org/"):
        doi = dataset_id.replace("https://doi.org/", "").strip()
        doi = clean_doi(doi) or doi
        escaped = re.escape(doi)
        return re.compile(escaped, flags=re.IGNORECASE)

    return re.compile(re.escape(dataset_id), flags=re.IGNORECASE)


def lookup_article_title(repo: DatabaseRepository, article_id: str) -> str:
    article_key = clean_article_id(article_id, split=True)
    titles = repo.crossref_mapping.get(article_key, {}).get("titles", []) or []
    if isinstance(titles, list):
        return " ".join(str(title) for title in titles if title)
    return str(titles or "")


def lookup_dataset_title(repo: DatabaseRepository, dataset_id: str) -> str:
    if dataset_id.startswith("https://doi.org/"):
        doi = clean_doi(dataset_id.replace("https://doi.org/", "")) or ""
        doi_key = "".join(ch for ch in doi if ch.isalnum())
        titles = repo.datacite_mapping.get(doi_key, {}).get("titles", []) or []
    else:
        titles = repo.acc_mapping.get(dataset_id, {}).get("titles", []) or []

    if isinstance(titles, list):
        return " ".join(str(title) for title in titles if title)
    return str(titles or "")


def make_fallback_candidate(article, row, text, dataset_title: str, article_title: str, window=500):
    dataset_id = row["dataset_id"]
    pattern = build_exact_pattern(dataset_id)
    match = pattern.search(text)

    if match is None:
        return None

    start_idx = match.start()
    end_idx = match.end()

    text_chunk = text[max(0, start_idx - window): min(len(text), end_idx + window)]
    start_of_text = text[:500]

    return {
        "article_id": row["article_id"],
        "dataset_id": row["dataset_id"],
        "type": row["type"],
        "article_title": article_title,
        "dataset_title": dataset_title,
        "text_chunk": text_chunk,
        "start_of_text": start_of_text,
        "source": "fallback",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional override for output csv path",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    labels_path = Path(cfg.paths.train_labels_path)
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else Path(cfg.paths.output_dir) / "train_candidates.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    label_df = pd.read_csv(labels_path)
    label_df = label_df[label_df["type"] != "Missing"].copy()
    label_df = label_df.drop_duplicates(subset=["article_id", "dataset_id", "type"])

    repo = DatabaseRepository(cfg.paths.database_dir)
    repo.load()

    loader = PDFXMLLoader(
        pdf_dir=cfg.paths.pdf_dir,
        xml_dir=cfg.paths.xml_dir,
        pdf_text_mode=cfg.retrieval.pdf_text_mode,
    )

    all_rows = []
    article_ids = sorted(label_df["article_id"].unique())

    for article_id in article_ids:
        article = loader.load_article(article_id)
        article_labels = label_df[label_df["article_id"] == article_id].copy()
        for _, row in article_labels.iterrows():
            article_title = article.title or lookup_article_title(repo, row["article_id"])
            dataset_title = lookup_dataset_title(repo, row["dataset_id"])
            fallback = None

            if article.pdf_text:
                fallback = make_fallback_candidate(
                    article=article,
                    row=row,
                    text=article.pdf_text,
                    dataset_title=dataset_title,
                    article_title=article_title,
                )

            if fallback is None and article.xml_text:
                fallback = make_fallback_candidate(
                    article=article,
                    row=row,
                    text=article.xml_text,
                    dataset_title=dataset_title,
                    article_title=article_title,
                )

            if fallback is not None:
                all_rows.append(fallback)
            else:
                all_rows.append(
                    {
                        "article_id": row["article_id"],
                        "dataset_id": row["dataset_id"],
                        "type": row["type"],
                        "article_title": article_title,
                        "dataset_title": dataset_title,
                        "text_chunk": "",
                        "start_of_text": article.pdf_text[:500] if article.pdf_text else "",
                        "source": "unresolved",
                    }
                )

    out_df = pd.DataFrame(all_rows)
    out_df = out_df.drop_duplicates(subset=["article_id", "dataset_id", "type"])
    out_df = out_df.sort_values(["article_id", "dataset_id"]).reset_index(drop=True)

    out_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL, escapechar="\\")

    print(f"Saved: {output_csv}")
    print(f"Rows: {len(out_df)}")
    print("Source distribution:")
    print(out_df["source"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
