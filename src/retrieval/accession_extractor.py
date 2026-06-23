from __future__ import annotations

import re
from typing import List

from src.constants import PRIMARY, SECONDARY
from src.data.schemas import ArticalDocument, CitationCandidate
from src.utils.text import clean_doi_alpha_num_article_id, has_authors_match


class AccessionExtractor:
    def __init__(self, repository, retrieval_config):
        self.repo = repository
        self.cfg = retrieval_config

    def _extract(self, article: ArticalDocument, article_title, text: str) -> List[CitationCandidate]:
        acc_chunks: list[CitationCandidate] = []
        article_doi = clean_doi_alpha_num_article_id(article.article_id)
        matches_list_acc = self.repo.acc_article_dict.get(article_doi, [])

        for match in matches_list_acc:
            acc = match["dataset"]
            regex_matches = re.compile(acc).finditer(text)

            for regex_match in regex_matches:
                start_idx = regex_match.start()
                if start_idx == -1:
                    continue

                text_chunk = text[max(0, start_idx - self.cfg.text_chunk_size): start_idx + self.cfg.text_chunk_size]
                acc_type = ""

                if "SAMN" in acc:
                    acc_type = PRIMARY

                inv_matches = self.repo.acc_dataset_dict.get(acc, [])
                num_matches_mdc = len([mx for mx in inv_matches if mx.get("extraction") == "mdc"])
                if num_matches_mdc > 10:
                    acc_type = SECONDARY

                dataset_title = self.repo.acc_mapping.get(acc, {}).get("titles", []) or []
                article_title_meta = self.repo.crossref_mapping.get(article_doi, {}).get("titles", []) or []
                dataset_authors = self.repo.acc_mapping.get(acc, {}).get("authors", []) or []
                article_authors = self.repo.crossref_mapping.get(article_doi, {}).get("authors", []) or []

                if has_authors_match(dataset_authors, article_authors):
                    acc_type = PRIMARY

                acc_chunks.append(
                    CitationCandidate(
                        article_id=article.article_id,
                        article_title=article_title_meta,
                        dataset_id=acc,
                        dataset_title=dataset_title,
                        text_chunk=text_chunk,
                        start_of_text=text[:500],
                        source="accession",
                        type_label=acc_type or None,
                    )
                )

        return acc_chunks

    def extract(self, article: ArticalDocument, text: str) -> List[CitationCandidate]:
        try:
            return self._extract(article, article.title, text)
        except Exception:
            return []
