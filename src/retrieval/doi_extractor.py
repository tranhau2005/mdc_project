from __future__ import annotations
from typing import List
from src.constants import DOI_PREFIX,PRIMARY, SECONDARY
from src.data.schemas import CitationCandidate, ArticalDocument
from src.utils.text import (clean_doi, clean_doi_alpha_num_article_id, clean_doi_alpha_num,get_title_dist, has_authors_match)

class DOIExtractor:
    def __init__(self,repository, regex_registry,retrieval_config):
        self.repo=repository
        self.regex= regex_registry
        self.cfg= retrieval_config
    def _extract(self, article: ArticalDocument, 
                 org_article_title: str, 
                 text: str,min_len: int,
                 max_len: int)-> List[CitationCandidate]:
        start_matches = list(self.regex.doi_start.finditer(text))
        doi_chunks: dict[str, CitationCandidate] = {}
        article_doi=clean_doi_alpha_num_article_id(article.article_id)
        for match in start_matches:
            start_idx= match.start()
            for cur_len in range (min_len, max_len):
                end_idx = start_idx +cur_len  
                chunk= text[start_idx:end_idx]
                output_doi=clean_doi(chunk)
                doi=clean_doi_alpha_num(chunk)
                text_chunk= text[max(0,start_idx-self.cfg.text_chunk_size): end_idx+self.cfg.text_chunk_size]
                info = self.repo.doi_dataset_dict.get(doi)
                if not doi or not info or doi in doi_chunks or output_doi is None:
                    continue
                publications =[clean_doi_alpha_num(dct['publication']) for dct in info]
                if article_doi and article_doi in publications:
                    info_idx = publications.index(article_doi)
                    info_dct= info[info_idx]
                    org_dataset_title= info_dct['title'] or ""
                    dataset_title =self.repo.datacite_mapping.get(doi,{}).get("titles",[]) or []
                    article_title_meta = self.repo.crossref_mapping.get(doi,{}).get("titles",[]) or []
                    dataset_authors =self.repo.datacite_mapping.get(doi,{}).get("authors",[]) or []
                    article_authors =self.repo.crossref_mapping.get(article_doi,{}).get("authors",[]) or []

                    info_dct_title_dist = get_title_dist(dataset_title, article_title_meta)
                    dataset_type = ""
                    if len(info) >= 5:
                        dataset_type= SECONDARY
                    if len(org_dataset_title) >10 and  len(org_article_title or "" )>10 and len(info)>1 and get_title_dist(org_dataset_title, org_article_title) >40:
                        dataset_type= SECONDARY
                    if info_dct_title_dist >=90:
                        dataset_type =PRIMARY
                    if "dryad" in doi:
                        dataset_type=PRIMARY
                    if has_authors_match(dataset_authors, article_authors):
                        dataset_type=PRIMARY
                    doi_chunks[doi] = CitationCandidate(
                        article_id=article.article_id,
                        article_title=article_title_meta,
                        dataset_id=f"{DOI_PREFIX}{output_doi}",
                        dataset_title=dataset_title,
                        text_chunk=text_chunk,
                        start_of_text=text[:500],
                        source="doi",
                        type_label=dataset_type or None,
                    )

        return list(doi_chunks.values())

    def extract(self, article: ArticalDocument, text: str, min_len: int | None = None, max_len: int | None = None) -> List[CitationCandidate]:
        try:
            return self._extract(
                article,
                article.title,
                text,
                min_len or self.cfg.doi_min_len,
                max_len or self.cfg.doi_max_len,
            )
        except Exception:
            return []
