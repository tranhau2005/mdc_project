from __future__ import annotations

from src.constants import DOI_PREFIX


class CandidateValidator:
    @staticmethod
    def is_noisy_identifier(dataset_id: str) -> bool:
        lowered = str(dataset_id).lower()
        if dataset_id.startswith(DOI_PREFIX) and "figshare" in lowered:
            return True
        if str(dataset_id).startswith("GCA_"):
            return True
        if str(dataset_id).startswith("HGNC:") or str(dataset_id).startswith("rs"):
            return True
        return False

    @staticmethod
    def is_doi(dataset_id: str) -> bool:
        return str(dataset_id).startswith(DOI_PREFIX)
