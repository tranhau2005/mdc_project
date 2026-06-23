from __future__ import annotations

SYS_PROMPT_CLASSIFY_DOI_DATASET = """
You will be given a list of information of both an article and a dataset linked by that article, to decide if the given {type} dataset is a primary or secondary dataset.

Primary - raw or processed data generated as part of the paper, specifically for the study
Secondary - raw or processed data derived or reused from existing records or published data

The information you will be given is:
- Article Title
- Dataset Title
- Dataset {type}
- Article Beginning
- Article Context
""".strip()

USER_PROMPT_CLASSIFY_DOI_DATASET = """
[Article Title]
{article_title}

[Dataset Title]
{dataset_title}

[Dataset {type}]
{dataset_id}

[Article Context]
{text_chunk}

[Article Beginning]
{start_of_text}
""".strip()

SYS_PROMPT_CLASSIFY_ACC_DATASET = """
You will be given a list of information of both an article and a dataset linked by that article, to decide if the given {type} dataset is a primary or secondary dataset.

Primary - raw or processed data generated as part of the paper, specifically for the study
Secondary - raw or processed data derived or reused from existing records or published data

The information you will be given is:
- Article Title
- Dataset {type}
- Article Beginning
- Article Context
""".strip()

USER_PROMPT_CLASSIFY_ACC_DATASET = """
[Article Title]
{article_title}

[Dataset {type}]
{dataset_id}

[Article Context]
{text_chunk}

[Article Beginning]
{start_of_text}
""".strip()


def _clip_text(value: str, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_prompt(row: dict) -> str:
    article_title = _clip_text(row.get("article_title", ""), 300)
    dataset_title = _clip_text(row.get("dataset_title", ""), 300)
    dataset_id = _clip_text(row.get("dataset_id", ""), 200)
    text_chunk = _clip_text(row.get("text_chunk", ""), 1200)
    start_of_text = _clip_text(row.get("start_of_text", ""), 220)

    is_doi = str(row["dataset_id"]).startswith("https://doi.org/")
    if is_doi:
        type_str = "DOI"
        return (
            SYS_PROMPT_CLASSIFY_DOI_DATASET.format(type=type_str)
            + "\n\n"
            + USER_PROMPT_CLASSIFY_DOI_DATASET.format(
                type=type_str,
                article_title=article_title,
                dataset_title=dataset_title,
                dataset_id=dataset_id,
                start_of_text=start_of_text,
                text_chunk=text_chunk,
            )
        )

    type_str = "Accession ID"
    return (
        SYS_PROMPT_CLASSIFY_ACC_DATASET.format(type=type_str)
        + "\n\n"
        + USER_PROMPT_CLASSIFY_ACC_DATASET.format(
            type=type_str,
            article_title=article_title,
            dataset_id=dataset_id,
            start_of_text=start_of_text,
            text_chunk=text_chunk,
        )
    )
