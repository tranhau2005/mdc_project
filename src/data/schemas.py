from __future__ import annotations
from dataclasses import dataclass, field,asdict
from typing import Any,Optional
@dataclass
class ArticalDocument:
    """ Schema for representing an article document with its associated metadata and text content.

    Returns:
        article_id (str): Unique identifier for the article.
        title (str): Title of the article.
        pdf_text (str): Text extracted from the PDF version of the article.
        xml_text (str, optional): Text extracted from the XML version of the article. Defaults to an empty string.
        pdf_path (str, optional): File path to the PDF version of the article. Defaults to an empty string.
        xml_path (str, optional): File path to the XML version of the article. Defaults to an empty string.
    """
    article_id:str
    title:str
    pdf_text: str
    xml_text:str = ""
    pdf_path:str = ""
    xml_path:str = ""
    
    @property
    def all_text(self) -> str:
        return f"{self.pdf_text} \n\n {self.xml_text}" if self.xml_text else self.pdf_text
@dataclass
class CitationCandidate:
    """Sumarized schema for representing a citation candidate that links an article to a dataset, along with relevant metadata and probability scores.
    

    Returns:
        article_id (str): Unique identifier for the article.
        dataset_id (str): Unique identifier for the dataset.
        article_title (str, optional): Title of the article. Defaults to an empty string.
        dataset_title (str, optional): Title of the dataset. Defaults to an empty string.
        text_chunk (str, optional): Text chunk from the article that mentions the dataset. Defaults to an empty string.
        start_of_text (str, optional): Starting portion of the text chunk for context. Defaults to an empty string.
        source (str, optional): Source of the text chunk (e.g., "pdf", "xml"). Defaults to an empty string.
        type_label (str, optional): Type of citation (e.g., "doi", "url"). Defaults to None.
        primary_prob (float, optional): Primary probability score for the citation candidate. Defaults to 0.5.
        secondary_prob (float, optional): Secondary probability score for the citation candidate. Defaults to 0.5.
        metadata (dict[str, Any], optional): Additional metadata related to the citation candidate. Defaults to an empty dictionary.
    
    """
    article_id:str
    dataset_id:str
    article_title: Any =""
    dataset_title: Any =""
    text_chunk: str =""
    start_of_text: str =""
    source: str =""
    type_label: Optional[str] = None
    primary_prob: float =0.5
    secondary_prob: float =0.5
    metadata: dict[str,Any] = field(default_factory=dict)
    def to_dict(self)-> dict[str,Any]:
        data = asdict(self)
        data["type"]=data.pop("type_label")
        return data
"""
Testing
"""
# def main():
#     article = ArticalDocument(
#         article_id="A001",
#         title="Sample Article",
#         pdf_text="This is text from PDF.",
#         xml_text="This is text from XML.",
#         pdf_path="/data/sample.pdf",
#         xml_path="/data/sample.xml"
#     )

#     candidate = CitationCandidate(
#         article_id="A001",
#         dataset_id="D001",
#         article_title="Sample Article",
#         dataset_title="Dataset Title",
#         text_chunk="This dataset is cited in the article.",
#         start_of_text="This dataset...",
#         source="pdf",
#         type_label="doi",
#         primary_prob=0.92,
#         secondary_prob=0.08,
#         metadata={"page": 3, "match_score": 97}
#     )

#     print(article)
#     print(article.all_text)
#     print(candidate)
#     print(candidate.to_dict())

# if __name__ == "__main__":
#     main()