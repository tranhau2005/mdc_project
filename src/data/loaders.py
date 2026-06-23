from __future__ import annotations

from pathlib import Path

import fitz

from src.data.schemas import ArticalDocument
from src.utils.text import normalize


class PDFXMLLoader:
    def __init__(self, pdf_dir: Path, xml_dir: Path, pdf_text_mode: str = "plain"):
        self.pdf_dir = Path(pdf_dir)
        self.xml_dir = Path(xml_dir)
        self.pdf_text_mode = pdf_text_mode
        fitz.TOOLS.mupdf_display_errors(False)

    def list_article_ids(self) -> list[str]:
        return sorted([p.stem for p in self.pdf_dir.glob("*.pdf")])

    @staticmethod
    def extract_page_text(page: fitz.Page, mode: str = "plain") -> str:
        if mode == "plain":
            return page.get_text()
        if mode == "sorted_text":
            return page.get_text("text", sort=True)
        if mode == "blocks":
            blocks = page.get_text("blocks", sort=True)
            return "\n".join(block[4] for block in blocks if len(block) > 4 and block[4].strip())
        raise ValueError(f"Unsupported pdf_text_mode: {mode}")

    def load_article(self, article_id: str, pdf_text_mode: str | None = None) -> ArticalDocument:
        pdf_path = self.pdf_dir / f"{article_id}.pdf"
        xml_path = self.xml_dir / f"{article_id}.xml"

        doc = fitz.open(pdf_path)
        article_title = doc.metadata.get("title", "") or ""
        mode = pdf_text_mode or self.pdf_text_mode
        text = ""
        for page in doc:
            text += self.extract_page_text(page, mode=mode) + "\n"
        text = normalize(text)

        xml_text = ""
        if xml_path.exists():
            xml_text = normalize(xml_path.read_text(encoding="utf-8", errors="ignore"))

        return ArticalDocument(
            article_id=article_id,
            title=article_title,
            pdf_text=text,
            xml_text=xml_text,
            pdf_path=str(pdf_path),
            xml_path=str(xml_path),
        )
