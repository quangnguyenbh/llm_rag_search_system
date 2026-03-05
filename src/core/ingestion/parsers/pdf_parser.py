"""PDF parsing using PyMuPDF (fitz) with fallback to pdfplumber."""

from pathlib import Path
import fitz  # PyMuPDF

from src.core.ingestion.parsers.base import DocumentParser, ParsedDocument, ParsedPage


class PdfParser(DocumentParser):
    def parse(self, file_path: Path) -> ParsedDocument:
        """Extract text and structure from a PDF file."""
        doc = fitz.open(str(file_path))
        pages: list[ParsedPage] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            headings = self._extract_headings(page)
            pages.append(ParsedPage(
                page_number=page_num + 1,
                text=text.strip(),
                headings=headings,
            ))

        title = doc.metadata.get("title", "") or file_path.stem
        doc.close()

        return ParsedDocument(
            title=title,
            pages=pages,
            format="pdf",
        )

    def _extract_headings(self, page: fitz.Page) -> list[str]:
        """Extract headings by detecting larger/bold font spans."""
        headings: list[str] = []
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["size"] > 14 or "bold" in span["font"].lower():
                        text = span["text"].strip()
                        if text:
                            headings.append(text)
        return headings
