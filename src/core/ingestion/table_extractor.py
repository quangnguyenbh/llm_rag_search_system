"""Table detection and extraction from PDF/HTML documents."""

from dataclasses import dataclass
from pathlib import Path

from src.core.ingestion.parsers.base import ParsedDocument


@dataclass
class ExtractedTable:
    table_id: str = ""
    document_id: str = ""
    page_number: int | None = None
    structured_data: list[dict] | None = None  # List of row dicts
    markdown: str = ""
    nl_summary: str = ""  # Natural language summary for embedding


class TableExtractor:
    def extract(self, file_path: Path, document: ParsedDocument) -> list[ExtractedTable]:
        """Extract tables from a document as structured units."""
        if document.format == "pdf":
            return self._extract_from_pdf(file_path, document)
        elif document.format == "html":
            return self._extract_from_html(file_path, document)
        return []

    def _extract_from_pdf(self, file_path: Path, document: ParsedDocument) -> list[ExtractedTable]:
        """Extract tables from PDF using camelot/pdfplumber."""
        # TODO: Use camelot for well-structured tables
        # TODO: Fallback to pdfplumber for complex layouts
        # TODO: Generate markdown representation
        # TODO: Generate NL summary for each table
        return []

    def _extract_from_html(
        self, file_path: Path, document: ParsedDocument
    ) -> list[ExtractedTable]:
        """Extract tables from HTML using pandas/BeautifulSoup."""
        # TODO: Use pandas.read_html for table extraction
        # TODO: Generate markdown + NL summaries
        return []
