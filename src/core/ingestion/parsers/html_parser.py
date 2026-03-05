"""HTML parsing using Trafilatura for main content extraction."""

from pathlib import Path
import trafilatura

from src.core.ingestion.parsers.base import DocumentParser, ParsedDocument, ParsedPage


class HtmlParser(DocumentParser):
    def parse(self, file_path: Path) -> ParsedDocument:
        """Extract main content from an HTML file."""
        raw_html = file_path.read_text(encoding="utf-8", errors="replace")

        extracted = trafilatura.extract(
            raw_html,
            include_tables=True,
            include_links=False,
            output_format="txt",
        )

        metadata = trafilatura.extract(
            raw_html,
            output_format="xmltei",
        )

        title = file_path.stem
        # Try to extract title from trafilatura metadata
        if metadata:
            import re
            title_match = re.search(r"<title[^>]*>(.*?)</title>", metadata, re.DOTALL)
            if title_match:
                title = title_match.group(1).strip()

        return ParsedDocument(
            title=title,
            pages=[ParsedPage(page_number=1, text=extracted or "")],
            raw_text=extracted or "",
            format="html",
        )
