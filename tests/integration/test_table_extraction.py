"""Integration tests for table extraction pipeline."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.ingestion.table_extractor import (
    TableChunk,
    TableExtractor,
    _nl_summary,
    _rows_to_markdown,
    _rows_to_structured,
)
from src.core.ingestion.parsers.base import ParsedDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_document(doc_id: str = "doc-1", fmt: str = "pdf") -> ParsedDocument:
    doc = MagicMock(spec=ParsedDocument)
    doc.document_id = doc_id
    doc.format = fmt
    return doc


# ---------------------------------------------------------------------------
# Pure helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_rows_to_markdown_basic(self):
        headers = ["Name", "Value"]
        rows = [["Alpha", "1"], ["Beta", "2"]]
        md = _rows_to_markdown(headers, rows)
        assert "| Name | Value |" in md
        assert "| Alpha | 1 |" in md
        assert "| Beta | 2 |" in md

    def test_rows_to_markdown_empty_headers(self):
        assert _rows_to_markdown([], []) == ""

    def test_rows_to_structured_basic(self):
        headers = ["col_a", "col_b"]
        rows = [["x", "y"], ["p", "q"]]
        result = _rows_to_structured(headers, rows)
        assert result == [{"col_a": "x", "col_b": "y"}, {"col_a": "p", "col_b": "q"}]

    def test_rows_to_structured_short_row(self):
        headers = ["a", "b", "c"]
        rows = [["only_a"]]
        result = _rows_to_structured(headers, rows)
        assert result[0]["a"] == "only_a"
        assert result[0]["b"] is None
        assert result[0]["c"] is None

    def test_nl_summary_format(self):
        summary = _nl_summary(10, 3, ["A", "B", "C"], page_number=5)
        assert "10 rows" in summary
        assert "3 columns" in summary
        assert "page 5" in summary
        assert "A, B, C" in summary

    def test_nl_summary_no_page(self):
        summary = _nl_summary(5, 2, ["X", "Y"])
        assert "page" not in summary

    def test_nl_summary_truncates_long_column_list(self):
        cols = [f"col_{i}" for i in range(20)]
        summary = _nl_summary(100, 20, cols)
        assert "20 total" in summary


# ---------------------------------------------------------------------------
# TableExtractor — HTML
# ---------------------------------------------------------------------------


class TestHTMLExtraction:
    def test_extract_html_table(self, tmp_path):
        html_content = """
        <html><body>
        <table>
          <tr><th>Make</th><th>Model</th><th>Year</th></tr>
          <tr><td>Toyota</td><td>Camry</td><td>2020</td></tr>
          <tr><td>Honda</td><td>Civic</td><td>2021</td></tr>
        </table>
        </body></html>
        """
        html_file = tmp_path / "test.html"
        html_file.write_text(html_content)

        extractor = TableExtractor()
        doc = _make_document(fmt="html")
        chunks = extractor.extract_as_chunks(html_file, doc)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert isinstance(chunk, TableChunk)
        assert len(chunk.structured_json) == 2  # 2 data rows
        assert "Make" in chunk.column_names or "Make" in str(chunk.column_names)
        assert chunk.markdown != ""
        assert chunk.nl_summary != ""
        assert "rows" in chunk.nl_summary

    def test_extract_multiple_html_tables(self, tmp_path):
        html_content = """
        <html><body>
        <table>
          <tr><th>A</th><th>B</th></tr>
          <tr><td>1</td><td>2</td></tr>
        </table>
        <table>
          <tr><th>C</th><th>D</th></tr>
          <tr><td>3</td><td>4</td></tr>
        </table>
        </body></html>
        """
        html_file = tmp_path / "multi.html"
        html_file.write_text(html_content)

        extractor = TableExtractor()
        doc = _make_document(fmt="html")
        chunks = extractor.extract_as_chunks(html_file, doc)

        assert len(chunks) == 2

    def test_extract_html_no_tables(self, tmp_path):
        html_file = tmp_path / "no_tables.html"
        html_file.write_text("<html><body><p>No tables here</p></body></html>")

        extractor = TableExtractor()
        doc = _make_document(fmt="html")
        chunks = extractor.extract_as_chunks(html_file, doc)

        assert chunks == []

    def test_three_representations_present(self, tmp_path):
        """Each chunk must have all three representations."""
        html_content = """
        <html><body>
        <table>
          <tr><th>X</th><th>Y</th></tr>
          <tr><td>10</td><td>20</td></tr>
        </table>
        </body></html>
        """
        html_file = tmp_path / "repr.html"
        html_file.write_text(html_content)

        extractor = TableExtractor()
        doc = _make_document(fmt="html")
        chunks = extractor.extract_as_chunks(html_file, doc)

        assert chunks
        chunk = chunks[0]
        # 1. Structured JSON
        assert isinstance(chunk.structured_json, list)
        assert len(chunk.structured_json) > 0
        # 2. Markdown
        assert "|" in chunk.markdown
        # 3. NL summary
        assert "rows" in chunk.nl_summary and "columns" in chunk.nl_summary


# ---------------------------------------------------------------------------
# TableExtractor — PDF (mocked)
# ---------------------------------------------------------------------------


class TestPDFExtractionMocked:
    """PDF extraction tests using mocked Camelot / pdfplumber."""

    def test_pdf_via_pdfplumber_mock(self, tmp_path):
        """Test PDF extraction path using a mocked pdfplumber."""
        pdf_path = tmp_path / "fake.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_page = MagicMock()
        mock_page.page_number = 1
        mock_page.extract_tables.return_value = [
            [["Part", "Qty", "Price"], ["Bolt M6", "100", "0.50"], ["Nut M6", "80", "0.30"]]
        ]

        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page]

        extractor = TableExtractor()
        doc = _make_document(fmt="pdf")

        # Patch pdfplumber.open and make _camelot_extract raise so pdfplumber is used
        with (
            patch.object(extractor, "_camelot_extract", side_effect=RuntimeError("no camelot")),
            patch("pdfplumber.open", return_value=mock_pdf),
        ):
            chunks = extractor.extract_as_chunks(pdf_path, doc)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.page_number == 1
        assert len(chunk.structured_json) == 2
        assert "Part" in chunk.column_names

    def test_pdf_camelot_mock(self, tmp_path):
        """Test that camelot path is used when available."""
        pdf_path = tmp_path / "lattice.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        # Build mock camelot table
        import pandas as pd

        df = pd.DataFrame([["col_a", "col_b"], ["v1", "v2"]])
        mock_table = MagicMock()
        mock_table.df = df
        mock_table.page = 1

        mock_camelot = MagicMock()
        mock_camelot.read_pdf.return_value = [mock_table]

        extractor = TableExtractor()
        doc = _make_document(fmt="pdf")

        with patch.dict("sys.modules", {"camelot": mock_camelot}):
            chunks = extractor.extract_as_chunks(pdf_path, doc)

        assert len(chunks) == 1

    def test_unsupported_format_returns_empty(self, tmp_path):
        extractor = TableExtractor()
        doc = _make_document(fmt="docx")
        chunks = extractor.extract_as_chunks(tmp_path / "file.docx", doc)
        assert chunks == []

    def test_legacy_extract_method_compat(self, tmp_path):
        """The legacy extract() method returns ExtractedTable objects."""
        html_content = "<html><body><table><tr><th>A</th></tr><tr><td>1</td></tr></table></body></html>"
        html_file = tmp_path / "compat.html"
        html_file.write_text(html_content)

        from src.core.ingestion.table_extractor import ExtractedTable

        extractor = TableExtractor()
        doc = _make_document(fmt="html")
        tables = extractor.extract(html_file, doc)

        assert tables
        assert isinstance(tables[0], ExtractedTable)
