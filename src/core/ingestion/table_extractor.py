"""Table detection and extraction from PDF/HTML documents."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from src.core.ingestion.parsers.base import ParsedDocument

logger = structlog.get_logger()


@dataclass
class TableChunk:
    """All representations of a single extracted table."""

    table_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_doc_id: str = ""
    page_number: int | None = None
    structured_json: list[dict[str, Any]] = field(default_factory=list)
    markdown: str = ""
    nl_summary: str = ""
    column_names: list[str] = field(default_factory=list)


# Keep the original dataclass name as an alias for backward compatibility
@dataclass
class ExtractedTable:
    table_id: str = ""
    document_id: str = ""
    page_number: int | None = None
    structured_data: list[dict] | None = None  # List of row dicts
    markdown: str = ""
    nl_summary: str = ""  # Natural language summary for embedding


def _rows_to_markdown(headers: list[str], rows: list[list[Any]]) -> str:
    """Convert a table to a Markdown string."""
    if not headers:
        return ""
    sep = " | ".join(["---"] * len(headers))
    header_row = " | ".join(str(h) for h in headers)
    data_rows = [" | ".join(str(cell) for cell in row) for row in rows]
    return "\n".join([f"| {header_row} |", f"| {sep} |"] + [f"| {r} |" for r in data_rows])


def _rows_to_structured(headers: list[str], rows: list[list[Any]]) -> list[dict[str, Any]]:
    """Convert rows to list-of-dicts format."""
    result = []
    for row in rows:
        record: dict[str, Any] = {}
        for i, header in enumerate(headers):
            record[header] = row[i] if i < len(row) else None
        result.append(record)
    return result


def _nl_summary(
    n_rows: int,
    n_cols: int,
    col_names: list[str],
    page_number: int | None = None,
) -> str:
    page_hint = f" on page {page_number}" if page_number else ""
    cols_str = ", ".join(col_names[:10])
    if len(col_names) > 10:
        cols_str += f", … ({len(col_names)} total)"
    return (
        f"This table{page_hint} contains {n_rows} rows and {n_cols} columns. "
        f"Columns: {cols_str}."
    )


class TableExtractor:
    """Extract tables from PDF and HTML documents.

    For PDFs, tries Camelot (lattice mode first, then stream mode) and falls
    back to pdfplumber if Camelot is unavailable.  For HTML, uses
    ``pandas.read_html()``.

    Each extracted table is returned as a :class:`TableChunk` with three
    representations: structured JSON, Markdown, and a natural-language summary.
    """

    def extract_as_chunks(
        self, file_path: Path, document: ParsedDocument
    ) -> list[TableChunk]:
        """Extract tables and return them as :class:`TableChunk` objects."""
        if document.format == "pdf":
            return self._extract_pdf_chunks(file_path, document)
        if document.format == "html":
            return self._extract_html_chunks(file_path, document)
        return []

    def extract(self, file_path: Path, document: ParsedDocument) -> list[ExtractedTable]:
        """Legacy interface — returns :class:`ExtractedTable` objects."""
        chunks = self.extract_as_chunks(file_path, document)
        results = []
        for chunk in chunks:
            results.append(
                ExtractedTable(
                    table_id=chunk.table_id,
                    document_id=chunk.source_doc_id,
                    page_number=chunk.page_number,
                    structured_data=chunk.structured_json,
                    markdown=chunk.markdown,
                    nl_summary=chunk.nl_summary,
                )
            )
        return results

    # ------------------------------------------------------------------
    # PDF extraction
    # ------------------------------------------------------------------

    def _extract_pdf_chunks(
        self, file_path: Path, document: ParsedDocument
    ) -> list[TableChunk]:
        # Try Camelot first
        try:
            return self._camelot_extract(file_path, document)
        except Exception as exc:  # noqa: BLE001
            logger.warning("table_extractor.camelot_failed", error=str(exc))

        # Fallback to pdfplumber
        try:
            return self._pdfplumber_extract(file_path, document)
        except Exception as exc:  # noqa: BLE001
            logger.warning("table_extractor.pdfplumber_failed", error=str(exc))

        return []

    def _camelot_extract(
        self, file_path: Path, document: ParsedDocument
    ) -> list[TableChunk]:
        import camelot  # type: ignore[import]

        chunks: list[TableChunk] = []

        for mode in ("lattice", "stream"):
            try:
                tables = camelot.read_pdf(str(file_path), pages="all", flavor=mode)
                if not tables:
                    continue
                for tbl in tables:
                    df = tbl.df
                    if df.empty or df.shape[0] < 2:
                        continue
                    headers = [str(h) for h in df.iloc[0].tolist()]
                    rows = [row.tolist() for _, row in df.iloc[1:].iterrows()]
                    page_num = int(getattr(tbl, "page", None) or 0) or None
                    chunk = self._build_chunk(
                        headers, rows, document.document_id, page_num
                    )
                    chunks.append(chunk)
                if chunks:
                    break  # lattice succeeded — no need for stream
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "table_extractor.camelot_mode_failed", mode=mode, error=str(exc)
                )
                continue

        logger.info(
            "table_extractor.camelot_done", file=str(file_path), tables=len(chunks)
        )
        return chunks

    def _pdfplumber_extract(
        self, file_path: Path, document: ParsedDocument
    ) -> list[TableChunk]:
        import pdfplumber  # type: ignore[import]

        chunks: list[TableChunk] = []
        with pdfplumber.open(str(file_path)) as pdf:
            for page in pdf.pages:
                raw_tables = page.extract_tables()
                if not raw_tables:
                    continue
                for raw in raw_tables:
                    if not raw or len(raw) < 2:
                        continue
                    headers = [str(h) if h else f"col_{i}" for i, h in enumerate(raw[0])]
                    rows = [list(r) for r in raw[1:]]
                    chunk = self._build_chunk(
                        headers, rows, document.document_id, page.page_number
                    )
                    chunks.append(chunk)

        logger.info(
            "table_extractor.pdfplumber_done",
            file=str(file_path),
            tables=len(chunks),
        )
        return chunks

    # ------------------------------------------------------------------
    # HTML extraction
    # ------------------------------------------------------------------

    def _extract_html_chunks(
        self, file_path: Path, document: ParsedDocument
    ) -> list[TableChunk]:
        try:
            import pandas as pd  # type: ignore[import]
        except ImportError:
            logger.warning("table_extractor.pandas_not_available")
            return []

        try:
            dfs = pd.read_html(str(file_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("table_extractor.html_read_failed", error=str(exc))
            return []

        chunks: list[TableChunk] = []
        for df in dfs:
            if df.empty:
                continue
            headers = [str(c) for c in df.columns.tolist()]
            rows = df.values.tolist()
            chunk = self._build_chunk(headers, rows, document.document_id, page_number=None)
            chunks.append(chunk)

        logger.info(
            "table_extractor.html_done", file=str(file_path), tables=len(chunks)
        )
        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_chunk(
        self,
        headers: list[str],
        rows: list[list[Any]],
        doc_id: str,
        page_number: int | None,
    ) -> TableChunk:
        structured = _rows_to_structured(headers, rows)
        markdown = _rows_to_markdown(headers, rows)
        summary = _nl_summary(len(rows), len(headers), headers, page_number)
        return TableChunk(
            source_doc_id=doc_id,
            page_number=page_number,
            structured_json=structured,
            markdown=markdown,
            nl_summary=summary,
            column_names=headers,
        )
