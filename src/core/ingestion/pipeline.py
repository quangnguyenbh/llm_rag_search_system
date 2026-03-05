"""Orchestrates the document ingestion flow: parse → chunk → extract → embed → index."""

import structlog
from pathlib import Path

from src.core.ingestion.parsers.base import ParsedDocument
from src.core.ingestion.parsers.pdf_parser import PdfParser
from src.core.ingestion.parsers.html_parser import HtmlParser
from src.core.ingestion.chunker import SemanticChunker
from src.core.ingestion.table_extractor import TableExtractor
from src.core.ingestion.metadata import MetadataExtractor
from src.core.ingestion.embedder import BatchEmbedder

logger = structlog.get_logger()

PARSER_MAP = {
    ".pdf": PdfParser,
    ".html": HtmlParser,
    ".htm": HtmlParser,
}


class IngestionPipeline:
    def __init__(
        self,
        chunker: SemanticChunker,
        table_extractor: TableExtractor,
        metadata_extractor: MetadataExtractor,
        embedder: BatchEmbedder,
    ):
        self.chunker = chunker
        self.table_extractor = table_extractor
        self.metadata_extractor = metadata_extractor
        self.embedder = embedder

    async def ingest(self, file_path: Path, doc_metadata: dict | None = None) -> str:
        """Ingest a single document through the full pipeline."""
        logger.info("ingestion.start", path=str(file_path))

        # 1. Parse
        parser_cls = PARSER_MAP.get(file_path.suffix.lower())
        if not parser_cls:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        parser = parser_cls()
        parsed: ParsedDocument = parser.parse(file_path)

        # 2. Extract metadata
        metadata = self.metadata_extractor.extract(parsed, doc_metadata)

        # 3. Extract tables
        tables = self.table_extractor.extract(file_path, parsed)

        # 4. Chunk text
        chunks = self.chunker.chunk(parsed, metadata)

        # 5. Embed (text chunks + table summaries)
        all_texts = [c.text for c in chunks] + [t.nl_summary for t in tables]
        embeddings = await self.embedder.embed_batch(all_texts)

        # 6. Index in vector DB
        # TODO: Upsert to Qdrant
        # TODO: Store metadata in PostgreSQL
        # TODO: Update ingestion status

        logger.info(
            "ingestion.complete",
            path=str(file_path),
            chunks=len(chunks),
            tables=len(tables),
        )
        return parsed.document_id
