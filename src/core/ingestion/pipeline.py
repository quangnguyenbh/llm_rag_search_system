"""Orchestrates the document ingestion flow: parse → chunk → embed → index."""

import structlog
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from src.core.ingestion.parsers.base import ParsedDocument
from src.core.ingestion.parsers.pdf_parser import PdfParser
from src.core.ingestion.parsers.html_parser import HtmlParser
from src.core.ingestion.chunker import SemanticChunker, Chunk
from src.core.ingestion.metadata import MetadataExtractor
from src.core.ingestion.embedder import BatchEmbedder
from src.db.vector.qdrant_client import upsert_chunks

logger = structlog.get_logger()

PARSER_MAP = {
    ".pdf": PdfParser,
    ".html": HtmlParser,
    ".htm": HtmlParser,
}


@dataclass
class IngestionResult:
    document_id: str
    file_path: str
    title: str
    chunks_count: int
    vectors_upserted: int


class IngestionPipeline:
    def __init__(
        self,
        chunker: SemanticChunker,
        metadata_extractor: MetadataExtractor,
        embedder: BatchEmbedder,
        qdrant: QdrantClient,
    ):
        self.chunker = chunker
        self.metadata_extractor = metadata_extractor
        self.embedder = embedder
        self.qdrant = qdrant

    async def ingest(self, file_path: Path, doc_metadata: dict | None = None) -> IngestionResult:
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
        title = metadata.get("title", parsed.title) or file_path.stem

        # 3. Chunk text
        chunks: list[Chunk] = self.chunker.chunk(parsed, metadata)
        logger.info("ingestion.chunked", chunks=len(chunks))

        if not chunks:
            logger.warning("ingestion.no_chunks", path=str(file_path))
            return IngestionResult(
                document_id=parsed.document_id,
                file_path=str(file_path),
                title=title,
                chunks_count=0,
                vectors_upserted=0,
            )

        # 4. Embed
        texts = [c.text for c in chunks]
        embeddings = await self.embedder.embed_batch(texts)

        # 5. Index in Qdrant
        upserted = upsert_chunks(
            client=self.qdrant,
            chunks=chunks,
            embeddings=embeddings,
            source_file=file_path.name,
            title=title,
        )

        logger.info(
            "ingestion.complete",
            path=str(file_path),
            document_id=parsed.document_id,
            chunks=len(chunks),
            upserted=upserted,
        )

        return IngestionResult(
            document_id=parsed.document_id,
            file_path=str(file_path),
            title=title,
            chunks_count=len(chunks),
            vectors_upserted=upserted,
        )
