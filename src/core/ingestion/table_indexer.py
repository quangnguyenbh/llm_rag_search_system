"""Table-aware indexing into a dedicated Qdrant collection."""

from __future__ import annotations

import uuid
from typing import Any

import structlog

from src.config import settings
from src.core.ingestion.table_extractor import TableChunk
from src.db.vector.qdrant_client import init_tables_collection

logger = structlog.get_logger()


class TableIndexer:
    """Embeds table NL summaries and upserts them into the tables collection.

    Parameters
    ----------
    qdrant_client:
        A connected :class:`qdrant_client.QdrantClient` instance.
    embedder:
        A :class:`src.core.ingestion.embedder.BatchEmbedder` for dense vectors.
    """

    def __init__(self, qdrant_client: Any, embedder: Any) -> None:
        self.qdrant = qdrant_client
        self.embedder = embedder

    async def index_tables(self, tables: list[TableChunk]) -> int:
        """Embed + upsert *tables* into the tables collection.

        Returns the number of tables successfully upserted.
        """
        if not tables:
            return 0

        init_tables_collection(self.qdrant)

        texts = [t.nl_summary for t in tables]
        embeddings = await self.embedder.embed_batch(texts)

        from qdrant_client.models import PointStruct  # avoid circular at module level

        points = []
        for table, embedding in zip(tables, embeddings, strict=True):
            point_id = table.table_id or str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "table_id": point_id,
                        "doc_id": table.source_doc_id,
                        "page_number": table.page_number,
                        "markdown": table.markdown,
                        "structured_json": table.structured_json,
                        "column_names": table.column_names,
                        "nl_summary": table.nl_summary,
                        # text field mirrors nl_summary for retriever compatibility
                        "text": table.nl_summary,
                        "document_id": table.source_doc_id,
                    },
                )
            )

        self.qdrant.upsert(
            collection_name=settings.qdrant_tables_collection,
            points=points,
        )
        logger.info(
            "table_indexer.upserted",
            collection=settings.qdrant_tables_collection,
            count=len(points),
        )
        return len(points)
