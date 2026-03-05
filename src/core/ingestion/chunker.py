"""Semantic chunking with section awareness and heading hierarchy."""

from dataclasses import dataclass, field

from src.core.ingestion.parsers.base import ParsedDocument


@dataclass
class Chunk:
    chunk_id: str = ""
    text: str = ""
    document_id: str = ""
    page_number: int | None = None
    section_path: str = ""
    heading_hierarchy: list[str] = field(default_factory=list)
    token_count: int = 0


class SemanticChunker:
    def __init__(
        self,
        target_size: int = 512,
        overlap: int = 64,
        min_size: int = 100,
        max_size: int = 1024,
    ):
        self.target_size = target_size
        self.overlap = overlap
        self.min_size = min_size
        self.max_size = max_size

    def chunk(self, document: ParsedDocument, metadata: dict) -> list[Chunk]:
        """Split a parsed document into semantic chunks."""
        # TODO: Split by structural boundaries (headings, sections)
        # TODO: Within sections, split by paragraph boundaries
        # TODO: Merge small consecutive chunks from same section
        # TODO: Apply overlap at boundaries
        # TODO: Attach parent heading hierarchy
        # TODO: Prepend contextual header to each chunk
        return []
