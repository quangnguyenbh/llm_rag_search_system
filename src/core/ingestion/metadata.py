"""Metadata extraction: document attributes, NER for entities."""

from src.core.ingestion.parsers.base import ParsedDocument


class MetadataExtractor:
    def extract(self, document: ParsedDocument, user_metadata: dict | None = None) -> dict:
        """Extract metadata from a parsed document, merged with user-provided metadata."""
        metadata = {
            "title": document.title,
            "format": document.format,
            "page_count": len(document.pages),
        }

        # TODO: NER for manufacturer, product model, part numbers
        # TODO: Document type classification (user manual, service manual, etc.)
        # TODO: Language detection
        # TODO: Date extraction

        if user_metadata:
            metadata.update(user_metadata)

        return metadata
