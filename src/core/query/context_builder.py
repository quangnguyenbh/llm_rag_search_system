"""Assemble retrieved chunks into a coherent context for LLM generation."""

from src.core.query.analyzer import QueryAnalysis
from src.core.query.retriever import RetrievedChunk


class ContextBuilder:
    def __init__(self, max_tokens: int = 12000):
        self.max_tokens = max_tokens

    def build(self, chunks: list[RetrievedChunk], analysis: QueryAnalysis) -> str:
        """Build a context string from reranked chunks, respecting token budget."""
        if not chunks:
            return ""

        sections: list[str] = []
        for i, chunk in enumerate(chunks):
            header = self._format_header(chunk, i + 1)
            sections.append(f"{header}\n{chunk.text}")

        context = "\n\n---\n\n".join(sections)
        # TODO: Token counting and truncation
        return context

    def _format_header(self, chunk: RetrievedChunk, index: int) -> str:
        meta = chunk.metadata
        source = meta.get("title", "Unknown Document")
        page = meta.get("page_number", "")
        section = meta.get("section_path", "")

        parts = [f"[Source {index}: {source}"]
        if section:
            parts.append(f" > {section}")
        if page:
            parts.append(f", Page {page}")
        parts.append("]")
        return "".join(parts)
