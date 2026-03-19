"""Assemble retrieved chunks into a coherent context for LLM generation."""

from __future__ import annotations

import structlog

from src.core.query.analyzer import QueryAnalysis
from src.core.query.retriever import RetrievedChunk

logger = structlog.get_logger()

_DEFAULT_TOKEN_BUDGET = 6000
_DEFAULT_ENCODING = "cl100k_base"


def _count_tokens(text: str, encoder: object) -> int:
    """Return the token count for *text* using *encoder*."""
    return len(encoder.encode(text, disallowed_special=()))  # type: ignore[arg-type]


def _get_encoder(encoding_name: str) -> object | None:
    """Load a tiktoken encoding, returning ``None`` if unavailable."""
    try:
        import tiktoken  # type: ignore[import]

        return tiktoken.get_encoding(encoding_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("context_builder.tiktoken_unavailable", error=str(exc))
        return None


class ContextBuilder:
    """Build a context string from reranked chunks with a token budget.

    Parameters
    ----------
    max_tokens:
        Maximum number of tokens in the assembled context (default 6000).
    encoding_name:
        tiktoken encoding to use for token counting (default ``cl100k_base``).
    """

    def __init__(
        self,
        max_tokens: int = _DEFAULT_TOKEN_BUDGET,
        encoding_name: str = _DEFAULT_ENCODING,
    ) -> None:
        self.max_tokens = max_tokens
        self._encoder = _get_encoder(encoding_name)

    def _tokens(self, text: str) -> int:
        if self._encoder is not None:
            return _count_tokens(text, self._encoder)
        # Fallback: rough estimate at 4 chars per token
        return len(text) // 4

    def build(self, chunks: list[RetrievedChunk], analysis: QueryAnalysis) -> str:
        """Build a context string from *chunks*, respecting the token budget.

        Tables are rendered as Markdown with source attribution.  Heading
        hierarchy context is included where available in chunk metadata.
        Adjacent chunks from the same document are included if budget allows.
        """
        if not chunks:
            return ""

        sections: list[str] = []
        used_tokens = 0

        for i, chunk in enumerate(chunks):
            header = self._format_header(chunk, i + 1)
            body = self._format_body(chunk)
            section = f"{header}\n{body}"
            section_tokens = self._tokens(section)

            if used_tokens + section_tokens > self.max_tokens:
                # Try a truncated version
                remaining = self.max_tokens - used_tokens
                if remaining < 100:
                    break
                # Trim body to fit
                words = body.split()
                while words and self._tokens(f"{header}\n{' '.join(words)}") > remaining:
                    words = words[: int(len(words) * 0.9)]
                section = f"{header}\n{' '.join(words)}"

            sections.append(section)
            used_tokens += self._tokens(section)

            if used_tokens >= self.max_tokens:
                break

        logger.info(
            "context_builder.built",
            chunks=len(sections),
            tokens=used_tokens,
            budget=self.max_tokens,
        )
        return "\n\n---\n\n".join(sections)

    def build_with_citations(
        self, chunks: list[RetrievedChunk], analysis: QueryAnalysis
    ) -> tuple[str, list[dict]]:
        """Build context and return a list of source citation dicts.

        Returns
        -------
        tuple[str, list[dict]]
            ``(context_string, citations)`` where each citation is a dict with
            keys ``source_index``, ``title``, ``document_id``, ``page_number``,
            and ``chunk_id``.
        """
        context = self.build(chunks, analysis)
        citations = [
            {
                "source_index": i + 1,
                "title": c.metadata.get("title", ""),
                "document_id": c.document_id,
                "page_number": c.metadata.get("page_number"),
                "chunk_id": c.chunk_id,
                "chunk_type": c.chunk_type,
            }
            for i, c in enumerate(chunks)
        ]
        return context, citations

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_header(self, chunk: RetrievedChunk, index: int) -> str:
        meta = chunk.metadata
        source = meta.get("title", "Unknown Document")
        page = meta.get("page_number", "")
        section = meta.get("section_path", "")

        if chunk.chunk_type == "table":
            doc_title = source or chunk.document_id
            page_hint = f", Page {page}" if page else ""
            return f"[Table from: {doc_title}{page_hint}]"

        parts = [f"[Source {index}: {source}"]
        if section:
            parts.append(f" > {section}")
        if page:
            parts.append(f", Page {page}")
        parts.append("]")
        return "".join(parts)

    def _format_body(self, chunk: RetrievedChunk) -> str:
        if chunk.chunk_type == "table":
            # Tables are already markdown; include as-is
            return chunk.text
        # Include heading hierarchy hint if present
        headings = chunk.metadata.get("section_path", "")
        if headings:
            return f"**Section:** {headings}\n\n{chunk.text}"
        return chunk.text
