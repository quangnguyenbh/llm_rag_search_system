"""Citation extraction and verification against retrieved context."""

import re
from dataclasses import dataclass, field

from src.core.query.generator import GenerationResult
from src.core.query.retriever import RetrievedChunk


@dataclass
class VerifiedResponse:
    answer: str
    citations: list[dict] = field(default_factory=list)
    confidence: float = 0.0


class CitationVerifier:
    def verify(
        self, response: GenerationResult, chunks: list[RetrievedChunk]
    ) -> VerifiedResponse:
        """Verify that citations in the generated answer map to actual retrieved chunks."""
        # Parse [Source N] references from answer
        cited_indices = set()
        for match in re.finditer(r"\[Source\s+(\d+)\]", response.answer):
            cited_indices.add(int(match.group(1)))

        # Map cited indices to actual chunks
        citations = []
        valid_count = 0
        for idx in sorted(cited_indices):
            if 1 <= idx <= len(chunks):
                chunk = chunks[idx - 1]
                citations.append({
                    "source_index": idx,
                    "document_id": chunk.document_id,
                    "title": chunk.metadata.get("title", ""),
                    "page_number": chunk.metadata.get("page_number"),
                    "section_path": chunk.metadata.get("section_path", ""),
                    "score": chunk.score,
                    "text_preview": chunk.text[:200],
                })
                valid_count += 1

        # Confidence: based on retrieval scores and citation coverage
        if not chunks:
            confidence = 0.0
        else:
            avg_score = sum(c.score for c in chunks) / len(chunks)
            citation_ratio = valid_count / max(len(cited_indices), 1)
            confidence = round(min(avg_score * citation_ratio, 1.0), 3)

        return VerifiedResponse(
            answer=response.answer,
            citations=citations,
            confidence=confidence,
        )
