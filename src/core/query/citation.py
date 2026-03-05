"""Citation extraction and verification against retrieved context."""

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
        # TODO: Parse [Source N] references from answer
        # TODO: Verify each citation maps to a real chunk
        # TODO: Calculate confidence based on citation coverage and retrieval scores
        return VerifiedResponse(
            answer=response.answer,
            citations=[],
            confidence=0.0,
        )
