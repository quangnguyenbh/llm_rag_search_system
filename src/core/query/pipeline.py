"""Orchestrates the full RAG query pipeline: analyze → retrieve → rerank → generate."""

from dataclasses import dataclass

from src.core.query.analyzer import QueryAnalyzer, QueryAnalysis
from src.core.query.retriever import HybridRetriever
from src.core.query.reranker import Reranker
from src.core.query.context_builder import ContextBuilder
from src.core.query.generator import Generator
from src.core.query.citation import CitationVerifier
from src.core.query.model_router import ModelRouter


@dataclass
class QueryResult:
    answer: str
    citations: list[dict]
    confidence: float
    conversation_id: str
    model_used: str


class QueryPipeline:
    def __init__(
        self,
        analyzer: QueryAnalyzer,
        retriever: HybridRetriever,
        reranker: Reranker,
        context_builder: ContextBuilder,
        generator: Generator,
        citation_verifier: CitationVerifier,
        model_router: ModelRouter,
    ):
        self.analyzer = analyzer
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.generator = generator
        self.citation_verifier = citation_verifier
        self.model_router = model_router

    async def execute(self, question: str, filters: dict | None = None) -> QueryResult:
        # 1. Analyze query
        analysis = await self.analyzer.analyze(question)

        # 2. Merge explicit filters with extracted entities
        merged_filters = self._merge_filters(filters, analysis)

        # 3. Hybrid retrieval (dense + sparse)
        candidates = await self.retriever.search(
            query=question,
            filters=merged_filters,
            top_k=50,
        )

        # 4. Rerank
        reranked = await self.reranker.rerank(question, candidates, top_k=8)

        # 5. Build context
        context = self.context_builder.build(reranked, analysis)

        # 6. Route to model
        model = self.model_router.select(analysis)

        # 7. Generate answer
        response = await self.generator.generate(
            question=question,
            context=context,
            model=model,
        )

        # 8. Verify citations
        verified = self.citation_verifier.verify(response, reranked)

        return QueryResult(
            answer=verified.answer,
            citations=verified.citations,
            confidence=verified.confidence,
            conversation_id="",  # TODO: conversation tracking
            model_used=model,
        )

    def _merge_filters(self, explicit: dict | None, analysis: QueryAnalysis) -> dict:
        merged = explicit or {}
        if analysis.entities:
            for key, value in analysis.entities.items():
                if key not in merged:
                    merged[key] = value
        return merged
