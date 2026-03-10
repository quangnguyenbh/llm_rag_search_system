"""Orchestrates the full RAG query pipeline: analyze → retrieve → rerank → generate."""

from dataclasses import dataclass

import structlog

from src.core.query.analyzer import QueryAnalyzer, QueryAnalysis
from src.core.query.retriever import Retriever
from src.core.query.reranker import Reranker
from src.core.query.context_builder import ContextBuilder
from src.core.query.generator import Generator
from src.core.query.citation import CitationVerifier
from src.core.query.model_router import ModelRouter

logger = structlog.get_logger()


@dataclass
class QueryResult:
    answer: str
    citations: list[dict]
    confidence: float
    model_used: str


class QueryPipeline:
    def __init__(
        self,
        analyzer: QueryAnalyzer,
        retriever: Retriever,
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
        """Run the full RAG pipeline: analyze → retrieve → rerank → build context → generate → verify."""
        # 1. Analyze query
        analysis = await self.analyzer.analyze(question)
        logger.info("query.analyzed", intent=analysis.intent, complexity=analysis.complexity)

        # 2. Retrieve relevant chunks
        candidates = await self.retriever.search(
            query=question,
            filters=filters,
            top_k=20,
        )

        # 3. Rerank for precision
        reranked = await self.reranker.rerank(question, candidates, top_k=8)

        # 4. Build context string
        context = self.context_builder.build(reranked, analysis)

        # 5. Route to model
        model = self.model_router.select(analysis)
        logger.info("query.model_selected", model=model)

        # 6. Generate answer
        response = await self.generator.generate(
            question=question,
            context=context,
            model=model,
        )

        # 7. Verify citations
        verified = self.citation_verifier.verify(response, reranked)

        return QueryResult(
            answer=verified.answer,
            citations=verified.citations,
            confidence=verified.confidence,
            model_used=model,
        )

    async def search_only(self, question: str, filters: dict | None = None, top_k: int = 10) -> list[dict]:
        """Vector search only — returns ranked chunks without LLM generation."""
        candidates = await self.retriever.search(query=question, filters=filters, top_k=top_k)
        reranked = await self.reranker.rerank(question, candidates, top_k=top_k)
        return [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "text": c.text,
                "score": c.score,
                "metadata": c.metadata,
            }
            for c in reranked
        ]

    async def stream(self, question: str, filters: dict | None = None):
        """Stream RAG response via generator."""
        analysis = await self.analyzer.analyze(question)
        candidates = await self.retriever.search(query=question, filters=filters, top_k=20)
        reranked = await self.reranker.rerank(question, candidates, top_k=8)
        context = self.context_builder.build(reranked, analysis)
        model = self.model_router.select(analysis)

        # Yield source metadata first
        sources = [
            {
                "source_index": i + 1,
                "title": c.metadata.get("title", ""),
                "document_id": c.document_id,
                "page_number": c.metadata.get("page_number"),
                "score": c.score,
            }
            for i, c in enumerate(reranked)
        ]

        yield {"type": "sources", "data": sources}

        # Stream answer tokens
        async for token in self.generator.generate_stream(question, context, model):
            yield {"type": "token", "data": token}

        yield {"type": "done", "data": ""}
