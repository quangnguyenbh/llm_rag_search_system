import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.core.ingestion.embedder import BatchEmbedder
from src.core.query.analyzer import QueryAnalyzer
from src.core.query.citation import CitationVerifier
from src.core.query.context_builder import ContextBuilder
from src.core.query.generator import Generator
from src.core.query.model_router import ModelRouter
from src.core.query.pipeline import QueryPipeline
from src.core.query.reranker import Reranker
from src.core.query.retriever import Retriever
from src.db.vector.qdrant_client import get_qdrant_client

router = APIRouter()


def _build_pipeline() -> QueryPipeline:
    qdrant = get_qdrant_client()
    embedder = BatchEmbedder()
    return QueryPipeline(
        analyzer=QueryAnalyzer(),
        retriever=Retriever(qdrant_client=qdrant, embedder=embedder),
        reranker=Reranker(),
        context_builder=ContextBuilder(),
        generator=Generator(),
        citation_verifier=CitationVerifier(),
        model_router=ModelRouter(),
    )


class QueryRequest(BaseModel):
    question: str
    filters: dict | None = None


class SearchRequest(BaseModel):
    question: str
    filters: dict | None = None
    top_k: int = 10


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    confidence: float
    model_used: str


@router.post("", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Execute a RAG query: retrieve relevant chunks → synthesize answer with Claude."""
    pipeline = _build_pipeline()
    result = await pipeline.execute(
        question=request.question,
        filters=request.filters,
    )
    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        confidence=result.confidence,
        model_used=result.model_used,
    )


@router.post("/search")
async def search_documents(request: SearchRequest):
    """Vector search only — returns ranked chunks without LLM generation."""
    pipeline = _build_pipeline()
    chunks = await pipeline.search_only(
        question=request.question,
        filters=request.filters,
        top_k=request.top_k,
    )
    return {"results": chunks, "count": len(chunks)}


@router.post("/stream")
async def query_documents_stream(request: QueryRequest):
    """Execute a RAG query with streaming response via SSE."""
    pipeline = _build_pipeline()

    async def event_generator():
        async for event in pipeline.stream(
            question=request.question,
            filters=request.filters,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
