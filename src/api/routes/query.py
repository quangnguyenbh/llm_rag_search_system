from fastapi import APIRouter, Depends
from pydantic import BaseModel

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    filters: dict | None = None
    conversation_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    confidence: float
    conversation_id: str


@router.post("", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Execute a RAG query against the document corpus."""
    # TODO: Wire up query pipeline
    raise NotImplementedError


@router.post("/stream")
async def query_documents_stream(request: QueryRequest):
    """Execute a RAG query with streaming response via SSE."""
    # TODO: Wire up streaming query pipeline
    raise NotImplementedError
