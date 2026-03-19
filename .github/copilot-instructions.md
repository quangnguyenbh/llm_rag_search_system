# GitHub Copilot Instructions — ManualAI RAG Search System

This document provides repository-wide guidelines, conventions, and context for GitHub Copilot when working on the ManualAI codebase.

## Project Overview

ManualAI is a production-grade RAG (Retrieval-Augmented Generation) platform that enables conversational search over 400,000+ digital manuals with grounded, citation-backed answers. The system handles massive document collections through an async ingestion pipeline and provides sub-2-second query responses with 90%+ answer faithfulness.

**Key Documentation**:
- [README.md](../README.md) — Project overview and quick start
- [plan.md](../plan.md) — Detailed architecture, implementation phases, and technical decisions
- [docs/how-the-rag-query-works.md](../docs/how-the-rag-query-works.md) — RAG query pipeline documentation
- [docs/system_architecture.md](../docs/system_architecture.md) — System architecture details

## Architecture Context

### Core Components
- **Backend**: FastAPI (Python 3.11+) with async/await patterns
- **Vector DB**: Qdrant for hybrid retrieval (dense + sparse vectors)
- **Relational DB**: PostgreSQL for metadata, users, billing, audit logs
- **Cache**: Redis for query caching and rate limiting
- **Task Queue**: Celery + Redis for async document ingestion
- **Object Storage**: AWS S3 / GCP GCS for documents and artifacts

### Key Design Principles
1. **Hybrid Retrieval**: Always combine dense vector search with sparse (BM25) retrieval using Reciprocal Rank Fusion
2. **Tables as First-Class Citizens**: Extract tables to structured JSON, render as markdown for LLM context
3. **Grounded Generation**: Every factual claim must cite a source; enforce citation verification
4. **Model Routing**: Route queries to appropriate models based on complexity (Haiku/GPT-4o-mini for simple, Sonnet/GPT-4o for complex)
5. **Async-First**: Use asyncio patterns throughout for I/O-bound operations

## Coding Standards

### Python Style Guide
- **Python Version**: 3.11+
- **Style**: Follow PEP 8 conventions
- **Line Length**: 100 characters (configured in pyproject.toml)
- **Linter**: Ruff with rules E, F, W, I, N, UP, S, B
- **Type Checker**: mypy with `strict = true`
- **Formatter**: Ruff (automatically applied)

### Type Hints
- **Required**: All function signatures must have complete type hints
- **Use modern syntax**: Use `list[str]` instead of `List[str]` (Python 3.11+)
- **Avoid `Any`**: Use specific types or protocols instead of `Any` when possible
- **Pydantic models**: Use for API schemas, configuration, and data validation

Example:
```python
from pydantic import BaseModel, Field

async def process_query(
    query: str,
    filters: dict[str, str] | None = None,
    limit: int = 10
) -> list[SearchResult]:
    """Process a search query with optional filters."""
    ...
```

### Error Handling
- **Use specific exceptions**: Create custom exceptions in `src/shared/exceptions.py`
- **Structured logging**: Use `structlog` for all logging (configured in `src/shared/logging.py`)
- **Async context managers**: Use `async with` for resource cleanup
- **Graceful degradation**: Always handle external service failures with fallbacks

Example:
```python
import structlog
from src.shared.exceptions import RetrievalError

logger = structlog.get_logger(__name__)

async def retrieve_documents(query: str) -> list[Document]:
    try:
        results = await vector_db.search(query)
        logger.info("documents_retrieved", count=len(results))
        return results
    except VectorDBError as e:
        logger.error("retrieval_failed", error=str(e))
        raise RetrievalError("Failed to retrieve documents") from e
```

### Testing Requirements
- **Test Framework**: pytest with pytest-asyncio
- **Coverage Target**: Aim for 80%+ coverage on core modules
- **Mock External Services**: Use `httpx.AsyncClient` mocking, mock LLM calls
- **Test Organization**: Place tests in `tests/` mirroring `src/` structure
- **Async Tests**: Use `async def test_*` with `pytest-asyncio`

Example:
```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_search_endpoint(client: AsyncClient):
    response = await client.post(
        "/api/v1/search",
        json={"query": "How to reset the device?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) > 0
```

## RAG-Specific Patterns

### Document Chunking
- **Semantic chunking**: Preserve paragraph/section boundaries
- **Target size**: 512-1024 tokens per chunk (configurable)
- **Overlap**: 10-20% overlap between chunks for context continuity
- **Metadata**: Always attach document_id, page_number, section_title

### Retrieval Pipeline
1. **Query Analysis**: Extract keywords, detect query type (factual, comparison, troubleshooting)
2. **Hybrid Search**: Run dense (vector) + sparse (BM25) in parallel
3. **Fusion**: Merge results using Reciprocal Rank Fusion (RRF)
4. **Reranking**: Apply cross-encoder reranking for top-N results
5. **Context Assembly**: Format retrieved chunks with citations

### Citation Format
- **Always include**: document_title, page_number, chunk_id
- **Inline format**: `[Source: Manual_Name p.42]`
- **Verification**: Post-generation check that cited content exists in context

### LLM Integration
- **Streaming**: Use SSE (Server-Sent Events) for response streaming
- **Context limits**: Monitor token counts, truncate if needed
- **Prompt templates**: Store in `src/core/query/prompts.py`
- **Model selection**: Route via `src/core/query/model_router.py`

Example prompt pattern:
```python
ANSWER_PROMPT = """You are an expert technical assistant specializing in equipment manuals.

Context:
{context}

User Question: {query}

Instructions:
1. Answer ONLY based on the provided context
2. Cite sources using [Source: Title p.XX] format
3. If unsure, say "I don't have enough information"
4. Be precise and technical when discussing specifications

Answer:"""
```

## API Design Conventions

### Endpoint Structure
- **Versioning**: All endpoints under `/api/v1/`
- **RESTful**: Use standard HTTP methods (GET, POST, PUT, DELETE)
- **Path parameters**: For resource IDs (e.g., `/documents/{document_id}`)
- **Query parameters**: For filtering, pagination, sorting
- **Request/Response**: Use Pydantic models for validation

### Response Format
```python
from pydantic import BaseModel

class APIResponse[T](BaseModel):
    success: bool
    data: T | None = None
    error: str | None = None
    metadata: dict[str, str] | None = None
```

### Authentication
- **JWT tokens**: Use Bearer token authentication
- **API keys**: Support API key auth for programmatic access
- **Middleware**: Apply in `src/api/middleware/auth.py`

## Database Patterns

### PostgreSQL (via SQLAlchemy)
- **Async ORM**: Use `AsyncSession` and async queries
- **Migrations**: Alembic migrations in `alembic/versions/`
- **Naming**: Use snake_case for tables and columns
- **Indexes**: Add indexes for frequently queried columns
- **Soft deletes**: Use `deleted_at` timestamp instead of hard deletes

Example:
```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

async def get_user_by_email(session: AsyncSession, email: str) -> User | None:
    stmt = select(User).where(User.email == email, User.deleted_at.is_(None))
    result = await session.execute(stmt)
    return result.scalar_one_or_none()
```

### Qdrant (Vector DB)
- **Collection naming**: Use environment prefixes (e.g., `prod_manuals`, `dev_manuals`)
- **Hybrid search**: Always use both dense and sparse vectors
- **Payload**: Include all searchable metadata in payload
- **Batch operations**: Use batching for bulk inserts (1000+ vectors)

### Redis
- **Key naming**: Use colons for namespaces (e.g., `cache:query:{hash}`)
- **Expiration**: Always set TTL on cache keys
- **Connection pooling**: Use connection pool from `src/db/redis.py`

## Security Best Practices

### Input Validation
- **Pydantic validation**: Validate all API inputs with Pydantic models
- **Length limits**: Enforce max lengths on text inputs (queries, user content)
- **Sanitization**: Sanitize user inputs before logging or database insertion
- **LLM injection**: Be cautious with user content in prompts; use delimiters

### Secrets Management
- **Environment variables**: All secrets via environment variables
- **No hardcoding**: Never commit API keys, passwords, or tokens
- **AWS Secrets Manager**: Production secrets from AWS Secrets Manager
- **Local development**: Use `.env` file (excluded from git)

### Data Privacy
- **PII handling**: Log only non-PII data
- **Query privacy**: Hash or anonymize queries in logs
- **Multi-tenancy**: Enforce tenant isolation in all queries

## Performance Considerations

### Scale Targets
- **Documents**: 400,000+ manuals
- **Chunks**: ~80,000,000 indexed chunks
- **Query Latency**: < 2 seconds (P95)
- **Cost**: < $0.05 per query average

### Optimization Guidelines
- **Async I/O**: Use async for all I/O-bound operations (DB, API calls, LLM)
- **Connection pooling**: Reuse connections for PostgreSQL, Redis, Qdrant
- **Caching**: Cache query results, embeddings, and frequently accessed data
- **Batch operations**: Batch database writes and vector inserts
- **Lazy loading**: Load large documents and embeddings only when needed

## Common Patterns

### Dependency Injection (FastAPI)
```python
from fastapi import Depends
from src.db.postgres import get_session

async def get_current_user(
    session: AsyncSession = Depends(get_session),
    token: str = Depends(oauth2_scheme)
) -> User:
    # Validate token and fetch user
    ...
```

### Background Tasks
```python
from fastapi import BackgroundTasks

@router.post("/ingest")
async def ingest_document(
    file: UploadFile,
    background_tasks: BackgroundTasks
):
    # Save file, then process in background
    background_tasks.add_task(process_document, file_path)
    return {"status": "processing"}
```

### Streaming Responses
```python
from fastapi.responses import StreamingResponse

async def generate_answer(query: str):
    async for chunk in llm_client.stream(query):
        yield f"data: {chunk}\n\n"

@router.post("/search/stream")
async def search_stream(request: SearchRequest):
    return StreamingResponse(
        generate_answer(request.query),
        media_type="text/event-stream"
    )
```

## Configuration Management

### Settings (Pydantic Settings)
- **File**: `src/config.py`
- **Environment-based**: Use `.env` files for local, environment variables for production
- **Type-safe**: All settings as Pydantic `BaseSettings` with types
- **Nested configs**: Group related settings (e.g., `DatabaseSettings`, `QdrantSettings`)

Example:
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str
    qdrant_url: str
    openai_api_key: str
    log_level: str = "INFO"
```

## Project Structure

```
src/
├── main.py                  # FastAPI app entry point
├── config.py                # Settings and configuration
├── api/                     # API routes and middleware
│   ├── v1/                  # API version 1 endpoints
│   ├── middleware/          # Auth, CORS, rate limiting
│   └── schemas/             # Request/response Pydantic models
├── core/
│   ├── query/               # Query pipeline (retrieval, reranking, generation)
│   ├── ingestion/           # Document parsing, chunking, embedding
│   ├── auth/                # Authentication & authorization
│   └── billing/             # Stripe integration, usage tracking
├── db/                      # Database clients
│   ├── postgres.py          # SQLAlchemy async session
│   ├── qdrant.py            # Qdrant client
│   ├── redis.py             # Redis client
│   └── s3.py                # S3 client
└── shared/                  # Shared utilities
    ├── logging.py           # Structured logging setup
    ├── monitoring.py        # Metrics and monitoring
    └── exceptions.py        # Custom exception classes
```

## When Working on Tasks

### Before Making Changes
1. **Read the architecture docs**: Review `plan.md` for context
2. **Check existing patterns**: Look for similar implementations in the codebase
3. **Understand the scope**: Ensure changes align with scale targets (400K+ docs, sub-2s queries)
4. **Consider testing**: Plan for unit and integration tests

### Implementation Checklist
- [ ] Code follows PEP 8 and passes Ruff linting
- [ ] Type hints are complete and pass mypy strict checks
- [ ] Error handling with structured logging
- [ ] Unit tests added or updated
- [ ] Documentation updated (docstrings, README, plan.md if needed)
- [ ] No hardcoded secrets or credentials
- [ ] Performance impact considered (especially for hot paths)
- [ ] Security implications reviewed (input validation, auth, etc.)

### After Making Changes
1. **Run linting**: `ruff check .` and `ruff format .`
2. **Type checking**: `mypy src/`
3. **Run tests**: `pytest tests/`
4. **Manual testing**: Test the feature end-to-end if applicable
5. **Update docs**: Ensure README and plan.md reflect changes

## Key Repositories and References

- **FastAPI**: https://fastapi.tiangolo.com/
- **Qdrant**: https://qdrant.tech/documentation/
- **Pydantic**: https://docs.pydantic.dev/
- **SQLAlchemy Async**: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
- **Structlog**: https://www.structlog.org/

## Support and Questions

For architecture decisions or clarifications, refer to:
- `plan.md` — Comprehensive architecture and implementation plan
- `docs/` — Detailed documentation on specific subsystems
- `.github/agents/pr-reviewer.md` — PR review guidelines and quality standards

---

**Remember**: This is a production-grade system handling 400K+ documents with strict latency, faithfulness, and uptime targets. Every change should consider scalability, performance, security, and maintainability.
