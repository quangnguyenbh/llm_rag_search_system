# ADR-002: Embedding Model and Vector Indexing Strategy

**Status:** Accepted  
**Date:** 2025-01-20  
**Deciders:** Core team  

## Context

After chunking (ADR-001), each text chunk must be converted to a dense vector embedding and stored in a vector database for similarity search. The choices of embedding model and vector store directly impact retrieval accuracy, latency, cost, and operational complexity.

Our constraints:
- **400K+ documents** → millions of chunks; cost matters
- **AWS-first infrastructure** in `ap-southeast-1` (Singapore) for data residency
- **Managed services preferred** to minimise operational burden
- **Configurable provider** so the team can swap models later without rewriting pipeline code

## Decision

### Embedding Model: Amazon Titan Embed Text V2

We use **`amazon.titan-embed-text-v2:0`** via AWS Bedrock in `ap-southeast-1`.

| Property | Value |
|----------|-------|
| Model ID | `amazon.titan-embed-text-v2:0` |
| Provider | AWS Bedrock |
| Region | `ap-southeast-1` (Singapore) |
| Dimensions | 1024 |
| Max input tokens | 8192 |
| Cost | $0.00002 / 1K input tokens |
| Normalisation | Built-in (L2 normalised) |

**Why Titan V2 over alternatives:**

| Model | Dimensions | Cost / 1K tokens | Notes |
|-------|-----------|-------------------|-------|
| Titan Embed V2 | 1024 | $0.00002 | Cheapest Bedrock option, native normalisation, configurable dimensions |
| Cohere Embed English v3 | 1024 | $0.00010 | 5× more expensive, slightly higher MTEB scores |
| OpenAI text-embedding-3-large | 3072 | $0.00013 | External API, higher latency from Singapore, 6.5× cost |

At projected scale (~50M chunks), Titan V2 costs ~$20 for a full re-embed vs ~$130 for Cohere or ~$650 for OpenAI.

### Vector Store: Qdrant Cloud

We use **Qdrant Cloud** (managed) rather than self-hosted Qdrant in Docker.

| Property | Value |
|----------|-------|
| Client library | `qdrant-client` (Python) |
| Connection | HTTPS URL + API key |
| Collection | `manual_chunks` |
| Distance metric | Cosine |
| Vector size | 1024 |

**Why Qdrant Cloud:**
- Fully managed — no Docker/Kubernetes ops overhead
- Horizontal scaling and backups built in
- Python SDK with async support
- Free tier sufficient for development; pay-as-you-go for production
- Supports filtering by `document_id` for scoped search and deletion

**Why not alternatives:**
- **Pinecone:** More expensive at scale, vendor lock-in on indexing configuration
- **Weaviate Cloud:** Heavier schema model, more complex for our simple payload structure
- **pgvector (PostgreSQL):** Already in our stack but ANN index performance degrades at >1M vectors without careful tuning; separate vector DB is more predictable
- **Local Docker Qdrant:** Works for dev but adds operational burden for production (backups, scaling, monitoring)

### Pipeline Architecture

```
PDF file
  → parse_pdf()                    # PyMuPDF structured blocks
  → extract_metadata()             # title, section map
  → SemanticChunker.chunk()        # ADR-001 strategy
  → BatchEmbedder.embed_batch()    # Bedrock Titan V2
  → upsert_chunks()                # Qdrant Cloud
  → IngestionResult                # summary dataclass
```

### Embedding: Batching and Retry

- **Batch size:** 256 chunks per API batch (configurable via `EMBEDDING_BATCH_SIZE`)
- **Titan V2 processes one text per invoke_model call**, so we parallelise within each batch using `asyncio.to_thread` / `run_in_executor`
- **Retry:** Exponential backoff (2^attempt seconds), max 3 retries per batch
- **Token counting:** Reuses `tiktoken` `cl100k_base` from chunker for consistent budget tracking

### Qdrant Payload Schema

Each point stored in Qdrant carries:

```json
{
  "document_id": "sha256-hex-of-source-path",
  "text": "chunk text with contextual header",
  "page_number": 1,
  "section_path": "Chapter 3 > Safety",
  "heading_hierarchy": ["Chapter 3", "Safety"],
  "token_count": 487,
  "title": "BTTM-803 User Manual",
  "source_file": "/data/manuals/bttm-803.pdf"
}
```

This enables:
- **Filtered search** by `document_id` (search within one manual)
- **Full-text display** of retrieved chunks (the `text` field)
- **Attribution** via `source_file`, `page_number`, `section_path`
- **Deletion** of all chunks for a document via `document_id` filter

### Configuration

All parameters are configurable via environment variables (pydantic-settings):

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBEDDING_PROVIDER` | `bedrock` | Provider selector (bedrock / openai) |
| `EMBEDDING_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Model identifier |
| `EMBEDDING_DIMENSIONS` | `1024` | Vector dimensionality |
| `EMBEDDING_BATCH_SIZE` | `256` | Chunks per embedding batch |
| `EMBEDDING_MAX_RETRIES` | `3` | Max retry attempts per batch |
| `AWS_BEDROCK_REGION` | `ap-southeast-1` | Bedrock endpoint region |
| `QDRANT_URL` | (required) | Qdrant Cloud HTTPS endpoint |
| `QDRANT_API_KEY` | (required) | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | `manual_chunks` | Collection name |

## Alternatives Considered

### Self-hosted embedding (sentence-transformers)
Running `all-MiniLM-L6-v2` or `bge-large-en-v1.5` on GPU. Lower per-token cost but requires GPU infrastructure in Singapore, model serving ops, and version management. Bedrock eliminates this entirely.

### OpenAI Embeddings API
Higher quality on some benchmarks but 6.5× cost, external API with cross-region latency, and data leaves AWS — problematic for data residency.

### ChromaDB
Simpler API but designed for single-node dev use. No managed cloud offering, limited filtering, no production scaling story.

## Consequences

### Positive
- Lowest embedding cost on Bedrock ($0.00002/1K tokens)
- No GPU infrastructure to manage — fully serverless
- Data stays within AWS ap-southeast-1
- Qdrant Cloud handles scaling, backups, and monitoring
- Provider-agnostic design allows swapping to Cohere, OpenAI, or local models later
- Payload schema enables both global and per-document search

### Negative
- Titan V2 scores lower than Cohere or OpenAI on MTEB benchmarks (~63 vs ~66 avg)
- One-text-per-call Bedrock API means higher round-trip overhead vs batch APIs
- Qdrant Cloud adds a SaaS dependency outside AWS

### Risks
- Bedrock throttling at high concurrency — mitigated by batch size limits and retry logic
- Qdrant Cloud availability — mitigated by collection snapshots and the option to fall back to self-hosted
- Embedding model quality may need upgrade for specialised technical domains — configurable provider pattern makes this a config change, not a code change

## Implementation

- Embedder: `src/core/ingestion/embedder.py` — `BatchEmbedder` class
- Qdrant client: `src/db/vector/qdrant_client.py` — connection, upsert, search, delete
- Pipeline: `src/core/ingestion/pipeline.py` — `IngestionPipeline` class
- Config: `src/config.py` — pydantic-settings with env var overrides
- Tests: `tests/unit/test_embedder.py`, `tests/unit/test_qdrant_client.py`, `tests/integration/test_ingestion_pipeline.py`, `tests/e2e/test_ingest_and_search.py`
