# ManualAI — Conversational Search for Digital Manuals

Production-grade RAG platform that enables users to conversationally query 400,000+ digital manuals with grounded, citation-backed answers.

---

## What It Does

ManualAI ingests massive collections of PDF and HTML manuals and makes them searchable through natural language conversation. Users ask questions in plain English and receive accurate, cited answers drawn directly from source documents — including data locked inside tables.

**Key capabilities:**

- **Conversational search** over 400K+ documents with streaming responses
- **Hybrid retrieval** combining dense vector search and sparse (BM25) retrieval for high recall and precision
- **Table-aware RAG** — tables extracted as structured units, not flattened text
- **Contextual filtering** by manufacturer, model, document type, section, and custom metadata
- **Hallucination prevention** through citation enforcement, grounding constraints, and automated faithfulness evaluation
- **Multi-tenant subscriptions** with usage-based billing
- **Enterprise-ready** — SSO, RBAC, audit logging, data isolation

---

## Architecture

```
  Next.js Frontend
        │
   API Gateway / LB
        │
  ┌─────┼─────┐
  │     │     │
 Auth  Query  Admin     ← FastAPI services
        │
  ┌─────┼─────┐
  │     │     │
Qdrant  LLM  Reranker   ← Retrieval + Generation
(hybrid)Router

  Async Ingestion Pipeline
  S3/GCS → Parse → Chunk → Embed → Index
```

Documents flow through an async ingestion pipeline that parses, chunks semantically, extracts tables, generates embeddings, and indexes into Qdrant. Queries go through analysis, hybrid retrieval, reranking, context assembly, and grounded LLM generation with citation verification.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Backend** | Python, FastAPI | API services, async-native, streaming SSE |
| **Vector DB** | Qdrant | Dense + sparse vector search, metadata filtering |
| **Relational DB** | PostgreSQL | Users, orgs, billing, document metadata, audit |
| **Cache** | Redis | Query caching, rate limiting, session management |
| **Object Storage** | AWS S3 / GCP GCS | Raw documents, processed artifacts |
| **Task Queue** | Celery + Redis | Async document ingestion pipeline |
| **LLM** | Claude / GPT-4o / Open models | Generation with model routing by query complexity |
| **Embeddings** | OpenAI text-embedding-3-large | Dense vectors (self-hosted migration path planned) |
| **Sparse Encoding** | SPLADE | BM25-style sparse vectors for hybrid retrieval |
| **Reranking** | Cohere Rerank / bge-reranker | Cross-encoder reranking for precision |
| **PDF Parsing** | PyMuPDF, pdfplumber | Text extraction, layout analysis |
| **Table Extraction** | Camelot, pdfplumber | Structured table extraction from PDFs |
| **HTML Parsing** | Trafilatura, BeautifulSoup | Content extraction, boilerplate removal |
| **Frontend** | Next.js, Tailwind CSS, shadcn/ui | Chat UI, search, admin dashboard |
| **Billing** | Stripe | Subscriptions, metered usage, customer portal |
| **Auth** | JWT, OAuth 2.0 | Authentication, SSO, API keys |
| **Containers** | Docker, ECS Fargate / Cloud Run | Deployment, auto-scaling |
| **IaC** | Terraform | Infrastructure provisioning |
| **CI/CD** | GitHub Actions | Automated testing, build, deploy |
| **Monitoring** | Prometheus, Grafana | Metrics, alerting, dashboards |
| **Evaluation** | RAGAS | Automated retrieval and generation quality scoring |

---

## Key Design Decisions

**Hybrid retrieval with RRF fusion** — Dense vectors alone miss keyword-specific queries common in technical manuals (part numbers, error codes). Combining dense + sparse with Reciprocal Rank Fusion gives consistently better results.

**Tables as first-class citizens** — Technical manuals are table-heavy (specs, torque values, error codes). Tables are extracted to structured JSON, rendered as markdown for LLM context, and summarized in natural language for embedding.

**Model routing** — Not every query needs the most expensive model. Simple lookups route to smaller models (Haiku / GPT-4o-mini), complex multi-doc reasoning routes to larger models (Sonnet / GPT-4o), and batch tasks use self-hosted open models.

**Grounded generation** — Every factual claim must cite a source. Post-generation verification checks that cited content exists in the retrieved context. Low-confidence queries trigger explicit "I don't have enough information" responses rather than hallucinated answers.

---

## Scale Targets

| Metric | Target |
|---|---|
| Documents | 400,000+ |
| Indexed chunks | ~80,000,000 |
| Query latency (P95) | < 2 seconds |
| Answer faithfulness | > 90% (RAGAS) |
| Uptime | 99.9% |
| Cost per query | < $0.05 average |

---

## Getting Started

```bash
# Clone
git clone <repo-url>
cd llm_rag_search_system

# Start local dev stack
docker compose up -d

# Install dependencies
pip install -e ".[dev]"

# Run migrations
alembic upgrade head

# Start API server
uvicorn src.main:app --reload

# Ingest sample documents
python scripts/bulk_ingest.py --input ./sample_docs/
```

---

## Project Structure

```
src/
├── main.py                  # FastAPI entry point
├── config.py                # Settings
├── api/                     # Routes, middleware, schemas
├── core/
│   ├── query/               # Query pipeline (retrieval, reranking, generation)
│   ├── ingestion/           # Document parsing, chunking, embedding
│   ├── auth/                # Authentication & authorization
│   └── billing/             # Stripe integration, usage tracking
├── db/                      # PostgreSQL, Qdrant, Redis, S3 clients
└── shared/                  # Logging, monitoring, exceptions
```

See [plan.md](plan.md) for the full architecture plan, implementation phases, and technical decisions.

---

## License

Proprietary. All rights reserved.
