# RAG Search System — Architecture & Implementation Plan

> Production-grade AI platform for conversational querying of 400,000+ digital manuals.

---

## 1. System Overview

### Core Capabilities

| Capability | Description |
|---|---|
| **Document Ingestion** | Batch + streaming ingestion of PDF and HTML manuals with table extraction |
| **Hybrid Retrieval** | Dense vector search + BM25 sparse retrieval with reciprocal rank fusion |
| **Table-Aware RAG** | Tables extracted as atomic structured units, separately indexed and retrievable |
| **Contextual Filtering** | Metadata-driven filter layer (manufacturer, model, year, doc type, section) |
| **Grounded Generation** | Citation-backed answers with hallucination guardrails |
| **Multi-Tenant Subscriptions** | Stripe-integrated usage-based and tiered subscription billing |
| **Enterprise Integration** | SSO, API keys, webhooks, audit logging |

### Non-Functional Requirements

- **Scale**: 400K+ documents, 50M+ chunks, sub-2s P95 query latency
- **Cost**: Optimized embedding pipeline, tiered storage, model routing by query complexity
- **Privacy**: No customer data used for model training, data residency controls, encryption at rest/transit
- **Availability**: 99.9% uptime target, graceful degradation under load

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js)                           │
│  Chat UI · Search · Doc Browser · Admin Dashboard · Billing Portal  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTPS / WebSocket
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     API GATEWAY / LOAD BALANCER                      │
│              (AWS ALB / GCP Cloud Load Balancing)                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌──────────────┐   ┌───────────────────┐   ┌──────────────────┐
│  Auth Service │   │   Query Service   │   │  Admin Service   │
│  (FastAPI)    │   │   (FastAPI)       │   │  (FastAPI)       │
│              │   │                   │   │                  │
│ - JWT/OAuth  │   │ - Query routing   │   │ - Doc management │
│ - API keys   │   │ - Hybrid retrieval│   │ - User/org mgmt  │
│ - RBAC       │   │ - Reranking       │   │ - Usage analytics│
│ - SSO        │   │ - LLM generation  │   │ - Billing/Stripe │
└──────────────┘   │ - Citation attach │   └──────────────────┘
                   │ - Streaming resp  │
                   └────────┬──────────┘
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
   ┌──────────────┐ ┌─────────────┐ ┌──────────────┐
   │  Vector DB   │ │ Search Index│ │   LLM Layer  │
   │  (Qdrant)    │ │ (OpenSearch)│ │  (Router)    │
   │              │ │             │ │              │
   │ Dense vectors│ │ BM25 + meta │ │ Claude/GPT/  │
   │ + metadata   │ │ filters     │ │ Open models  │
   └──────────────┘ └─────────────┘ └──────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE (Async)                       │
│                                                                     │
│  Object Store ──▶ Parser ──▶ Chunker ──▶ Embedder ──▶ Indexer      │
│  (S3 / GCS)      (PDF/HTML)  (Semantic)  (Batch)     (Vector+BM25) │
│                      │                                              │
│                      ├──▶ Table Extractor ──▶ Table Index           │
│                      └──▶ Metadata Extractor                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     DATA / INFRA LAYER                               │
│                                                                     │
│  PostgreSQL (users, orgs, billing, audit)                           │
│  Redis (session cache, rate limiting, query cache)                  │
│  S3/GCS (raw documents, processed artifacts)                       │
│  Celery + Redis/SQS (task queue for ingestion)                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Tech Stack Decisions

### 3.1 Backend Framework

| Choice | Rationale |
|---|---|
| **FastAPI** | Async-native, OpenAPI auto-docs, Pydantic validation, streaming SSE support |

Structure as a **modular monolith** initially (single deployable, domain-separated modules), with clear boundaries to extract microservices later if needed.

### 3.2 Vector Database — Recommendation: **Qdrant**

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **Qdrant** | Native hybrid search (dense+sparse), payload filtering, horizontal scaling, gRPC, on-prem/cloud | Smaller ecosystem than Pinecone | **Recommended** |
| Weaviate | Hybrid search, GraphQL, modules ecosystem | Heavier resource footprint | Good alternative |
| Pgvector | No extra infra, SQL familiarity | Poor at 50M+ scale, no native sparse | Not for this scale |
| Pinecone | Fully managed, fast | Vendor lock-in, cost at scale, no self-host | Backup option |
| Milvus | High performance, GPU support | Complex operations, heavy infra | Overly complex |

**Why Qdrant**: Native sparse vector support (built-in BM25-style retrieval without a separate search engine), payload-based filtering, snapshot/backup, and can run self-hosted or managed cloud. At 50M+ vectors it remains performant with sharding.

> **Alternative path**: If full-text search requirements grow beyond retrieval (faceted search, aggregations, analytics), add **OpenSearch** as the BM25 layer instead of Qdrant's sparse vectors.

### 3.3 BM25 / Sparse Retrieval

| Approach | When to use |
|---|---|
| **Qdrant sparse vectors** (SPLADE/BM25 encoded) | Default — keeps infra simple, single query target |
| **OpenSearch** | If you need faceted search, aggregations, or advanced text analytics beyond retrieval |

Start with Qdrant sparse vectors. Add OpenSearch only if needed.

### 3.4 LLM Layer — Model Router

No single model for all queries. Implement a **model router**:

| Query Type | Model | Rationale |
|---|---|---|
| Simple factual lookup | **Claude 3.5 Haiku / GPT-4o-mini** | Low cost, fast, sufficient quality |
| Complex multi-doc reasoning | **Claude 4 Sonnet / GPT-4o** | Better reasoning, citation accuracy |
| Table interpretation | **Claude 4 Sonnet** | Strong structured data understanding |
| Summarization (batch) | **Open model (Llama 3.x, Mistral)** | Cost optimization for offline tasks |
| Fallback / cost ceiling | **Open model self-hosted** | Cap spend under traffic spikes |

**Router logic** (v1 simple, v2 ML-based):
- v1: Rule-based on query length, detected complexity keywords, user tier
- v2: Lightweight classifier trained on query patterns

### 3.5 Embedding Model

| Model | Dimensions | Use Case |
|---|---|---|
| **OpenAI text-embedding-3-large** | 3072 (or 1536 with `dimensions` param) | Primary dense embeddings |
| **SPLADE v3** | Sparse | Sparse vector encoding for hybrid search |
| **Open alternative**: `bge-large-en-v1.5` or `GTE-large` | 1024 | Self-hosted fallback to eliminate API dependency |

Use OpenAI embeddings initially for quality. Plan a migration path to self-hosted embeddings for cost control at scale:
- 400K docs × ~200 chunks avg = 80M chunks
- At $0.13/1M tokens (text-embedding-3-large) ≈ **$2,600 one-time** for full corpus
- Re-embedding on model change is the real cost — minimize by abstracting the embedding layer

### 3.6 Infrastructure

| Component | Choice | Notes |
|---|---|---|
| **Container orchestration** | Docker + **ECS Fargate** (AWS) or **Cloud Run** (GCP) | Serverless containers, no cluster mgmt |
| **Object storage** | S3 / GCS | Raw docs + processed artifacts |
| **Relational DB** | PostgreSQL (RDS / Cloud SQL) | Users, orgs, billing, audit, doc metadata |
| **Cache** | Redis (ElastiCache / Memorystore) | Session, rate limit, query result cache |
| **Task queue** | Celery + Redis (or SQS) | Async ingestion pipeline |
| **Monitoring** | Datadog / Grafana + Prometheus | Metrics, traces, logs |
| **CI/CD** | GitHub Actions | Test → Build → Deploy |
| **IaC** | Terraform | Reproducible infra |

### 3.7 Frontend

| Choice | Rationale |
|---|---|
| **Next.js 14+ (App Router)** | SSR for SEO, streaming UI for chat, React Server Components for perf |
| **Tailwind CSS + shadcn/ui** | Rapid, consistent UI |
| **Vercel** (or self-hosted) | Deployment, edge functions, analytics |

---

## 4. Document Ingestion Pipeline (Deep Dive)

### 4.1 Pipeline Stages

```
Raw Document (S3/GCS)
    │
    ▼
┌─────────────────────┐
│  1. FORMAT DETECTION │  Detect PDF vs HTML, encoding, language
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  2. PARSING          │
│                     │
│  PDF → PyMuPDF      │  Fast, layout-aware text extraction
│        (fitz)       │  Fallback: pdfplumber for complex layouts
│                     │
│  HTML → Trafilatura │  Main content extraction (strips boilerplate)
│       + BeautifulSoup│  Structured element preservation
└─────────┬───────────┘
          │
          ├──────────────────────┐
          ▼                      ▼
┌──────────────────┐   ┌───────────────────────┐
│  3a. TEXT CHUNKER │   │  3b. TABLE EXTRACTOR  │
│                  │   │                       │
│  Semantic chunks │   │  PDF: Camelot/pdfplumber│
│  - ~512 tokens   │   │  HTML: pandas.read_html │
│  - Overlap 64    │   │                       │
│  - Section-aware │   │  Output: structured   │
│  - Heading hier- │   │  JSON per table       │
│    archy preserved│   │  + markdown repr     │
│                  │   │  + natural language   │
│                  │   │    summary of table   │
└────────┬─────────┘   └───────────┬───────────┘
         │                         │
         ▼                         ▼
┌─────────────────────────────────────────┐
│  4. METADATA EXTRACTION                  │
│                                         │
│  - Document title, author, date         │
│  - Manufacturer / brand (regex + NER)   │
│  - Model / product identifier           │
│  - Document type (user manual, service  │
│    manual, safety guide, etc.)          │
│  - Section / chapter mapping            │
│  - Language detection                   │
│  - Page numbers (for PDF citations)     │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│  5. EMBEDDING (Batched)                  │
│                                         │
│  Dense: text-embedding-3-large          │
│  Sparse: SPLADE encoder                │
│  Batch size: 2048 chunks per API call   │
│  Rate limiting + retry with backoff     │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│  6. INDEXING                             │
│                                         │
│  Qdrant: upsert vectors + payload       │
│  PostgreSQL: doc metadata + relations   │
│  Update ingestion status tracking       │
└─────────────────────────────────────────┘
```

### 4.2 Table Extraction Strategy

Tables are **first-class citizens**, not just text blobs:

1. **Detect tables** in document during parsing
2. **Extract to structured format** (list of dicts / DataFrame)
3. **Generate three representations**:
   - **Structured JSON** — for programmatic access and precise answers
   - **Markdown table** — for LLM context (models understand markdown tables well)
   - **Natural language summary** — for embedding (e.g., "This table shows torque specifications for the M8 engine bolts, ranging from 22 to 45 Nm")
4. **Index all three** — embed the NL summary, store JSON + markdown as payload
5. **Link table to parent document** — maintain chunk→table→document relationships

### 4.3 Chunking Strategy

```python
# Pseudocode for semantic chunking
class SemanticChunker:
    target_size = 512      # tokens
    overlap = 64           # tokens
    min_size = 100         # tokens — avoid tiny fragments
    max_size = 1024        # tokens — hard cap

    def chunk(self, document):
        # 1. Split by structural boundaries (headings, sections)
        # 2. Within sections, split by paragraph boundaries
        # 3. Merge small consecutive chunks from same section
        # 4. Apply overlap at boundaries
        # 5. Attach parent heading hierarchy to each chunk
        #    e.g., "Chapter 3 > Engine > Torque Specs"
        # 6. Prepend contextual header to each chunk:
        #    "{doc_title} | {section_path} | Page {n}"
```

### 4.4 Ingestion Scale Planning

| Metric | Estimate |
|---|---|
| Documents | 400,000 |
| Avg pages/doc | 50 |
| Avg chunks/doc | 200 |
| Total chunks | ~80,000,000 |
| Avg tokens/chunk | 400 |
| Total tokens | ~32B tokens |
| Embedding cost (one-time) | ~$4,200 (text-embedding-3-large) |
| Embedding throughput | ~3M tokens/min (OpenAI tier 5) |
| **Full corpus embedding time** | ~7-8 hours (parallelized) |
| Vector storage (3072d float32) | ~930 GB raw (quantized: ~230 GB) |
| Qdrant recommended RAM | 64-128 GB (with mmap + quantization) |

---

## 5. Query Pipeline (Deep Dive)

### 5.1 Query Flow

```
User Query
    │
    ▼
┌─────────────────────┐
│  1. QUERY ANALYSIS   │
│                     │
│  - Intent classify  │  (factual / procedural / comparative / troubleshoot)
│  - Entity extract   │  (manufacturer, model, part number)
│  - Complexity score │  (→ routes to model tier)
│  - Filter generation│  (metadata filters from entities)
│  - Query rewrite    │  (expand abbreviations, clarify)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  2. HYBRID RETRIEVAL │
│                     │
│  Dense search       │  top-K from Qdrant (cosine similarity)
│  Sparse search      │  top-K from Qdrant sparse / OpenSearch (BM25)
│  Table search       │  top-K from table collection
│  Metadata filter    │  applied to all searches
│                     │
│  Reciprocal Rank    │  Fuse results: RRF(k=60)
│  Fusion             │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  3. RERANKING        │
│                     │
│  Cross-encoder      │  Cohere Rerank / bge-reranker-v2-m3
│  reranker           │  Rerank top-50 → select top-8
│                     │
│  Diversity filter   │  Ensure results span multiple docs/sections
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  4. CONTEXT ASSEMBLY │
│                     │
│  - Select top chunks│
│  - Include parent   │  (heading hierarchy for context)
│  - Include adjacent │  (±1 chunk for continuity, if relevant)
│  - Include tables   │  (as markdown, with source attribution)
│  - Token budget     │  (fit within model context window)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  5. GENERATION       │
│                     │
│  System prompt:     │
│  - Role definition  │
│  - Citation rules   │  "Cite [Doc Title, Page X] for every claim"
│  - Grounding rules  │  "If info not in context, say so explicitly"
│  - Table handling   │  "Reproduce relevant table data accurately"
│                     │
│  Model: routed per  │
│  complexity score   │
│                     │
│  Stream response    │  SSE to frontend
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  6. POST-PROCESSING  │
│                     │
│  - Citation verify  │  Check cited content exists in context
│  - Confidence score │  Based on retrieval relevance + coverage
│  - Feedback logging │  Store for evaluation pipeline
└─────────────────────┘
```

### 5.2 Hallucination Prevention

| Technique | Implementation |
|---|---|
| **Grounded system prompt** | Explicit instruction: only answer from provided context |
| **Citation enforcement** | Require `[Source: doc, page]` for every factual claim |
| **Citation verification** | Post-generation check: verify cited text exists in retrieved chunks |
| **Confidence scoring** | If top retrieval score < threshold → "I don't have enough information" |
| **Abstention** | Model explicitly told: better to say "not found" than hallucinate |
| **Retrieval transparency** | Show users the source chunks alongside the answer |
| **Evaluation pipeline** | Automated faithfulness scoring on sample queries (RAGAS framework) |

### 5.3 Contextual Filtering

Filters are extracted from user query AND from user session context:

```
Available filter dimensions:
├── manufacturer        (e.g., "Toyota", "Bosch", "Siemens")
├── product_model       (e.g., "Camry 2024", "S7-1500")
├── document_type       (user_manual, service_manual, safety, parts_catalog)
├── section_category    (troubleshooting, specifications, installation, maintenance)
├── language            (en, de, ja, ...)
├── year_published      (range filter)
├── organization_id     (tenant isolation)
└── custom_tags         (user-defined labels)
```

Filter application:
1. **Explicit**: User selects filters in UI
2. **Implicit**: NER extracts entities from query → auto-apply as filters
3. **Session**: Previous conversation context narrows scope (e.g., already discussing Toyota Camry)
4. **Tenant**: Organization-level document access control (always applied)

---

## 6. Anti-Hallucination & Evaluation

### 6.1 Evaluation Pipeline (RAGAS-based)

Run nightly on a curated test set:

| Metric | What it measures | Target |
|---|---|---|
| **Faithfulness** | Are claims grounded in retrieved context? | > 0.90 |
| **Answer relevancy** | Does the answer address the question? | > 0.85 |
| **Context precision** | Are retrieved chunks relevant? | > 0.80 |
| **Context recall** | Did we retrieve all needed info? | > 0.75 |

### 6.2 Human-in-the-Loop

- Thumbs up/down on responses → feeds evaluation dataset
- Flag & review pipeline for low-confidence answers
- Weekly review of flagged responses by domain experts

---

## 7. Subscription & Billing

### 7.1 Tier Structure

| Tier | Queries/mo | Features | Price |
|---|---|---|---|
| **Free** | 50 | Basic search, 1 user | $0 |
| **Pro** | 2,000 | Full RAG, chat history, API access | $49/mo |
| **Team** | 10,000 | Multi-user, shared collections, priority | $199/mo |
| **Enterprise** | Unlimited | SSO, SLA, custom models, on-prem option | Custom |

### 7.2 Stripe Integration

```
User action → API middleware → check quota → process query → increment usage → Stripe metering
                                    │
                                    └── if over limit → return 429 + upgrade prompt
```

- **Stripe Billing** for subscription management
- **Stripe Usage Records** for metered billing (overage charges)
- **Stripe Customer Portal** for self-service plan changes
- Webhook handlers for: subscription created/updated/canceled, payment failed, invoice paid

---

## 8. Security & Privacy

| Layer | Implementation |
|---|---|
| **Authentication** | JWT + refresh tokens, OAuth 2.0 (Google, Microsoft SSO) |
| **Authorization** | RBAC (admin, editor, viewer) + document-level ACLs |
| **API security** | Rate limiting (Redis), API key management, CORS |
| **Data encryption** | AES-256 at rest, TLS 1.3 in transit |
| **Data isolation** | Tenant ID on every query, row-level security in PostgreSQL |
| **LLM privacy** | Zero data retention agreements with model providers, no training on user data |
| **Audit logging** | Every query, document access, admin action logged to immutable audit trail |
| **Compliance** | SOC 2 readiness from day 1, GDPR data export/deletion |
| **Secrets** | AWS Secrets Manager / GCP Secret Manager — never in code or env vars |

---

## 9. Cost Optimization Strategy

### 9.1 Major Cost Drivers (estimated at scale)

| Component | Monthly Estimate | Optimization |
|---|---|---|
| LLM API calls | $3,000 - $15,000 | Model routing, caching, open model fallback |
| Embedding API | $200 (incremental only) | Batch processing, self-hosted migration path |
| Vector DB infra | $800 - $2,000 | Quantization, mmap, tiered storage |
| Compute (API servers) | $500 - $1,500 | Auto-scaling, right-sizing |
| Object storage | $300 - $500 | Lifecycle policies, compression |
| PostgreSQL + Redis | $400 - $800 | Reserved instances |

### 9.2 Cost Control Mechanisms

1. **Query result caching** — Cache embedding + retrieval results for identical/similar queries (Redis, TTL 1hr)
2. **Model routing** — 70% of queries to cheap models, 30% to premium
3. **Embedding caching** — Don't re-embed unchanged documents
4. **Quantization** — Binary/scalar quantization in Qdrant (4x storage reduction, minimal quality loss)
5. **Async batch processing** — Ingestion during off-peak hours
6. **Self-hosted embedding model** — Migrate to BGE/GTE on GPU instance when volume justifies it
7. **Response streaming** — Stream tokens to reduce perceived latency (no cost impact, but better UX)

---

## 10. Project Structure

```
llm_rag_search_system/
│
├── docker-compose.yml              # Local development stack
├── docker-compose.prod.yml         # Production overrides
├── Dockerfile                      # API service image
├── Makefile                        # Common commands
├── pyproject.toml                  # Python project config (uv/poetry)
├── alembic.ini                     # DB migrations config
│
├── terraform/                      # Infrastructure as Code
│   ├── modules/
│   ├── environments/
│   │   ├── staging/
│   │   └── production/
│   └── main.tf
│
├── src/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application entry
│   ├── config.py                   # Settings (pydantic-settings)
│   ├── dependencies.py             # Dependency injection
│   │
│   ├── api/                        # API layer
│   │   ├── __init__.py
│   │   ├── middleware/
│   │   │   ├── auth.py
│   │   │   ├── rate_limit.py
│   │   │   └── tenant.py
│   │   ├── routes/
│   │   │   ├── query.py            # /v1/query, /v1/query/stream
│   │   │   ├── documents.py        # /v1/documents (CRUD, upload)
│   │   │   ├── collections.py      # /v1/collections
│   │   │   ├── auth.py             # /v1/auth
│   │   │   ├── billing.py          # /v1/billing, webhooks
│   │   │   └── admin.py            # /v1/admin
│   │   └── schemas/                # Pydantic request/response models
│   │       ├── query.py
│   │       ├── document.py
│   │       └── billing.py
│   │
│   ├── core/                       # Business logic
│   │   ├── __init__.py
│   │   ├── query/
│   │   │   ├── pipeline.py         # Orchestrates full query flow
│   │   │   ├── analyzer.py         # Query analysis, intent, entities
│   │   │   ├── retriever.py        # Hybrid retrieval + RRF fusion
│   │   │   ├── reranker.py         # Cross-encoder reranking
│   │   │   ├── context_builder.py  # Assemble context for LLM
│   │   │   ├── generator.py        # LLM call + streaming
│   │   │   ├── citation.py         # Citation extraction + verification
│   │   │   └── model_router.py     # Route to appropriate LLM
│   │   │
│   │   ├── ingestion/
│   │   │   ├── pipeline.py         # Orchestrates ingestion flow
│   │   │   ├── parsers/
│   │   │   │   ├── pdf_parser.py
│   │   │   │   ├── html_parser.py
│   │   │   │   └── base.py
│   │   │   ├── chunker.py          # Semantic chunking
│   │   │   ├── table_extractor.py  # Table detection + extraction
│   │   │   ├── metadata.py         # Metadata extraction + NER
│   │   │   └── embedder.py         # Batch embedding
│   │   │
│   │   ├── crawler/                # Data acquisition
│   │   │   ├── base.py             # BaseCrawler ABC + CrawlResult
│   │   │   └── sources/
│   │   │       ├── internet_archive.py  # IA search + download
│   │   │       └── manufacturer.py      # Base for per-site adapters
│   │   │
│   │   ├── auth/
│   │   │   ├── service.py
│   │   │   ├── jwt.py
│   │   │   └── rbac.py
│   │   │
│   │   └── billing/
│   │       ├── service.py
│   │       ├── stripe_client.py
│   │       └── usage_tracker.py
│   │
│   ├── db/                         # Data layer
│   │   ├── __init__.py
│   │   ├── postgres/
│   │   │   ├── models.py           # SQLAlchemy models
│   │   │   ├── session.py          # DB session management
│   │   │   └── repositories/      # Data access patterns
│   │   ├── vector/
│   │   │   ├── qdrant_client.py    # Qdrant operations
│   │   │   └── collections.py     # Collection management
│   │   ├── cache/
│   │   │   └── redis_client.py    # Cache operations
│   │   └── storage/
│   │       └── s3_client.py       # Object storage operations
│   │
│   └── shared/                    # Cross-cutting concerns
│       ├── logging.py
│       ├── monitoring.py
│       ├── exceptions.py
│       └── constants.py
│
├── migrations/                    # Alembic migrations
│   └── versions/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── fixtures/
│
├── scripts/
│   ├── seed_db.py
│   ├── bulk_ingest.py             # CLI for bulk document ingestion
│   ├── crawl_internet_archive.py  # CLI for IA manual downloads
│   ├── evaluate.py                # Run RAGAS evaluation
│   └── benchmark.py               # Retrieval quality benchmarks
│
├── data/
│   ├── raw/                       # Downloaded source documents
│   └── processed/                 # Post-ingestion artifacts
│
├── evaluation/                    # Quality evaluation
│   ├── test_sets/                 # Curated Q&A pairs
│   ├── metrics/                   # Custom evaluation metrics
│   └── reports/                   # Generated evaluation reports
│
└── docs/
    ├── system_architecture.md     # Detailed system architecture
    ├── api.md                     # API documentation
    ├── deployment.md              # Deployment runbook
    └── runbook.md                 # Operations runbook
```

---

## 11. Data Sourcing Strategy

Building a 400K+ manual corpus requires a phased, multi-source acquisition strategy.

### 11.1 Tier 1 — Open / Public Domain Sources (Start Here)

These are freely available and legally safe for development and initial product launch.

| Source | What You Get | Volume Estimate | Method |
|---|---|---|---|
| **Internet Archive** (archive.org) | Out-of-copyright manuals, vintage electronics, military TMs | 50K–100K+ | Advanced Search API → bulk PDF download |
| **U.S. Government / Military** (everyspec.com, MIL-STD) | Technical manuals, maintenance procedures, spec sheets | 20K–40K | Web scraping, FOIA archives |
| **Manufacturer Open Portals** | Support docs published publicly (datasheets, install guides) | 10K–30K per OEM | Crawler per manufacturer site |
| **WikiBooks / Wikidata** | Structured how-to and reference content | 5K–10K | MediaWiki API |
| **Project Gutenberg / HathiTrust** | Historical technical texts | 5K–10K | Bulk download |

> **Internet Archive is the primary starting point.** The `InternetArchiveCrawler` module
> (see `src/core/crawler/sources/internet_archive.py`) is already implemented and can
> search by collection, media type, and keyword, then download PDFs with metadata sidecars.

### 11.2 Tier 2 — Licensed / Partnership Sources

To reach 400K and ensure modern, commercially relevant content:

| Source | What You Get | Access Model |
|---|---|---|
| **ManualsLib** | 8M+ manuals across all categories | Partnership / API license |
| **iFixit** | Repair guides, teardowns, device manuals | Creative Commons (BY-NC-SA) |
| **Manufacturer partnerships** (OEM programs) | Official service / repair manuals | Revenue-share or bulk license |
| **Industry associations** (IEEE, SAE, ASHRAE) | Standards and technical publications | Institutional license |

### 11.3 Tier 3 — Community & User-Contributed

Long-tail coverage and freshness:

- **User uploads** — customers upload manuals for devices they own (de-dup against existing corpus).
- **Crowdsourced metadata correction** — flag wrong manufacturer/model associations.
- **OCR improvement pipeline** — users report bad OCR; feed corrections back.

### 11.4 Acquisition Pipeline

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  Source   │───▶│   Crawler    │───▶│  De-dup &    │───▶│ Ingestion│
│ (IA, OEM) │    │  (per-source)│    │  Validation  │    │ Pipeline │
└──────────┘    └──────────────┘    └──────────────┘    └──────────┘
                   │                    │
                   ▼                    ▼
              metadata.json        quality score
              (title, author,      (OCR confidence,
               year, format)        page count, lang)
```

**Quality gates before ingestion:**
1. **De-duplication** — SHA-256 content hash + fuzzy title matching to avoid indexing the same manual twice.
2. **Language detection** — Reject non-English documents (Phase 1); expand later.
3. **OCR quality score** — Run a sample page through Tesseract; reject if character confidence < 70%.
4. **Minimum content threshold** — Reject documents with < 2 pages or < 500 characters of extractable text.

### 11.5 Licensing & Legal Considerations

| Source Type | License / Right | Action Required |
|---|---|---|
| Public domain (pre-1929, US gov) | Free use | None — ingest directly |
| Creative Commons (BY, BY-SA) | Attribution required | Store license metadata, display attribution |
| CC Non-Commercial (BY-NC-SA) | Non-commercial only | OK for free tier; review for paid tiers |
| Manufacturer-published support docs | Fair use / implied license | Respect robots.txt, link back to source |
| Licensed / partnership content | Per-agreement | Separate storage, access controls per contract |

### 11.6 Corpus Growth Roadmap

| Phase | Target Corpus Size | Primary Sources |
|---|---|---|
| Phase 1 (Foundation) | 1K–5K | Internet Archive, US Gov/MIL |
| Phase 2 (Hybrid Retrieval) | 10K–50K | + Manufacturer portals, iFixit |
| Phase 3 (Scale) | 50K–200K | + ManualsLib partnership, bulk OEM deals |
| Phase 4 (Full Corpus) | 200K–400K+ | + User uploads, community, long-tail crawling |

---

## 12. Implementation Phases

### Phase 1 — Foundation (Weeks 1-4)

**Goal**: Core retrieval working end-to-end with a small document set.

- [ ] Project scaffolding (FastAPI, Docker, pyproject.toml, CI)
- [ ] PostgreSQL schema + Alembic migrations (users, documents, chunks)
- [ ] S3/GCS integration for document storage
- [ ] PDF parser (PyMuPDF) + HTML parser (Trafilatura)
- [ ] Semantic chunker (basic version)
- [ ] Embedding pipeline (OpenAI text-embedding-3-large)
- [ ] Qdrant setup + dense vector indexing
- [ ] Basic query endpoint: embed query → vector search → return chunks
- [ ] Simple LLM generation with citation prompt
- [ ] Docker Compose for local dev (Qdrant, PostgreSQL, Redis, API)
- [ ] Basic test suite

**Deliverable**: Query 1,000 documents via API, get grounded answers with citations.

### Phase 2 — Hybrid Retrieval & Tables (Weeks 5-8)

**Goal**: Production-quality retrieval with hybrid search and table support.

- [ ] SPLADE sparse vector encoding
- [ ] Hybrid retrieval (dense + sparse) with RRF fusion
- [ ] Cross-encoder reranking (Cohere or self-hosted)
- [ ] Table extraction pipeline (Camelot + pdfplumber)
- [ ] Table-aware indexing (separate collection, NL summaries)
- [ ] Metadata extraction (NER for manufacturer, model, doc type)
- [ ] Contextual filtering (metadata filters in Qdrant queries)
- [ ] Query analysis (intent classification, entity extraction)
- [ ] Context window management (token budgeting)
- [ ] Streaming response (SSE)
- [ ] Query caching (Redis)
- [ ] Integration tests for full query pipeline

**Deliverable**: High-quality hybrid retrieval with table support on 10K documents.

### Phase 3 — Scale & Ingestion (Weeks 9-12)

**Goal**: Ingest the full 400K corpus, optimize for throughput and cost.

- [ ] Celery task queue for async ingestion
- [ ] Bulk ingestion CLI with progress tracking
- [ ] Batch embedding with rate limiting and retry
- [ ] Ingestion status dashboard
- [ ] Qdrant sharding configuration for 80M+ vectors
- [ ] Vector quantization (scalar/binary)
- [ ] Model router (rule-based v1)
- [ ] Cost tracking per query
- [ ] Load testing (Locust)
- [ ] Monitoring setup (Prometheus + Grafana)
- [ ] Full corpus ingestion run

**Deliverable**: Full 400K corpus indexed, query latency < 2s P95.

### Phase 4 — Auth, Billing & Frontend (Weeks 13-16)

**Goal**: Multi-tenant platform with subscription billing and web UI.

- [ ] JWT authentication + refresh tokens
- [ ] API key management
- [ ] RBAC (admin, editor, viewer)
- [ ] Tenant isolation (org-level document access)
- [ ] Stripe subscription integration
- [ ] Usage tracking + metered billing
- [ ] Stripe webhook handlers
- [ ] Next.js frontend: chat UI, search, doc browser
- [ ] Streaming chat rendering
- [ ] Source citation display with doc links
- [ ] Admin dashboard (usage, documents, users)
- [ ] Billing portal integration

**Deliverable**: Fully functional multi-tenant platform with billing.

### Phase 5 — Hardening & Enterprise (Weeks 17-20)

**Goal**: Production hardening, enterprise features, evaluation.

- [ ] SSO integration (SAML/OIDC)
- [ ] Audit logging (immutable)
- [ ] RAGAS evaluation pipeline
- [ ] Automated faithfulness testing (nightly)
- [ ] Hallucination detection + abstention tuning
- [ ] Conversation memory (multi-turn context)
- [ ] Terraform infrastructure
- [ ] Blue-green deployment setup
- [ ] Disaster recovery + backup procedures
- [ ] Security audit
- [ ] Performance optimization pass
- [ ] Documentation (API docs, runbook, architecture)

**Deliverable**: Production-ready, enterprise-grade platform.

### Phase 6 — Optimization & Intelligence (Weeks 21+)

**Goal**: Continuous improvement, advanced features.

- [ ] ML-based model router (v2)
- [ ] Self-hosted embedding model migration
- [ ] Agentic workflows (multi-step reasoning)
- [ ] Document comparison ("diff these two manuals")
- [ ] Automated document update detection
- [ ] Analytics dashboard (query patterns, knowledge gaps)
- [ ] Webhook integrations for enterprise
- [ ] Multi-language support
- [ ] Mobile-responsive redesign

---

## 13. Key Technical Decisions to Validate Early

| Decision | Validation Method | When |
|---|---|---|
| Qdrant vs alternatives at 80M vectors | Load test with synthetic data | Week 2 |
| Chunking strategy (size, overlap) | Retrieval quality eval on 1K doc sample | Week 3 |
| SPLADE vs BM25 (OpenSearch) for sparse | A/B retrieval quality comparison | Week 5 |
| Reranker model selection | Benchmark on curated test set | Week 6 |
| Table extraction accuracy | Manual review of 100 table-heavy docs | Week 7 |
| Model router effectiveness | Cost vs quality analysis | Week 10 |
| Quantization quality impact | Before/after retrieval eval | Week 11 |

---

## 14. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| LLM API cost exceeds budget | High | Model router, caching, self-hosted fallback |
| Qdrant performance at 80M vectors | High | Early load testing, sharding strategy, Pgvector fallback |
| Table extraction accuracy for complex PDFs | Medium | Multiple extraction libraries, manual review pipeline |
| Embedding model quality changes | Medium | Abstract embedding layer, keep evaluation baselines |
| Stripe integration complexity | Low | Use Stripe's pre-built components (Customer Portal, Checkout) |
| OpenAI/Anthropic API outages | Medium | Multi-provider fallback, circuit breaker pattern |
| Data privacy compliance | High | Zero-retention agreements, data residency config, legal review |

---

## 15. Success Metrics

| Metric | Target | How to Measure |
|---|---|---|
| Answer faithfulness | > 90% | RAGAS faithfulness score |
| Query latency (P95) | < 2 seconds | API response time monitoring |
| Retrieval precision@10 | > 80% | Evaluation test set |
| User satisfaction | > 4.2/5 | In-app feedback |
| Monthly active users | Growth | Analytics |
| Cost per query | < $0.05 avg | Cost tracking |
| Uptime | 99.9% | Monitoring |
| Ingestion throughput | > 1000 docs/hour | Pipeline metrics |

---

## 16. References

| # | Title | Link | Topic |
|---|---|---|---|
| 1 | Comparing SPLADE Sparse Vectors with BM25 | [Medium — Zilliz](https://medium.com/@zilliz_learn/comparing-splade-sparse-vectors-with-bm25-53368877359f) | Sparse retrieval, SPLADE vs BM25 |

---

*This plan is a living document. Update as decisions are validated and requirements evolve.*
