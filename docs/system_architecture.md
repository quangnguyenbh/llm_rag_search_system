# System Architecture

> ManualAI — detailed architecture for the RAG search platform.

---

## 1. System Context Diagram

```
                    ┌──────────────┐
                    │   End Users  │
                    │  (Browser)   │
                    └──────┬───────┘
                           │ HTTPS
                           ▼
                    ┌──────────────┐
                    │   Next.js    │
                    │   Frontend   │
                    └──────┬───────┘
                           │ REST / SSE
                           ▼
              ┌────────────────────────┐
              │   API Gateway / ALB    │
              │   (Rate limit, TLS)    │
              └────────────┬───────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
  ┌───────────┐    ┌──────────────┐    ┌───────────┐
  │   Auth    │    │    Query     │    │   Admin   │
  │  Service  │    │   Service    │    │  Service  │
  └───────────┘    └──────┬───────┘    └─────┬─────┘
                          │                  │
                          ▼                  │
              ┌───────────────────┐          │
              │   Retrieval Layer │          │
              │  (Qdrant + LLM)  │          │
              └───────────────────┘          │
                                             │
              ┌──────────────────────────────┘
              ▼
  ┌────────────────────────────────────────────────┐
  │              Data Acquisition Layer             │
  │                                                │
  │  ┌──────────────┐  ┌─────────────────────────┐│
  │  │   Crawler    │  │   Ingestion Pipeline     ││
  │  │  (IA, Mfg)  │──│  Parse→Chunk→Embed→Index ││
  │  └──────────────┘  └─────────────────────────┘│
  └────────────────────────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌──────────┐ ┌────────┐ ┌──────────┐
        │ Qdrant   │ │Postgres│ │  S3/GCS  │
        │ (vectors)│ │ (meta) │ │  (docs)  │
        └──────────┘ └────────┘ └──────────┘
```

---

## 2. Data Flow: Document Acquisition & Ingestion

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA ACQUISITION                                 │
│                                                                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌────────────────────┐  │
│  │ Internet Archive │    │  Manufacturer   │    │   User Upload      │  │
│  │    Crawler       │    │   Crawlers      │    │   (API endpoint)   │  │
│  │                  │    │                 │    │                    │  │
│  │  Search API      │    │  Per-site       │    │  Validate format   │  │
│  │  → Filter PDFs   │    │  adapters       │    │  → Store to S3     │  │
│  │  → Download      │    │  (future)       │    │  → Queue ingest    │  │
│  │  → Save + meta   │    │                 │    │                    │  │
│  └────────┬─────────┘    └────────┬────────┘    └────────┬───────────┘  │
│           └──────────────────────┼───────────────────────┘              │
└─────────────────────────────────┼───────────────────────────────────────┘
                                  ▼
                    ┌──────────────────────┐
                    │   Object Storage     │
                    │   (S3/GCS/local)     │
                    │   data/raw/          │
                    └──────────┬───────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE (Celery)                         │
│                                                                        │
│  ┌────────────┐   ┌───────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │  1. Parse  │──▶│ 2. Chunk  │──▶│  3. Extract  │──▶│  4. Embed   │  │
│  │            │   │           │   │              │   │             │  │
│  │ PDF:PyMuPDF│   │ Semantic  │   │ Tables:      │   │ Dense:      │  │
│  │ HTML:Traf. │   │ 512 tok   │   │  Camelot     │   │  OpenAI     │  │
│  │            │   │ 64 overlap│   │ Metadata:    │   │ Sparse:     │  │
│  │            │   │ Section-  │   │  NER         │   │  SPLADE     │  │
│  │            │   │  aware    │   │              │   │ Batch 2048  │  │
│  └────────────┘   └───────────┘   └──────────────┘   └──────┬──────┘  │
│                                                              │         │
│                                                              ▼         │
│                                                    ┌─────────────────┐ │
│                                                    │   5. Index      │ │
│                                                    │                 │ │
│                                                    │ Qdrant: vectors │ │
│                                                    │ PG: metadata    │ │
│                                                    │ Status: indexed │ │
│                                                    └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow: Query Pipeline

```
User Question
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│  1. QUERY ANALYSIS                                              │
│     • Intent: factual | procedural | comparative | troubleshoot │
│     • NER: manufacturer, model, part number                     │
│     • Complexity score → model routing                          │
│     • Implicit filter extraction from query                     │
└─────────────────────┬──────────────────────────────────────────┘
                      ▼
┌────────────────────────────────────────────────────────────────┐
│  2. HYBRID RETRIEVAL                                            │
│                                                                │
│     ┌─────────────────┐     ┌──────────────────┐              │
│     │  Dense Search    │     │  Sparse Search   │              │
│     │  (Qdrant cosine) │     │  (SPLADE / BM25) │              │
│     │  top-50          │     │  top-50          │              │
│     └────────┬─────────┘     └─────────┬────────┘              │
│              └───────────┬─────────────┘                       │
│                          ▼                                     │
│              ┌─────────────────────┐                           │
│              │ Reciprocal Rank     │                           │
│              │ Fusion (k=60)       │                           │
│              └─────────┬───────────┘                           │
│                        ▼                                       │
│              + Table collection search (merged via RRF)        │
│              + Metadata filters applied to all searches        │
└────────────────────────┬───────────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────────────┐
│  3. RERANKING                                                   │
│     Cross-encoder (Cohere / bge-reranker-v2-m3)               │
│     Top-50 → Top-8                                             │
│     Diversity filter: ensure multi-doc coverage                │
└────────────────────────┬───────────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────────────┐
│  4. CONTEXT ASSEMBLY                                            │
│     • Select top chunks with source headers                    │
│     • Include heading hierarchy for context                    │
│     • Include adjacent chunks if relevant                      │
│     • Include table markdown with attribution                  │
│     • Token budget management                                  │
└────────────────────────┬───────────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────────────┐
│  5. GENERATION                                                  │
│     Model router → Haiku (simple) | Sonnet (complex)           │
│     Grounded system prompt with citation rules                 │
│     Stream tokens via SSE to frontend                          │
└────────────────────────┬───────────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────────────┐
│  6. POST-PROCESSING                                             │
│     • Citation verification (cited text exists in context)     │
│     • Confidence scoring                                       │
│     • Feedback/audit logging                                   │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. Crawler Architecture

The crawler subsystem is designed as a pluggable framework where each data source
gets its own adapter, but shares common infrastructure.

```
src/core/crawler/
├── base.py                          # BaseCrawler ABC + CrawlResult
└── sources/
    ├── internet_archive.py          # IA search API + download (implemented)
    └── manufacturer.py              # Base for per-manufacturer adapters (template)
```

### Internet Archive Crawler Flow

```
1. Search IA API
   GET archive.org/advancedsearch.php
   ?q=(query) AND collection:(manuals) AND mediatype:(texts)
   → list of item identifiers
            │
            ▼
2. For each item:
   GET archive.org/metadata/{identifier}
   → find PDF files in item metadata
            │
            ▼
3. Download PDF
   GET archive.org/download/{identifier}/{filename}.pdf
   → save to data/raw/internet_archive/{identifier}.pdf
            │
            ▼
4. Save metadata sidecar
   → data/raw/internet_archive/{identifier}.meta.json
   Contains: title, creator, date, source_url, description
```

### Adding New Crawl Sources

1. Create a new file in `src/core/crawler/sources/`
2. Subclass `BaseCrawler` from `base.py`
3. Implement `async crawl(**kwargs) -> CrawlResult`
4. Use `self._rate_limit()` between HTTP requests
5. Use `self._safe_filename()` for output file naming
6. Save a `.meta.json` sidecar with provenance data for each document

---

## 5. Infrastructure Topology

### Local Development

```
docker compose up -d
├── api          (FastAPI, port 8000)
├── worker       (Celery worker)
├── postgres     (PostgreSQL 16, port 5432)
├── redis        (Redis 7, port 6379)
└── qdrant       (Qdrant, ports 6333/6334)
```

### Production (AWS)

```
                    ┌─────────────┐
                    │ Route 53    │
                    │ (DNS)       │
                    └──────┬──────┘
                           ▼
                    ┌──────────────┐
                    │  CloudFront  │
                    │  (CDN)       │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  ALB         │
                    └──────┬───────┘
                           ▼
              ┌────────────────────────┐
              │  ECS Fargate Cluster   │
              │  ┌──────┐  ┌────────┐ │
              │  │ API  │  │ Worker │ │
              │  │ (x3) │  │ (x2)  │ │
              │  └──────┘  └────────┘ │
              └────────────────────────┘
                     │          │
        ┌────────────┼──────────┼────────────┐
        ▼            ▼          ▼            ▼
  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌────────┐
  │ RDS PG   │ │ Qdrant   │ │ Elasti │ │  S3    │
  │ (primary │ │ (EC2 or  │ │ Cache  │ │(docs)  │
  │ +replica)│ │  managed)│ │(Redis) │ │        │
  └──────────┘ └──────────┘ └────────┘ └────────┘
```

---

## 6. Data Model (PostgreSQL)

```
┌──────────────────┐       ┌──────────────────┐
│   organizations  │       │      users       │
├──────────────────┤       ├──────────────────┤
│ id (PK)          │───┐   │ id (PK)          │
│ name             │   │   │ email (unique)   │
│ plan             │   │   │ name             │
│ stripe_sub_id    │   │   │ hashed_password  │
│ created_at       │   │   │ role             │
└──────────────────┘   ├──▶│ organization_id  │
                       │   │ stripe_cust_id   │
                       │   │ created_at       │
                       │   └──────────────────┘
                       │
                       │   ┌──────────────────┐
                       │   │    documents     │
                       │   ├──────────────────┤
                       └──▶│ id (PK)          │
                           │ title            │
                           │ format           │
                           │ storage_path     │
                           │ status           │
                           │ organization_id  │
                           │ metadata_json    │
                           │ chunk_count      │
                           │ page_count       │
                           │ source_url       │
                           │ source_type      │
                           │ created_at       │
                           └──────────────────┘

┌──────────────────┐
│   crawl_jobs     │
├──────────────────┤
│ id (PK)          │
│ source           │
│ status           │
│ params_json      │
│ documents_found  │
│ documents_dl     │
│ error_message    │
│ created_at       │
│ completed_at     │
└──────────────────┘
```

---

## 7. Vector Storage (Qdrant)

### Collections

| Collection | Vectors | Purpose |
|---|---|---|
| `manual_chunks_dense` | 3072-dim (text-embedding-3-large) | Dense semantic search |
| `manual_chunks_sparse` | Sparse (SPLADE) | Keyword-aware retrieval |
| `manual_tables` | 3072-dim (NL summary embeddings) | Table-specific search |

### Payload Schema (per point)

```json
{
  "document_id": "uuid",
  "chunk_id": "uuid",
  "text": "chunk text content",
  "page_number": 42,
  "section_path": "Chapter 3 > Engine > Torque Specs",
  "heading_hierarchy": ["Chapter 3", "Engine", "Torque Specs"],
  "manufacturer": "Toyota",
  "product_model": "Camry 2024",
  "document_type": "service_manual",
  "language": "en",
  "organization_id": "uuid",
  "chunk_type": "text"
}
```

---

## 8. Security Architecture

```
┌─────────────────────────────────────────────────┐
│                   REQUEST FLOW                    │
│                                                  │
│  Client → TLS 1.3 → ALB → Rate Limiter (Redis)  │
│         → JWT Validation → RBAC Check            │
│         → Tenant Isolation (org_id filter)        │
│         → Business Logic                          │
│         → Audit Log                               │
└─────────────────────────────────────────────────┘

Encryption:
  - In transit: TLS 1.3
  - At rest: AES-256 (RDS, S3, EBS)

Auth:
  - JWT access tokens (short-lived, 30 min)
  - Refresh tokens (long-lived, rotated)
  - API keys for programmatic access
  - OAuth 2.0 / SAML for enterprise SSO

Data isolation:
  - Every Qdrant query includes organization_id filter
  - PostgreSQL row-level security by organization
  - S3 objects keyed by org prefix

Secrets:
  - AWS Secrets Manager (never in env vars or code)
  - Rotated on schedule
```

---

*See [plan.md](plan.md) for implementation phases, cost estimates, and risk register.*
