# Project Status Report — ManualAI RAG System

**Date**: 2026-03-19
**Repo**: quangnguyenbh/llm_rag_search_system
**Review scope**: Current implementation state vs. plan.md phases

---

## Executive Summary

The ManualAI project has achieved **Phase 1.5** completion with a functional RAG pipeline using **Amazon Nova 2 Lite + AWS Bedrock Titan Embeddings + Qdrant** for vector search. The system can ingest PDFs, chunk semantically, embed with AWS Bedrock, store in Qdrant, and answer questions via a web dashboard.

**What works:**
- ✅ End-to-end RAG pipeline (ingest → query → answer with citations)
- ✅ AWS Bedrock integration (Nova 2 Lite for generation, Titan Embed v2 for embeddings)
- ✅ Qdrant vector database with dense search
- ✅ Semantic chunking with configurable parameters
- ✅ Web dashboard UI for chat and search
- ✅ Document crawlers (Internet Archive + HuggingFace datasets)
- ✅ FastAPI backend with modular structure

**What's missing (critical gaps):**
- ❌ Hybrid retrieval (sparse vectors / BM25) — dense-only limits keyword precision
- ❌ Reranking — retrieval precision needs improvement
- ❌ Table extraction — no structured table handling yet
- ❌ Query caching — no Redis integration for performance
- ❌ Authentication & authorization — no auth layer
- ❌ Billing & subscriptions — no Stripe integration
- ❌ Database migrations — no Alembic setup for PostgreSQL
- ❌ Testing infrastructure — limited test coverage
- ❌ Production deployment config — no infrastructure-as-code

---

## Detailed Implementation Status

### ✅ Phase 1 — Foundation (Week 1-4) — MOSTLY COMPLETE

| Component | Status | Notes |
|-----------|--------|-------|
| FastAPI scaffolding | ✅ Complete | `src/main.py` with routes, CORS, health endpoint |
| PostgreSQL schema | ⚠️ Partial | Models defined (`src/db/postgres/models.py`) but no Alembic migrations |
| S3/GCS integration | ⚠️ Partial | Client exists (`src/db/storage/s3_client.py`) but local-only mode |
| PDF parser | ✅ Complete | PyMuPDF-based parser with text + metadata extraction |
| HTML parser | ✅ Complete | Trafilatura + BeautifulSoup parser |
| Semantic chunker | ✅ Complete | 351 LOC with heading hierarchy, token counting, overlap |
| Embedding pipeline | ✅ Complete | AWS Bedrock Titan Embed v2 with batch + retry logic |
| Qdrant setup | ✅ Complete | Dense vector indexing, collection management, search |
| Query endpoint | ✅ Complete | `/v1/query` with chat + search-only modes |
| LLM generation | ✅ Complete | Amazon Nova 2 Lite via Bedrock with grounding prompt |
| Docker Compose | ✅ Complete | Postgres, Redis, Qdrant, API, worker services |
| Test suite | ⚠️ Minimal | Unit tests exist but no pytest installed, no CI integration |

**Phase 1 blockers resolved:**
- ✅ Embedding model: Uses AWS Bedrock Titan Embed v2 (1024d)
- ✅ LLM: Uses Amazon Nova 2 Lite via Bedrock
- ✅ Vector DB: Qdrant local + cloud support

**Phase 1 remaining work:**
- 🔴 Set up Alembic migrations for PostgreSQL schema
- 🔴 Add comprehensive test suite with pytest
- 🔴 Integrate S3 for document storage (currently local-only)

---

### ⚠️ Phase 2 — Hybrid Retrieval & Tables (Week 5-8) — NOT STARTED

| Component | Status | Notes |
|-----------|--------|-------|
| Sparse vector encoding (SPLADE/BM25) | ❌ Not started | Qdrant supports sparse vectors but not implemented |
| Hybrid retrieval with RRF fusion | ❌ Not started | Only dense search currently |
| Cross-encoder reranking | ❌ Not started | No reranker integration (Cohere/open model) |
| Table extraction | ❌ Not started | `table_extractor.py` exists but not implemented |
| Table indexing | ❌ Not started | No separate table collection |
| Metadata extraction (NER) | ⚠️ Partial | `metadata.py` exists with basic extraction, no spaCy NER |
| Contextual filtering | ⚠️ Partial | Qdrant supports filters but limited query analysis |
| Query analysis | ⚠️ Partial | `analyzer.py` has structure but minimal logic |
| Context window management | ✅ Complete | `context_builder.py` handles token budgeting |
| Streaming response (SSE) | ✅ Complete | Generator supports streaming via Bedrock |
| Query caching (Redis) | ❌ Not started | Redis container exists but no cache integration |
| Integration tests | ❌ Not started | Some exist but incomplete |

**Phase 2 critical priorities:**
1. 🔴 **Hybrid retrieval** — Add sparse vectors (SPLADE or BM25) + RRF fusion for better keyword recall
2. 🔴 **Reranking** — Integrate cross-encoder reranker (Cohere API or self-hosted BGE)
3. 🔴 **Table extraction** — Implement Camelot/pdfplumber table extraction + structured indexing
4. 🟡 **Query caching** — Add Redis caching for embeddings + retrieval results
5. 🟡 **Query analysis** — Enhance intent classification, entity extraction, complexity scoring

---

### ❌ Phase 3 — Scale & Ingestion (Week 9-12) — NOT STARTED

| Component | Status | Notes |
|-----------|--------|-------|
| Celery task queue | ⚠️ Partial | Worker service in docker-compose but no task definitions |
| Bulk ingestion CLI | ✅ Complete | `scripts/bulk_ingest.py` with progress tracking |
| Batch embedding optimization | ✅ Complete | Bedrock embedder handles batching + retry + jitter |
| Ingestion status tracking | ❌ Not started | No DB tracking of ingestion jobs |
| Qdrant sharding config | ❌ Not started | Single-node setup, no sharding for 80M+ vectors |
| Vector quantization | ❌ Not started | No scalar/binary quantization configured |
| Model router | ⚠️ Partial | `model_router.py` exists but rule-based routing not implemented |
| Cost tracking | ❌ Not started | No query cost metrics |
| Load testing | ❌ Not started | Locust not configured |
| Monitoring | ⚠️ Partial | structlog logging exists, no Prometheus/Grafana |

**Phase 3 blockers:**
- Need Phase 2 complete (hybrid retrieval) before scaling to 400K docs
- Need PostgreSQL migrations + schema for ingestion status tracking

---

### ❌ Phase 4 — Auth, Billing & Frontend (Week 13-16) — NOT STARTED

| Component | Status | Notes |
|-----------|--------|-------|
| JWT authentication | ⚠️ Scaffolded | `src/core/auth/jwt.py` exists but not integrated |
| API key management | ❌ Not started | No API key generation/validation |
| RBAC | ⚠️ Scaffolded | `rbac.py` exists but not enforced |
| Tenant isolation | ❌ Not started | No org-level document access control |
| Stripe subscription | ⚠️ Scaffolded | `billing/stripe_client.py` exists but not integrated |
| Usage tracking | ⚠️ Scaffolded | `usage_tracker.py` exists but not implemented |
| Stripe webhooks | ❌ Not started | No webhook handlers |
| Next.js frontend | ⚠️ Dashboard only | Static HTML dashboard, not full Next.js app |
| Admin dashboard | ❌ Not started | Admin routes exist but no UI |
| Billing portal | ❌ Not started | No Stripe Customer Portal integration |

**Phase 4 dependencies:**
- Requires PostgreSQL migrations (user/org/subscription tables)
- Requires Phase 1-2 complete for production-ready query pipeline

---

### ❌ Phase 5 — Hardening & Enterprise (Week 17-20) — NOT STARTED

All components not started (SSO, audit logging, RAGAS evaluation, Terraform, etc.)

---

### ❌ Phase 6 — Optimization & Intelligence (Week 21+) — NOT STARTED

All components not started (ML model router, self-hosted embeddings, agentic workflows, etc.)

---

## Architecture Assessment

### What's Good ✅

1. **Clean modular structure** — Well-organized `src/` with clear separation (core, api, db, shared)
2. **AWS Bedrock integration** — Smart choice for embeddings + generation (Nova 2 Lite is cost-effective)
3. **Semantic chunking** — Thoughtful implementation with heading hierarchy and token management
4. **Qdrant integration** — Properly configured with payload indexing
5. **Dashboard UI** — Functional web UI with chat + search modes
6. **Crawler architecture** — Extensible crawler base with Internet Archive + HuggingFace sources
7. **Structured logging** — Uses structlog for structured logging

### What Needs Work ⚠️

1. **No hybrid retrieval** — Dense-only search will miss keyword queries (part numbers, error codes)
2. **No reranking** — Retrieval precision likely <70% without reranking
3. **No table support** — Technical manuals are table-heavy, this is a critical gap
4. **No caching** — Redis exists but not used, causing redundant embedding/retrieval
5. **No auth** — Cannot go to production without authentication
6. **No testing** — Limited test coverage, no CI integration
7. **No monitoring** — No metrics collection (Prometheus/Grafana)
8. **No cost tracking** — No visibility into per-query costs (Bedrock API calls)

### Technical Debt & Risks 🔴

1. **PostgreSQL not used** — Models defined but no migrations, no actual DB usage
2. **Celery tasks undefined** — Worker service exists but no async tasks implemented
3. **Model router not functional** — All queries use same model (no routing by complexity)
4. **No document metadata tracking** — No DB record of ingested documents
5. **No error handling** — Limited exception handling in API routes
6. **No rate limiting** — No protection against abuse
7. **No validation** — Minimal input validation on API endpoints

---

## Recommendations: What to Do Next

### 🚨 Critical Path — Get to Production-Ready Phase 2

#### Priority 1: Complete Core RAG Quality (2-3 weeks)

**Goal:** Achieve >80% retrieval precision and >90% answer faithfulness

1. **Implement hybrid retrieval** (1 week)
   - Add sparse vector encoding (start with Qdrant's built-in sparse support)
   - Implement reciprocal rank fusion (RRF) to merge dense + sparse results
   - Validate on test set: compare dense-only vs hybrid retrieval quality

2. **Add reranking** (3-5 days)
   - Integrate Cohere Rerank API (fastest path) or self-host `bge-reranker-v2-m3`
   - Rerank top-50 candidates → select top-8 for context
   - Measure precision improvement

3. **Implement table extraction** (1 week)
   - Use Camelot for PDFs with lattice/stream table detection
   - Extract tables to structured JSON + markdown + NL summary
   - Index tables in separate Qdrant collection or as table-type chunks
   - Update context builder to include table data

4. **Add query caching** (2-3 days)
   - Cache embedding vectors in Redis (TTL 1hr)
   - Cache retrieval results for identical queries
   - Measure latency reduction + cost savings

#### Priority 2: Database & Persistence (1 week)

**Goal:** Track documents, users, ingestion status in PostgreSQL

1. **Set up Alembic migrations**
   - Initialize Alembic: `alembic init migrations`
   - Create initial schema migration from `src/db/postgres/models.py`
   - Add document tracking table (id, title, source_file, status, chunks_count, created_at)

2. **Integrate PostgreSQL in ingestion pipeline**
   - Save document metadata after successful ingestion
   - Track ingestion status (pending, processing, completed, failed)
   - Add `/v1/documents` API to list/filter/delete documents

3. **Add basic error handling**
   - Wrap API routes in try/except with proper error responses
   - Log errors with structured logging
   - Return user-friendly error messages

#### Priority 3: Testing & Quality (1 week)

**Goal:** 60%+ test coverage, CI integration

1. **Install pytest and run existing tests**
   - Add `pytest` to pyproject.toml dev dependencies
   - Fix any broken tests
   - Run: `pytest -v`

2. **Add critical path tests**
   - Test chunking with sample PDFs
   - Test embedding batching + retry logic
   - Test Qdrant indexing + search
   - Test query pipeline end-to-end

3. **Set up GitHub Actions CI**
   - Run linting (ruff, mypy) on every PR
   - Run tests on every PR
   - Block merge if tests fail

#### Priority 4: Monitoring & Observability (3-5 days)

**Goal:** Understand system behavior in production

1. **Add Prometheus metrics**
   - Query latency histogram
   - Embedding API call count + latency
   - Qdrant search latency
   - Error rate by endpoint

2. **Add cost tracking**
   - Log Bedrock token usage per query
   - Calculate cost per query (Nova + Titan Embed pricing)
   - Track daily/weekly cost trends

3. **Set up Grafana dashboard** (optional)
   - Visualize key metrics
   - Set up alerts for high latency or error rates

---

### 🎯 Medium-Term Goals (Next 2-3 months)

1. **Authentication & Authorization** (Phase 4)
   - Implement JWT auth with refresh tokens
   - Add API key management for programmatic access
   - Enforce RBAC on API endpoints

2. **Billing & Subscriptions** (Phase 4)
   - Integrate Stripe for subscription management
   - Track usage per user/org
   - Implement rate limiting based on plan tier

3. **Scale Testing** (Phase 3)
   - Ingest 10K documents via bulk ingestion CLI
   - Load test with Locust (100 concurrent users)
   - Identify bottlenecks (Qdrant, Bedrock API, Redis)
   - Configure Qdrant sharding if needed

4. **Table-Aware RAG** (Phase 2)
   - Validate table extraction quality on 100 sample PDFs
   - Tune table detection thresholds
   - Implement table-specific query routing

---

### 🔮 Long-Term Vision (6+ months)

1. **Corpus Expansion**
   - Target: 50K → 200K → 400K documents
   - Implement de-duplication pipeline
   - Add manufacturer-specific crawlers (OEM portals)

2. **Advanced Features**
   - Multi-turn conversation with context memory
   - Document comparison ("diff these two manuals")
   - Automated question generation from docs
   - Multilingual support (start with Spanish, German)

3. **Self-Hosted Models**
   - Migrate embeddings to self-hosted BGE/GTE (cost optimization)
   - Add open LLM fallback (Llama 3.x) for cost ceiling

4. **Enterprise Hardening**
   - SSO integration (SAML, OIDC)
   - Audit logging (immutable event log)
   - Data residency controls
   - SOC 2 compliance

---

## Resource Estimates

### Phase 2 Completion (Production-Ready RAG)

**Timeline:** 4-6 weeks
**Effort:** 1 full-time engineer

**Breakdown:**
- Week 1-2: Hybrid retrieval + reranking
- Week 2-3: Table extraction + indexing
- Week 3-4: Query caching + PostgreSQL integration
- Week 4-5: Testing + CI setup
- Week 5-6: Monitoring + cost tracking

**Cost estimate (AWS):**
- Bedrock API (Nova + Titan Embed): ~$50-200/week for development
- Qdrant Cloud (optional): ~$50-100/month for managed service
- Other infra (Postgres, Redis, EC2): ~$100/month

---

## Key Decisions Needed

1. **Sparse retrieval approach:**
   - Option A: Use Qdrant's native sparse vectors (SPLADE encoding)
   - Option B: Add OpenSearch for BM25 (more infra complexity)
   - **Recommendation:** Start with Option A (simpler), migrate to OpenSearch if needed

2. **Reranker:**
   - Option A: Cohere Rerank API (fastest, $1/1K queries)
   - Option B: Self-hosted `bge-reranker-v2-m3` (one-time GPU cost)
   - **Recommendation:** Start with Cohere API, self-host at scale

3. **Table extraction:**
   - Option A: Camelot (lattice/stream detection, good for PDFs)
   - Option B: Unstructured.io (paid API, handles more formats)
   - **Recommendation:** Start with Camelot (open source)

4. **Testing strategy:**
   - Option A: Unit tests only (fast, limited coverage)
   - Option B: Unit + integration + e2e tests (comprehensive)
   - **Recommendation:** Prioritize integration tests for critical path (ingest → query)

5. **Deployment:**
   - Option A: AWS ECS Fargate (serverless containers)
   - Option B: Kubernetes (more control, more complexity)
   - Option C: AWS Lambda + API Gateway (for API only)
   - **Recommendation:** ECS Fargate (matches plan.md, good balance)

---

## Summary: Current State vs. Plan

| Phase | Plan Weeks | Status | Completion % |
|-------|------------|--------|--------------|
| Phase 1 (Foundation) | 1-4 | ✅ Mostly done | ~80% |
| Phase 2 (Hybrid + Tables) | 5-8 | ⚠️ Not started | ~15% |
| Phase 3 (Scale) | 9-12 | ❌ Not started | ~10% |
| Phase 4 (Auth + Billing) | 13-16 | ❌ Scaffolded only | ~5% |
| Phase 5 (Hardening) | 17-20 | ❌ Not started | 0% |
| Phase 6 (Optimization) | 21+ | ❌ Not started | 0% |

**Overall project completion:** ~30% (foundation solid, but critical features missing)

---

## Next Steps — Action Items

### Immediate (This Week)

1. ✅ Review this status report with team
2. 🔴 Decide on hybrid retrieval approach (Qdrant sparse vs OpenSearch)
3. 🔴 Set up Alembic migrations for PostgreSQL
4. 🔴 Install pytest and fix existing tests
5. 🔴 Add input validation to API endpoints

### Week 2-3

1. 🔴 Implement sparse vector encoding + RRF fusion
2. 🔴 Integrate Cohere Rerank API
3. 🔴 Add Redis caching for embeddings + retrieval
4. 🟡 Create evaluation test set (50-100 Q&A pairs)

### Week 4-6

1. 🔴 Implement table extraction pipeline
2. 🔴 Set up GitHub Actions CI
3. 🔴 Add Prometheus metrics + cost tracking
4. 🟡 Load test with 10K documents

### Month 2-3

1. 🟡 Implement JWT authentication
2. 🟡 Add document metadata tracking in PostgreSQL
3. 🟡 Create admin dashboard UI
4. 🟡 Set up production deployment (ECS Fargate + Terraform)

---

**Legend:**
- ✅ Complete
- ⚠️ Partial / In Progress
- ❌ Not Started
- 🔴 Critical Priority
- 🟡 Medium Priority
- 🟢 Low Priority / Nice to Have
