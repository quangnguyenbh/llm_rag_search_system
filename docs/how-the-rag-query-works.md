# How the RAG Query Works — End-to-End Walkthrough

> This document explains what happens under the hood when a user asks a question on the ManualAI dashboard, from the browser click all the way to the generated answer.

---

## Architecture Overview

```
┌──────────┐    POST /v1/query     ┌──────────┐
│ Dashboard │ ───────────────────▶  │ FastAPI  │
│ (browser) │                       │ Route    │
└──────────┘                        └────┬─────┘
                                         │
                                    _build_pipeline()
                                         │
                    ┌────────────────────▼─────────────────────┐
                    │           QueryPipeline.execute()         │
                    │                                           │
                    │  1. Analyzer     → intent + complexity    │
                    │  2. Retriever    ──┐                      │
                    │     ├─ Embedder    │ embed question       │
                    │     │  (Titan V2)  │ → 1024-dim vector    │
                    │     └─ Qdrant     ◀┘ cosine search        │
                    │        returns 20 chunks                  │
                    │  3. Reranker     → top 8 (diverse)        │
                    │  4. ContextBuilder → "[Source N]..." text  │
                    │  5. ModelRouter  → nova-2-lite             │
                    │  6. Generator    → Bedrock Nova LLM       │
                    │     (system prompt + context + question)   │
                    │  7. CitationVerifier → validate [Source N] │
                    └──────────────────┬───────────────────────┘
                                       │
                                  JSON response
                                       │
                    ┌──────────────────▼───────────────────┐
                    │  { answer, citations, confidence }    │
                    └──────────────────────────────────────┘
```

Two external AWS Bedrock calls happen per query:

1. **Bedrock Titan Embed V2** — converts the question text into a vector (~100ms)
2. **Bedrock Nova 2 Lite** — generates the natural language answer from context (~2–5s)

---

## Step 1 — Dashboard (Browser JavaScript)

**File:** `dashboard/index.html`

When the user types a question and presses Enter (or clicks Send), the `send()` function fires:

```
User types question → send() → doChat(question)
```

`doChat()` makes an HTTP POST request to the FastAPI backend:

```javascript
const resp = await fetch('/v1/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: "How do I clean the inner cannula..." })
});
```

The dashboard also supports a **Search mode** (`/v1/query/search`) that returns raw chunks without LLM generation, and a **Streaming mode** (`/v1/query/stream`) that uses Server-Sent Events.

---

## Step 2 — FastAPI Route

**File:** `src/api/routes/query.py`

The request hits the `query_documents()` endpoint:

```python
@router.post("")
async def query_documents(request: QueryRequest):
    pipeline = _build_pipeline()
    result = await pipeline.execute(question=request.question)
```

`_build_pipeline()` wires up **7 components**:

| Component          | Class             | Responsibility                        |
|--------------------|-------------------|---------------------------------------|
| Analyzer           | `QueryAnalyzer`   | Classify intent & score complexity    |
| Retriever          | `Retriever`       | Embed query + search Qdrant           |
| Reranker           | `Reranker`        | Sort by score + enforce diversity     |
| Context Builder    | `ContextBuilder`  | Format chunks into a prompt string    |
| Generator          | `Generator`       | Call Bedrock Nova LLM                 |
| Citation Verifier  | `CitationVerifier` | Validate `[Source N]` references     |
| Model Router       | `ModelRouter`     | Pick the right model by complexity    |

---

## Step 3 — Pipeline Orchestration

**File:** `src/core/query/pipeline.py` → `QueryPipeline.execute()`

This is the heart of the system. It runs **7 steps in sequence**:

### 3a. Analyze the Query

**File:** `src/core/query/analyzer.py` → `QueryAnalyzer.analyze()`

Classifies the question using keyword matching:

| Keywords detected                          | Intent          |
|--------------------------------------------|-----------------|
| `"how to"`, `"steps"`, `"procedure"`, `"install"` | **procedural**  |
| `"error"`, `"fix"`, `"troubleshoot"`       | **troubleshoot** |
| `"compare"`, `"vs"`, `"difference"`        | **comparative**  |
| none of the above                          | **factual**      |

Scores **complexity** by word count:

| Word count | Complexity |
|------------|------------|
| < 8        | 0.2 (simple) |
| 8–20       | 0.5 (medium) |
| > 20       | 0.8 (complex) |

**Example:** *"How do I clean the inner cannula of a tracheostomy tube?"* → `intent="procedural"`, `complexity=0.5`

---

### 3b. Retrieve 20 Candidate Chunks from Qdrant

**Files:** `src/core/query/retriever.py` → `Retriever.search()`, `src/core/ingestion/embedder.py`, `src/db/vector/qdrant_client.py`

This step has two sub-steps:

#### 1. Embed the question into a 1024-dimensional vector

The `Retriever` calls `BatchEmbedder.embed_query()`, which sends the question to **Amazon Titan Embed V2** on Bedrock:

```json
{
  "inputText": "How do I clean the inner cannula of a tracheostomy tube?",
  "dimensions": 1024,
  "normalize": true
}
```

Bedrock returns a 1024-float vector: `[0.0234, -0.0891, 0.1245, ...]`

This vector is a numerical representation of the question's **meaning** — semantically similar texts will have similar vectors.

#### 2. Query Qdrant with cosine similarity search

The `Retriever` passes the vector to `search_chunks()` in `src/db/vector/qdrant_client.py`, which calls Qdrant's `query_points()`:

```python
results = client.query_points(
    collection_name="manual_chunks",
    query=query_vector,      # the 1024-dim vector
    limit=20,                # return top 20
    with_payload=True,       # include text + metadata
)
```

**How Qdrant finds matches:**

Qdrant compares the query vector against **every stored chunk vector** using cosine similarity:

$$\text{similarity} = \cos(\theta) = \frac{\vec{q} \cdot \vec{c}}{|\vec{q}| \cdot |\vec{c}|}$$

Where:
- $\vec{q}$ = the query vector (your question)
- $\vec{c}$ = a stored chunk vector (a piece of a PDF)

The score ranges from 0.0 (completely unrelated) to 1.0 (identical meaning). Qdrant uses indexing (HNSW graph) to do this efficiently without brute-force comparison.

**Example results:**

| Chunk content                              | Score |
|--------------------------------------------|-------|
| "cleaning inner cannula...twist clockwise" | 0.735 |
| "hold the neck flange steady..."           | 0.723 |
| "fire sprinkler maintenance procedure"     | 0.150 |

Each result includes the chunk's `text`, `document_id`, `title`, `page_number`, `section_path`, and `source_file` from the stored payload.

---

### 3c. Rerank Down to 8 Chunks

**File:** `src/core/query/reranker.py` → `Reranker.rerank()`

The reranker takes the 20 candidates and selects the **top 8** with two goals:

1. **Sort by score** — highest similarity first
2. **Enforce diversity** — no single document can take more than `top_k // 2 = 4` slots

This prevents one long PDF from monopolizing all 8 context slots, ensuring the LLM sees evidence from multiple sources when available.

```python
max_per_doc = max(top_k // 2, 2)  # = 4

for chunk in sorted_chunks:
    if doc_counts[chunk.document_id] < max_per_doc:
        selected.append(chunk)
```

---

### 3d. Build Context String

**File:** `src/core/query/context_builder.py` → `ContextBuilder.build()`

Formats the 8 reranked chunks into a single text string with numbered source headers:

```
[Source 1: pe2569s, Page 4]
Reinsert the clean twist-lock inner cannula into the tube. Secure it by
gently twisting it clockwise until the blue dot on the inner cannula
lines up with the blue dot on the tube.

---

[Source 2: pe2569s, Page 2]
Hold the neck flange steady in your child with one hand. With the other
hand, grasp the twist lock inner cannula connector and carefully unlock
it by turning counterclockwise.

---

[Source 3: ...]
...
```

The `[Source N]` numbering is critical — the LLM will reference these numbers in its answer.

---

### 3e. Route to Model

**File:** `src/core/query/model_router.py` → `ModelRouter.select()`

Selects the LLM based on complexity:

| Complexity       | Tier       | Model                         |
|------------------|------------|-------------------------------|
| < 0.3 (simple)  | FAST       | `us.amazon.nova-micro-v1:0`   |
| 0.3–0.7 (medium)| STANDARD   | `us.amazon.nova-2-lite-v1:0`  |
| > 0.7 (complex) | HEAVY      | `us.amazon.nova-2-lite-v1:0`  |

Comparative queries are always bumped to at least STANDARD tier.

**Example:** complexity 0.5 → STANDARD → `us.amazon.nova-2-lite-v1:0`

---

### 3f. Generate Answer via LLM

**File:** `src/core/query/generator.py` → `Generator.generate()`

Calls **Amazon Nova 2 Lite** on Bedrock with the assembled prompt:

```json
{
  "system": [{"text": "You are ManualAI, a technical documentation assistant..."}],
  "messages": [
    {
      "role": "user",
      "content": [{"text": "Context from retrieved documents:\n[Source 1: pe2569s, Page 4]\nReinsert the clean...\n---\n[Source 2: ...]\n...\n---\nQuestion: How do I clean the inner cannula of a tracheostomy tube?\n\nPlease answer the question using ONLY the context above. Cite sources using [Source N] notation."}]
    }
  ],
  "inferenceConfig": {"maxTokens": 4096}
}
```

**The system prompt** enforces these rules:

1. Only use information from the provided context
2. Cite every claim with `[Source N]`
3. If context is insufficient, say so honestly
4. Never fabricate part numbers, specs, or procedures
5. Prefer bullet points for multi-step procedures

Nova reads the context, synthesizes a coherent answer, and cites `[Source N]` tags as instructed. The response includes the answer text and token usage statistics.

---

### 3g. Verify Citations

**File:** `src/core/query/citation.py` → `CitationVerifier.verify()`

Parses the generated answer to validate citations:

1. **Extract** all `[Source N]` references using regex: `\[Source\s+(\d+)\]`
2. **Map** each cited number back to the actual chunk at that position
3. **Calculate confidence**:

$$\text{confidence} = \text{avg\_retrieval\_score} \times \text{citation\_validity\_ratio}$$

Where `citation_validity_ratio` = (number of valid citations) / (total citations found).

**Example:** If 3 citations are found and all 3 map to real chunks, and the average retrieval score is 0.65, then confidence = 0.65 × 1.0 = 0.65.

Returns the verified citations with full metadata (title, page number, score, text preview).

---

## Step 4 — Response Back to Dashboard

FastAPI serializes the `QueryResult` into JSON:

```json
{
  "answer": "To clean the inner cannula...\n1. Wash your hands [Source 4]...",
  "citations": [
    {
      "source_index": 1,
      "title": "pe2569s",
      "page_number": 4,
      "score": 0.735,
      "text_preview": "Reinsert the clean twist-lock inner cannula..."
    },
    {
      "source_index": 2,
      "title": "pe2569s",
      "page_number": 2,
      "score": 0.723,
      "text_preview": "Hold the neck flange steady in your child..."
    }
  ],
  "confidence": 0.448,
  "model_used": "us.amazon.nova-2-lite-v1:0"
}
```

The dashboard JavaScript receives this and renders:
- The **answer** with basic markdown formatting (bold, code, line breaks)
- **Source badges** showing `[Source N] Title — Page X (score%)`
- A **metadata bar** with model name and confidence percentage
- A **sidebar** panel listing all cited sources

---

## API Endpoints Summary

| Endpoint             | Method | Purpose                                  |
|----------------------|--------|------------------------------------------|
| `POST /v1/query`     | POST   | Full RAG: retrieve + generate answer     |
| `POST /v1/query/search` | POST | Vector search only (no LLM generation) |
| `POST /v1/query/stream` | POST | Full RAG with SSE streaming response   |
| `GET /health`        | GET    | Health check                             |

---

## Key Technologies

| Component        | Technology                     | Purpose                          |
|------------------|--------------------------------|----------------------------------|
| Embedding model  | Amazon Titan Embed V2          | Convert text → 1024-dim vectors  |
| Vector database  | Qdrant Cloud (us-east-1)       | Store & search chunk vectors     |
| LLM              | Amazon Nova 2 Lite (Bedrock)   | Generate answers from context    |
| API framework    | FastAPI + uvicorn              | HTTP endpoints                   |
| Dashboard        | Vanilla HTML/CSS/JS            | Chat & search UI                 |
