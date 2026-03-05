# ADR-001: Semantic Chunking Strategy

**Status:** Accepted  
**Date:** 2025-01-20  
**Deciders:** Core team  

## Context

Our RAG platform ingests 400K+ digital manuals (PDFs) and needs to split them into chunks suitable for embedding with `text-embedding-3-large` (8192-token context window). Chunk quality directly impacts retrieval precision and answer quality — chunks that are too large dilute relevance, chunks that are too small lose context.

PDFs in our corpus have structure: headings, sections, tables, figures. Naively splitting on fixed character or token counts ignores this structure and produces chunks that straddle topics, harming retrieval.

## Decision

We adopt a **structure-first semantic chunking** strategy with the following pipeline:

```
PDF → Structure Parsing (headings, font-size detection)
    → Heading-delimited Sections
    → Small-section Merging
    → Paragraph-boundary Splitting (target 512 tokens)
    → Intra-section Overlap (100 tokens)
    → Contextual Header Prepending
```

### Key Parameters

| Parameter    | Value  | Rationale |
|-------------|--------|-----------|
| target_size | 512    | Sweet spot for embedding models; fits ~1 paragraph of dense technical content |
| overlap     | 100    | ~20% overlap preserves sentence-level context at chunk boundaries |
| min_size    | 100    | Sections below this are merged into the previous section |
| max_size    | 1024   | Hard cap; paragraphs exceeding this are force-split |

### Design Rules

1. **Split at heading boundaries first.** Heading detection uses PyMuPDF font-size (>14pt) and bold detection. Each heading starts a new section, preserving topic coherence.

2. **Merge small sections.** Sections under `min_size` tokens are absorbed into the preceding section rather than becoming tiny standalone chunks.

3. **Split within sections at paragraph boundaries.** When a section exceeds `target_size`, we accumulate paragraphs until the target is reached, then emit a chunk.

4. **Overlap is intra-section only.** We apply 100-token overlap between consecutive chunks from the *same* section. We do NOT overlap across heading boundaries — different topics should not bleed into each other.

5. **Contextual header on every chunk.** Each chunk is prefixed with:  
   `Document: {title} | Section: {section_path}`  
   This gives the embedding model and LLM context about where the chunk came from, improving both retrieval and answer grounding.

6. **Force-split for degenerate paragraphs.** Single paragraphs exceeding `max_size` are split at sentence boundaries (`. ! ?`), with a word-level fallback for text lacking punctuation.

### Tokenizer

We use `tiktoken` with the `cl100k_base` encoding, which matches the tokenizer used by `text-embedding-3-large`. This ensures our token counts are accurate for the embedding model.

## Alternatives Considered

### Fixed-size character/token splitting
Simple but ignores document structure. Produces chunks that split mid-sentence or mid-paragraph, harming retrieval quality together with coherence.

### Recursive text splitting (LangChain-style)
Splits by separator hierarchy (`\n\n` → `\n` → `. ` → ` `). Better than fixed-size but still structure-unaware — cannot distinguish a heading from a paragraph.

### Full-page chunks
One chunk per PDF page. Simple but page boundaries are arbitrary (mid-paragraph, mid-table). Pages vary wildly in token count.

### Sliding window with no structure awareness
Fixed window + overlap. Produces uniform chunks but frequently straddles section boundaries, leading to topically incoherent chunks.

## Consequences

### Positive
- Chunks are topically coherent (aligned to document sections)
- Overlap preserves sentence context at chunk boundaries without mixing topics
- Contextual headers improve retrieval relevance and answer attribution
- Predictable token sizes (512 ± variance) work well with embedding models
- Force-split handles degenerate edge cases gracefully

### Negative
- Heading detection is heuristic (font-size + bold) and may miss some headings or false-positive on styled text
- Very short documents (< min_size) produce a single chunk with high header-to-content ratio
- Paragraph grouping assumes blank-line separation; some PDFs have irregular whitespace

### Risks
- Heading detection quality varies across PDF generators; may need per-source tuning
- Token counting adds ~10% overhead to chunking; acceptable for batch ingestion

## Implementation

- Module: `src/core/ingestion/chunker.py`
- Class: `SemanticChunker`
- Tests: `tests/unit/test_chunker.py` (18 tests)
- Validated on real PDFs from `kaizen9/finepdfs_en` dataset
