# Scripts — How to Run

All scripts are run from the project root with the virtual environment activated:

```bash
cd /path/to/llm_rag_search_system
source .venv/bin/activate
```

---

## 1. Crawl — HuggingFace Datasets

Download PDFs from a pre-configured HuggingFace dataset. Files are saved to
`data/raw/huggingface/<dataset_name>/` with `.meta.json` sidecars and a
`downloaded.jsonl` ledger for resume support.

```bash
# List all pre-configured datasets
python -m scripts.crawl_huggingface --list-configs

# Download 20 PDFs from kaizen9/finepdfs_en
python -m scripts.crawl_huggingface --config kaizen9/finepdfs_en --max-items 20

# Resume later — already-downloaded URLs are skipped automatically
python -m scripts.crawl_huggingface --config kaizen9/finepdfs_en --max-items 100

# Use a different pre-configured dataset
python -m scripts.crawl_huggingface --config pixparse/pdfa-eng-wds --max-items 50

# Use a custom (non-configured) dataset
python -m scripts.crawl_huggingface --dataset "my-org/manuals" --url-column link --max-items 50
```

| Flag | Description | Default |
|---|---|---|
| `--config` | Name of a pre-configured dataset (see `--list-configs`) | — |
| `--dataset` | HuggingFace dataset ID for custom datasets | — |
| `--max-items` | Max rows to process (threshold for large datasets) | 50 |
| `--split` | Dataset split | `train` |
| `--url-column` | Column containing PDF URLs (custom datasets only) | `url` |
| `--title-column` | Column for document title (optional) | — |
| `--subset` | Dataset config/subset name (optional) | — |
| `--output-dir` | Base output directory | `./data/raw` |

**Makefile shortcut:**

```bash
make crawl-hf args="--config kaizen9/finepdfs_en --max-items 20"
```

---

## 2. Crawl — Internet Archive

Download manuals from the Internet Archive's public collections. Files are saved
to `data/raw/internet_archive/`.

```bash
# Download 50 electronics manuals
python -m scripts.crawl_internet_archive --query "electronics manual" --max-items 50

# Search a specific collection
python -m scripts.crawl_internet_archive --query "service manual" --collection manuals --max-items 100
```

| Flag | Description | Default |
|---|---|---|
| `--query` | Search query string | `manual` |
| `--collection` | Internet Archive collection to search | `manuals` |
| `--max-items` | Maximum items to download | 50 |
| `--output-dir` | Output directory | `./data/raw` |

**Makefile shortcut:**

```bash
make crawl-ia args="--query 'electronics manual' --max-items 50"
```

---

## 3. Bulk Ingest

Ingest downloaded documents into the RAG pipeline (parse, chunk, embed, index).

```bash
# Ingest all documents from a directory
python -m scripts.bulk_ingest --input ./data/raw/huggingface/kaizen9_finepdfs_en/

# Ingest Internet Archive downloads
python -m scripts.bulk_ingest --input ./data/raw/internet_archive/ --batch-size 20
```

| Flag | Description | Default |
|---|---|---|
| `--input` | Input directory containing documents (required) | — |
| `--batch-size` | Batch size for embedding API calls | 10 |

**Makefile shortcut:**

```bash
make ingest args="--input ./data/raw/huggingface/kaizen9_finepdfs_en/"
```

> **Note:** The ingestion pipeline is under development. The script currently
> discovers supported files but does not yet run the full embed/index flow.

---

## 4. Seed Database

Populate the database with initial data (users, orgs, sample documents).

```bash
python -m scripts.seed_db
```

> **Status:** Placeholder — not yet implemented.

---

## 5. Evaluate (RAGAS)

Run the RAGAS evaluation pipeline against a test set to measure answer
faithfulness, relevance, and retrieval quality.

```bash
python -m scripts.evaluate
```

> **Status:** Placeholder — not yet implemented.

---

## 6. Benchmark

Run retrieval quality benchmarks (precision@k, recall@k, MRR) on a curated
test set.

```bash
python -m scripts.benchmark
```

> **Status:** Placeholder — not yet implemented.

---

## Makefile Reference

All common operations are available as Make targets:

```bash
make dev              # Start FastAPI dev server (port 8000)
make test             # Run pytest with coverage
make lint             # Ruff lint check
make format           # Ruff auto-format
make docker-up        # Start all services (Docker Compose)
make docker-down      # Stop all services
make migrate          # Run Alembic migrations
make migrate-create msg="description"  # Create a new migration
make ingest args="--input ./data/raw/..."
make crawl-ia args="--query '...' --max-items 50"
make crawl-hf args="--config kaizen9/finepdfs_en --max-items 20"
make worker           # Start Celery worker (not yet configured)
```
