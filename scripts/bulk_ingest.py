"""Bulk document ingestion with resume support, concurrency control, and progress tracking.

Usage:
    PYTHONPATH=. python -m scripts.bulk_ingest --input ./data/raw --concurrency 3
    PYTHONPATH=. python -m scripts.bulk_ingest --input ./data/raw --concurrency 5 --dry-run
    PYTHONPATH=. python -m scripts.bulk_ingest --input ./data/raw --concurrency 3 --collection manual_chunks

Re-run the same command to resume — already-ingested files are skipped via the JSONL ledger.
"""

import argparse
import asyncio
import json
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from src.config import settings
from src.core.ingestion.chunker import SemanticChunker
from src.core.ingestion.embedder import BatchEmbedder
from src.core.ingestion.metadata import MetadataExtractor
from src.core.ingestion.pipeline import IngestionPipeline
from src.db.vector.qdrant_client import get_qdrant_client, init_collection
from src.shared.constants import SUPPORTED_EXTENSIONS

LEDGER_FILENAME = "ingestion_ledger.jsonl"


# ── Ledger ────────────────────────────────────────────────────────────────────

def load_ledger(ledger_path: Path) -> dict[str, dict]:
    """Load ledger from JSONL. Returns {file_path_str: entry_dict}."""
    entries: dict[str, dict] = {}
    if not ledger_path.exists():
        return entries
    with open(ledger_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entries[entry["file_path"]] = entry
    return entries


def append_ledger(ledger_path: Path, entry: dict) -> None:
    """Append a single entry to the JSONL ledger."""
    with open(ledger_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def make_ledger_entry(
    file_path: str,
    status: str,
    document_id: str = "",
    title: str = "",
    chunks: int = 0,
    error: str = "",
) -> dict:
    return {
        "file_path": file_path,
        "status": status,
        "document_id": document_id,
        "title": title,
        "chunks": chunks,
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Sidecar metadata ─────────────────────────────────────────────────────────

def load_sidecar_metadata(file_path: Path) -> dict | None:
    """Load .meta.json sidecar if it exists next to the file."""
    meta_path = file_path.with_suffix(file_path.suffix + ".meta.json")
    if not meta_path.exists():
        # Also try replacing the full suffix
        meta_path = file_path.with_suffix(".meta.json")
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


# ── Ingestion ─────────────────────────────────────────────────────────────────

async def ingest_one(
    pipeline: IngestionPipeline,
    file_path: Path,
    ledger_path: Path,
    semaphore: asyncio.Semaphore,
) -> str:
    """Ingest a single file with semaphore gating. Returns status string."""
    async with semaphore:
        try:
            doc_metadata = load_sidecar_metadata(file_path)
            result = await pipeline.ingest(file_path, doc_metadata)
            append_ledger(
                ledger_path,
                make_ledger_entry(
                    file_path=str(file_path),
                    status="success",
                    document_id=result.document_id,
                    title=result.title,
                    chunks=result.chunks_count,
                ),
            )
            return "success"
        except Exception as e:
            append_ledger(
                ledger_path,
                make_ledger_entry(
                    file_path=str(file_path),
                    status="failed",
                    error=f"{type(e).__name__}: {e}",
                ),
            )
            tqdm.write(f"  FAILED: {file_path.name} — {type(e).__name__}: {e}")
            return "failed"


async def main(
    input_dir: str,
    concurrency: int,
    collection: str | None,
    dry_run: bool,
) -> None:
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: {input_dir} does not exist")
        return

    # Discover files
    files = sorted(
        f for f in input_path.rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    print(f"Found {len(files)} documents in {input_dir}")

    # Load ledger — skip already-succeeded files
    ledger_path = input_path / LEDGER_FILENAME
    ledger = load_ledger(ledger_path)
    succeeded = {fp for fp, e in ledger.items() if e["status"] == "success"}

    to_ingest = [f for f in files if str(f) not in succeeded]
    skipped = len(files) - len(to_ingest)

    print(f"  Skipping {skipped} already-ingested (ledger: {ledger_path.name})")
    print(f"  To ingest: {len(to_ingest)}")

    if dry_run:
        print("\n[DRY RUN] Would ingest:")
        for f in to_ingest[:20]:
            print(f"  {f}")
        if len(to_ingest) > 20:
            print(f"  ... and {len(to_ingest) - 20} more")
        return

    if not to_ingest:
        print("\nNothing to ingest. All files already processed.")
        return

    # Override collection if specified
    if collection:
        settings.qdrant_collection = collection

    # Initialize pipeline components
    qdrant = get_qdrant_client()
    init_collection(qdrant)
    pipeline = IngestionPipeline(
        chunker=SemanticChunker(),
        metadata_extractor=MetadataExtractor(),
        embedder=BatchEmbedder(),
        qdrant=qdrant,
    )
    semaphore = asyncio.Semaphore(concurrency)

    print(f"\nIngesting {len(to_ingest)} documents (concurrency={concurrency}, "
          f"collection={settings.qdrant_collection})\n")

    # Process with progress bar
    counts = {"success": 0, "failed": 0}
    start_time = time.monotonic()

    with tqdm(total=len(to_ingest), unit="doc", dynamic_ncols=True) as pbar:
        tasks = []
        for file_path in to_ingest:
            task = asyncio.create_task(
                ingest_one(pipeline, file_path, ledger_path, semaphore)
            )
            tasks.append(task)

        for coro in asyncio.as_completed(tasks):
            status = await coro
            counts[status] += 1
            pbar.update(1)
            pbar.set_postfix(ok=counts["success"], fail=counts["failed"])

    elapsed = time.monotonic() - start_time

    # Summary
    print(f"\n{'=' * 50}")
    print(f"  Bulk Ingestion Complete")
    print(f"{'=' * 50}")
    print(f"  Success:  {counts['success']}")
    print(f"  Failed:   {counts['failed']}")
    print(f"  Skipped:  {skipped} (already ingested)")
    print(f"  Total:    {len(files)}")
    print(f"  Time:     {elapsed:.1f}s ({elapsed/max(len(to_ingest),1):.2f}s/doc)")
    print(f"  Ledger:   {ledger_path}")
    print(f"  Collection: {settings.qdrant_collection}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bulk document ingestion into Qdrant via Bedrock embeddings"
    )
    parser.add_argument("--input", required=True, help="Input directory containing documents")
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent document ingestions")
    parser.add_argument("--collection", type=str, default=None, help="Override Qdrant collection name")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested without doing it")

    args = parser.parse_args()
    asyncio.run(main(args.input, args.concurrency, args.collection, args.dry_run))
