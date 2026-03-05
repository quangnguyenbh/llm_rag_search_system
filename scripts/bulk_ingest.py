"""CLI for bulk document ingestion.

Usage:
    python -m scripts.bulk_ingest --input ./data/raw/internet_archive/
"""

import argparse
import asyncio
from pathlib import Path

from src.shared.constants import SUPPORTED_EXTENSIONS


async def main(input_dir: str, batch_size: int) -> None:
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: {input_dir} does not exist")
        return

    # Collect all supported files
    files = [
        f for f in input_path.rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    print(f"Found {len(files)} documents to ingest")

    # TODO: Initialize ingestion pipeline
    # TODO: Process in batches
    # TODO: Track progress and report

    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {file_path.name} — TODO: ingest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk document ingestion")
    parser.add_argument("--input", required=True, help="Input directory containing documents")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for embedding API calls")

    args = parser.parse_args()
    asyncio.run(main(args.input, args.batch_size))
