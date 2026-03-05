"""CLI script to download sample manuals from Internet Archive.

Usage:
    python -m scripts.crawl_internet_archive --query "electronics manual" --max-items 50
    python -m scripts.crawl_internet_archive --query "service manual" --collection manuals --max-items 100
"""

import argparse
import asyncio
from pathlib import Path

from src.core.crawler.sources.internet_archive import InternetArchiveCrawler
from src.config import settings


async def main(query: str, collection: str, max_items: int, output_dir: str) -> None:
    out = Path(output_dir) / "internet_archive"
    crawler = InternetArchiveCrawler(
        output_dir=out,
        rate_limit_seconds=settings.crawler_rate_limit_seconds,
    )

    result = await crawler.crawl(
        query=query,
        collection=collection,
        max_items=max_items,
    )

    print(f"\n{'='*60}")
    print(f"Crawl complete: Internet Archive")
    print(f"  Query:      {query}")
    print(f"  Found:      {result.total_found}")
    print(f"  Downloaded: {result.downloaded}")
    print(f"  Skipped:    {result.skipped}")
    print(f"  Failed:     {result.failed}")
    print(f"  Output dir: {out}")
    print(f"{'='*60}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for err in result.errors[:10]:
            print(f"  - {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download manuals from Internet Archive")
    parser.add_argument("--query", default="manual", help="Search query (default: manual)")
    parser.add_argument("--collection", default="manuals", help="IA collection (default: manuals)")
    parser.add_argument("--max-items", type=int, default=50, help="Max items to download (default: 50)")
    parser.add_argument("--output-dir", default=settings.crawler_data_dir, help="Output directory")

    args = parser.parse_args()
    asyncio.run(main(args.query, args.collection, args.max_items, args.output_dir))
