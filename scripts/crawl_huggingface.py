"""CLI script to download PDFs from a HuggingFace dataset.

Usage with a pre-configured dataset (recommended):
    python -m scripts.crawl_huggingface --config kaizen9/finepdfs_en --max-items 20
    python -m scripts.crawl_huggingface --config pixparse/pdfa-eng-wds --max-items 50

Usage with custom dataset (manual columns):
    python -m scripts.crawl_huggingface --dataset "my-org/manuals" --url-column link --max-items 50

List available configs:
    python -m scripts.crawl_huggingface --list-configs
"""

import argparse
import asyncio
import sys
from pathlib import Path

from src.core.crawler.sources.huggingface_datasets import (
    HuggingFaceCrawler,
    DATASET_CONFIGS,
    DatasetConfig,
)
from src.config import settings


async def main(
    config: DatasetConfig,
    max_items: int,
    output_dir: str,
) -> None:
    out = Path(output_dir) / "huggingface"
    crawler = HuggingFaceCrawler(
        output_dir=out,
        rate_limit_seconds=settings.crawler_rate_limit_seconds,
    )

    result = await crawler.crawl(config=config, max_items=max_items)

    print(f"\n{'='*60}")
    print(f"Crawl complete: HuggingFace ({config.dataset_name})")
    print(f"  Split:      {config.split}")
    print(f"  URL column: {config.url_column}")
    print(f"  Found:      {result.total_found}")
    print(f"  Downloaded: {result.downloaded}")
    print(f"  Skipped:    {result.skipped} (already downloaded)")
    print(f"  Failed:     {result.failed}")
    print(f"  Output dir: {out / crawler._safe_filename(config.dataset_name)}")
    print(f"{'='*60}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for err in result.errors[:10]:
            print(f"  - {err}")


def list_configs() -> None:
    """Print all pre-configured datasets."""
    print("Pre-configured HuggingFace datasets:\n")
    for name, cfg in DATASET_CONFIGS.items():
        print(f"  {name}")
        print(f"    {cfg.description}")
        print(f"    url_column={cfg.url_column}  split={cfg.split}")
        print()
    print("Run with:  python -m scripts.crawl_huggingface --config <name> --max-items 20")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download PDFs from a HuggingFace dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m scripts.crawl_huggingface --config kaizen9/finepdfs_en --max-items 20\n"
            "  python -m scripts.crawl_huggingface --config pixparse/pdfa-eng-wds --max-items 50\n"
            "  python -m scripts.crawl_huggingface --list-configs\n"
        ),
    )
    parser.add_argument(
        "--config",
        help="Name of a pre-configured dataset (see --list-configs)",
    )
    parser.add_argument(
        "--dataset",
        help="HuggingFace dataset name for custom (non-configured) datasets",
    )
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument(
        "--url-column", default="url", help="Column containing PDF URLs (default: url)"
    )
    parser.add_argument(
        "--title-column", default=None, help="Column for document title (optional)"
    )
    parser.add_argument("--subset", default=None, help="Dataset config/subset name (optional)")
    parser.add_argument(
        "--max-items", type=int, default=50, help="Max rows to process (default: 50)"
    )
    parser.add_argument(
        "--output-dir", default=settings.crawler_data_dir, help="Output directory"
    )
    parser.add_argument(
        "--list-configs", action="store_true", help="List pre-configured datasets and exit"
    )

    args = parser.parse_args()

    if args.list_configs:
        list_configs()
    elif args.config:
        if args.config not in DATASET_CONFIGS:
            parser.error(
                f"Unknown config '{args.config}'. "
                f"Available: {', '.join(DATASET_CONFIGS.keys())}. "
                f"Or use --dataset for a custom one."
            )
        cfg = DATASET_CONFIGS[args.config]
        asyncio.run(main(cfg, args.max_items, args.output_dir))
        sys.exit(0)
    elif args.dataset:
        cfg = DatasetConfig(
            dataset_name=args.dataset,
            url_column=args.url_column,
            split=args.split,
            title_column=args.title_column,
            subset=args.subset,
        )
        asyncio.run(main(cfg, args.max_items, args.output_dir))
        sys.exit(0)
    else:
        parser.error("Provide --config <name> or --dataset <hf_dataset>. See --list-configs.")
