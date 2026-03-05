"""HuggingFace Datasets crawler for downloading PDFs from dataset repositories.

Many HuggingFace datasets contain direct PDF links or bundled PDF files that
can serve as a corpus source. This crawler uses the `datasets` library to
stream rows from a dataset, extract PDF URLs from a configurable column, and
download them locally.

It maintains a `downloaded.jsonl` ledger in the output directory so that
subsequent runs skip already-downloaded URLs (resume support).

Usage:
    crawler = HuggingFaceCrawler(output_dir=Path("./data/raw/huggingface"))
    result = await crawler.crawl(config=DATASET_CONFIGS["kaizen9/finepdfs_en"], max_items=200)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx
import structlog

from src.core.crawler.base import BaseCrawler, CrawlResult

logger = structlog.get_logger()


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a single HuggingFace dataset source."""

    dataset_name: str
    url_column: str
    split: str = "train"
    title_column: str | None = None
    subset: str | None = None
    description: str = ""


# ---------------------------------------------------------------------------
# Pre-configured datasets — add new entries here
# ---------------------------------------------------------------------------
DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "kaizen9/finepdfs_en": DatasetConfig(
        dataset_name="kaizen9/finepdfs_en",
        url_column="url",
        split="train",
        title_column=None,
        description="English PDF documents with text, id, dump, and direct URL to PDF files",
    ),
    "pixparse/pdfa-eng-wds": DatasetConfig(
        dataset_name="pixparse/pdfa-eng-wds",
        url_column="pdf_url",
        split="train",
        title_column="title",
        description="English PDF-A documents from Common Crawl (web-dataset shards)",
    ),
    "HuggingFaceFW/fineweb": DatasetConfig(
        dataset_name="HuggingFaceFW/fineweb",
        url_column="url",
        split="train",
        title_column=None,
        description="Large-scale web crawl; filter for PDF mime types",
    ),
}

# Keep a simple alias for backward-compat
KNOWN_DATASETS: dict[str, dict] = {
    name: {
        "description": cfg.description,
        "url_column": cfg.url_column,
        "title_column": cfg.title_column,
        "split": cfg.split,
    }
    for name, cfg in DATASET_CONFIGS.items()
}


class _DownloadLedger:
    """Append-only JSONL ledger that tracks which URLs have been downloaded.

    Each line is a JSON object: {"url": "...", "file": "...", "ts": "..."}
    """

    def __init__(self, ledger_path: Path):
        self._path = ledger_path
        self._seen: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        for line in self._path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                self._seen.add(entry["url"])
            except (json.JSONDecodeError, KeyError):
                continue

    def is_downloaded(self, url: str) -> bool:
        return url in self._seen

    def record(self, url: str, file_path: str) -> None:
        from datetime import datetime, timezone

        entry = {
            "url": url,
            "file": file_path,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        with self._path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        self._seen.add(url)

    @property
    def count(self) -> int:
        return len(self._seen)


class HuggingFaceCrawler(BaseCrawler):
    """Download PDFs referenced in HuggingFace datasets.

    Works with any HF dataset that has a column containing direct PDF URLs.
    Uses streaming mode to avoid downloading the entire dataset locally first.
    Tracks downloaded URLs in a JSONL ledger for resume support.
    """

    def __init__(
        self,
        output_dir: Path,
        rate_limit_seconds: float = 1.0,
        timeout_seconds: float = 60.0,
    ):
        super().__init__(output_dir, rate_limit_seconds)
        self.timeout = timeout_seconds

    async def crawl(
        self,
        config: DatasetConfig | None = None,
        *,
        # Legacy kwargs — used when config is None
        dataset_name: str | None = None,
        split: str = "train",
        url_column: str = "url",
        title_column: str | None = None,
        name: str | None = None,
        max_items: int = 100,
    ) -> CrawlResult:
        """Stream a HuggingFace dataset and download PDFs from a URL column.

        Args:
            config: A DatasetConfig object (preferred). If provided, the
                    legacy keyword arguments are ignored.
            dataset_name: HuggingFace dataset identifier (legacy).
            split: Dataset split (legacy).
            url_column: Column containing PDF URLs (legacy).
            title_column: Column for document title (legacy).
            name: Dataset config/subset (legacy).
            max_items: Maximum number of *rows to process* (threshold).
                       The crawler stops after scanning this many rows.

        Returns:
            CrawlResult with download statistics and file paths.
        """
        from datasets import load_dataset

        # Resolve config
        if config is not None:
            dataset_name = config.dataset_name
            split = config.split
            url_column = config.url_column
            title_column = config.title_column
            name = config.subset
        elif dataset_name is None:
            raise ValueError("Either config or dataset_name must be provided")

        result = CrawlResult(source=f"huggingface:{dataset_name}")

        # Per-dataset output sub-directory
        ds_dir = self.output_dir / self._safe_filename(dataset_name)
        ds_dir.mkdir(parents=True, exist_ok=True)

        # Download ledger for resume support
        ledger = _DownloadLedger(ds_dir / "downloaded.jsonl")

        logger.info(
            "hf_crawl.start",
            dataset=dataset_name,
            split=split,
            url_column=url_column,
            max_items=max_items,
            already_downloaded=ledger.count,
        )

        # Stream the dataset so we don't download everything upfront
        try:
            ds = load_dataset(
                dataset_name,
                name=name,
                split=split,
                streaming=True,
                trust_remote_code=False,
            )
        except Exception as e:
            logger.error("hf_crawl.load_failed", dataset=dataset_name, error=str(e))
            result.errors.append(f"Failed to load dataset: {e}")
            return result

        # Iterate and download PDFs
        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": "llm-rag-search-system/0.1 (research)"},
        ) as client:
            rows_processed = 0
            for row in ds:
                if rows_processed >= max_items:
                    break
                rows_processed += 1

                url = row.get(url_column)
                if not url or not isinstance(url, str):
                    continue

                # Basic URL validation
                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https"):
                    continue

                result.total_found += 1

                # Skip if already downloaded in a previous run
                if ledger.is_downloaded(url):
                    result.skipped += 1
                    continue

                title = (
                    row.get(title_column, "") if title_column else ""
                ) or self._title_from_url(url)

                try:
                    pdf_path = await self._download_pdf(client, url, title, row, ds_dir)
                    if pdf_path:
                        ledger.record(url, str(pdf_path))
                        result.downloaded += 1
                        result.files.append(pdf_path)
                        logger.info(
                            "hf_crawl.downloaded",
                            url=url[:120],
                            path=str(pdf_path),
                        )
                    else:
                        result.skipped += 1
                except Exception as e:
                    result.failed += 1
                    result.errors.append(f"{url[:120]}: {e}")
                    logger.warning("hf_crawl.download_failed", url=url[:120], error=str(e))

                self._rate_limit()

        logger.info(
            "hf_crawl.complete",
            dataset=dataset_name,
            rows_processed=rows_processed,
            found=result.total_found,
            downloaded=result.downloaded,
            skipped=result.skipped,
            failed=result.failed,
        )
        return result

    async def _download_pdf(
        self,
        client: httpx.AsyncClient,
        url: str,
        title: str,
        row: dict,
        ds_dir: Path,
    ) -> Path | None:
        """Download a single PDF from a URL."""
        safe_name = self._safe_filename(title) or self._safe_filename(url.split("/")[-1])
        if not safe_name:
            safe_name = self._safe_filename(str(hash(url)))

        out_path = ds_dir / f"{safe_name}.pdf"

        # Skip if file already on disk (belt-and-suspenders with ledger)
        if out_path.exists():
            logger.debug("hf_crawl.already_exists", path=str(out_path))
            return out_path

        resp = await client.get(url)
        resp.raise_for_status()

        # Ensure it's actually a PDF (check magic bytes or content-type header)
        content_type = resp.headers.get("content-type", "")
        is_pdf = (
            "application/pdf" in content_type
            or resp.content[:5] == b"%PDF-"
        )
        if not is_pdf:
            logger.debug("hf_crawl.not_pdf", url=url[:120], content_type=content_type)
            return None

        out_path.write_bytes(resp.content)

        # Write metadata sidecar
        meta_path = ds_dir / f"{safe_name}.meta.json"
        meta = {
            "title": title,
            "source": f"huggingface:{row.get('__dataset_name__', 'unknown')}",
            "source_url": url,
            "dataset_row": {
                k: v
                for k, v in row.items()
                if isinstance(v, (str, int, float, bool)) and k != url
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        return out_path

    @staticmethod
    def _title_from_url(url: str) -> str:
        """Extract a reasonable title from a URL path."""
        path = urlparse(url).path
        filename = path.rsplit("/", 1)[-1] if "/" in path else path
        # Strip .pdf extension for title
        if filename.lower().endswith(".pdf"):
            filename = filename[:-4]
        return filename or "untitled"
