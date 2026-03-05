"""Internet Archive crawler for downloading manuals from archive.org.

Uses the `internetarchive` library to search and download documents from
the Internet Archive's manuals collection.

Usage:
    crawler = InternetArchiveCrawler(output_dir=Path("./data/raw/internet_archive"))
    result = await crawler.crawl(query="electronics manual", max_items=100)
"""

from pathlib import Path

import httpx
import structlog

from src.core.crawler.base import BaseCrawler, CrawlResult

logger = structlog.get_logger()

# Internet Archive search API
IA_SEARCH_URL = "https://archive.org/advancedsearch.php"
IA_METADATA_URL = "https://archive.org/metadata"
IA_DOWNLOAD_URL = "https://archive.org/download"

# Default search parameters for manuals
DEFAULT_COLLECTION = "manuals"
ALLOWED_FORMATS = {"PDF", "Text PDF"}


class InternetArchiveCrawler(BaseCrawler):
    """Download manuals from the Internet Archive's manuals collection.

    The Internet Archive hosts millions of freely available manuals. This crawler
    searches the collection, filters for PDF documents, and downloads them to a
    local directory for pipeline development and testing.
    """

    def __init__(
        self,
        output_dir: Path,
        rate_limit_seconds: float = 2.0,
        timeout_seconds: float = 60.0,
    ):
        super().__init__(output_dir, rate_limit_seconds)
        self.timeout = timeout_seconds

    async def crawl(
        self,
        query: str = "manual",
        collection: str = DEFAULT_COLLECTION,
        max_items: int = 100,
        media_type: str = "texts",
    ) -> CrawlResult:
        """Search Internet Archive and download matching PDF manuals.

        Args:
            query: Search query string.
            collection: IA collection to search within.
            max_items: Maximum number of items to download.
            media_type: Media type filter (default: texts).

        Returns:
            CrawlResult with download statistics and file paths.
        """
        result = CrawlResult(source="internet_archive")
        logger.info(
            "ia_crawl.start",
            query=query,
            collection=collection,
            max_items=max_items,
        )

        # Step 1: Search for items
        items = await self._search(query, collection, media_type, max_items)
        result.total_found = len(items)
        logger.info("ia_crawl.search_complete", items_found=len(items))

        # Step 2: Download PDFs for each item
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            for item in items:
                identifier = item.get("identifier", "")
                title = item.get("title", identifier)

                try:
                    pdf_file = await self._download_item_pdf(client, identifier, title)
                    if pdf_file:
                        result.downloaded += 1
                        result.files.append(pdf_file)
                        logger.info(
                            "ia_crawl.downloaded",
                            identifier=identifier,
                            path=str(pdf_file),
                        )
                    else:
                        result.skipped += 1
                        logger.debug("ia_crawl.no_pdf", identifier=identifier)
                except Exception as e:
                    result.failed += 1
                    result.errors.append(f"{identifier}: {e}")
                    logger.warning("ia_crawl.download_failed", identifier=identifier, error=str(e))

                self._rate_limit()

        logger.info(
            "ia_crawl.complete",
            found=result.total_found,
            downloaded=result.downloaded,
            skipped=result.skipped,
            failed=result.failed,
        )
        return result

    async def _search(
        self,
        query: str,
        collection: str,
        media_type: str,
        max_items: int,
    ) -> list[dict]:
        """Search Internet Archive using the advanced search API."""
        params = {
            "q": f"({query}) AND collection:({collection}) AND mediatype:({media_type})",
            "fl[]": ["identifier", "title", "creator", "date", "description"],
            "rows": min(max_items, 500),  # IA caps at 10,000 per page
            "page": 1,
            "output": "json",
        }

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(IA_SEARCH_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        docs = data.get("response", {}).get("docs", [])
        return docs[:max_items]

    async def _download_item_pdf(
        self,
        client: httpx.AsyncClient,
        identifier: str,
        title: str,
    ) -> Path | None:
        """Find and download the first PDF file from an Internet Archive item."""
        # Check if already downloaded
        safe_name = self._safe_filename(identifier)
        existing = list(self.output_dir.glob(f"{safe_name}*.pdf"))
        if existing:
            logger.debug("ia_crawl.already_exists", identifier=identifier)
            return existing[0]

        # Get item metadata to find PDF files
        resp = await client.get(f"{IA_METADATA_URL}/{identifier}")
        resp.raise_for_status()
        metadata = resp.json()

        files = metadata.get("files", [])
        pdf_files = [
            f for f in files
            if f.get("format") in ALLOWED_FORMATS
            and f.get("name", "").lower().endswith(".pdf")
        ]

        if not pdf_files:
            return None

        # Download the first (usually main) PDF
        pdf_info = pdf_files[0]
        pdf_name = pdf_info["name"]
        download_url = f"{IA_DOWNLOAD_URL}/{identifier}/{pdf_name}"

        resp = await client.get(download_url)
        resp.raise_for_status()

        # Save to disk
        out_filename = f"{safe_name}.pdf"
        out_path = self.output_dir / out_filename
        out_path.write_bytes(resp.content)

        # Save metadata sidecar
        meta_path = self.output_dir / f"{safe_name}.meta.json"
        import json
        meta = {
            "identifier": identifier,
            "title": title,
            "source": "internet_archive",
            "source_url": f"https://archive.org/details/{identifier}",
            "original_filename": pdf_name,
            "creator": metadata.get("metadata", {}).get("creator", ""),
            "date": metadata.get("metadata", {}).get("date", ""),
            "description": metadata.get("metadata", {}).get("description", ""),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        return out_path
