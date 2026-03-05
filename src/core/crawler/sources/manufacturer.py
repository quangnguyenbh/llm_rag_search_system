"""Generic manufacturer website crawler (template for per-site adapters)."""

from pathlib import Path

from src.core.crawler.base import BaseCrawler, CrawlResult


class ManufacturerCrawler(BaseCrawler):
    """Base for crawling manufacturer support/documentation portals.

    Each manufacturer requires a site-specific adapter that implements the
    search and download logic for their documentation portal.

    Subclass this and implement `_get_document_urls()` and `_download_document()`.
    """

    async def crawl(self, **kwargs) -> CrawlResult:
        """Override in site-specific subclass."""
        raise NotImplementedError(
            "ManufacturerCrawler is a base class. "
            "Create a site-specific subclass (e.g., ToyotaCrawler)."
        )
