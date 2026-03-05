"""Base crawler interface and shared utilities."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import time

import structlog

logger = structlog.get_logger()


@dataclass
class CrawlResult:
    source: str
    total_found: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    files: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class BaseCrawler(ABC):
    """Base class for all document crawlers."""

    def __init__(self, output_dir: Path, rate_limit_seconds: float = 2.0):
        self.output_dir = output_dir
        self.rate_limit_seconds = rate_limit_seconds
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def crawl(self, **kwargs) -> CrawlResult:
        """Execute the crawl and return results."""
        ...

    def _rate_limit(self) -> None:
        """Simple rate limiter between requests."""
        time.sleep(self.rate_limit_seconds)

    def _safe_filename(self, name: str, max_length: int = 200) -> str:
        """Sanitize a string for use as a filename."""
        # Remove/replace unsafe characters
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        return safe[:max_length]
