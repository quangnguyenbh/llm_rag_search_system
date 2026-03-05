"""Tests for the HuggingFace datasets crawler."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from src.core.crawler.sources.huggingface_datasets import (
    HuggingFaceCrawler,
    DatasetConfig,
    _DownloadLedger,
)


@pytest.fixture
def crawler(tmp_path):
    return HuggingFaceCrawler(output_dir=tmp_path, rate_limit_seconds=0)


@pytest.fixture
def finepdfs_config():
    return DatasetConfig(
        dataset_name="kaizen9/finepdfs_en",
        url_column="url",
        split="train",
        description="test config",
    )


class TestDownloadLedger:
    def test_empty_ledger(self, tmp_path):
        ledger = _DownloadLedger(tmp_path / "downloaded.jsonl")
        assert ledger.count == 0
        assert not ledger.is_downloaded("https://example.com/a.pdf")

    def test_record_and_check(self, tmp_path):
        path = tmp_path / "downloaded.jsonl"
        ledger = _DownloadLedger(path)
        ledger.record("https://example.com/a.pdf", "/data/a.pdf")
        assert ledger.is_downloaded("https://example.com/a.pdf")
        assert not ledger.is_downloaded("https://example.com/b.pdf")
        assert ledger.count == 1

    def test_reload_from_disk(self, tmp_path):
        path = tmp_path / "downloaded.jsonl"
        ledger1 = _DownloadLedger(path)
        ledger1.record("https://example.com/a.pdf", "/data/a.pdf")
        ledger1.record("https://example.com/b.pdf", "/data/b.pdf")

        # Create a new ledger from the same file — should reload entries
        ledger2 = _DownloadLedger(path)
        assert ledger2.count == 2
        assert ledger2.is_downloaded("https://example.com/a.pdf")
        assert ledger2.is_downloaded("https://example.com/b.pdf")


class TestHuggingFaceCrawler:
    def test_title_from_url_with_pdf(self):
        title = HuggingFaceCrawler._title_from_url("https://example.com/docs/my-manual.pdf")
        assert title == "my-manual"

    def test_title_from_url_without_extension(self):
        title = HuggingFaceCrawler._title_from_url("https://example.com/docs/guide")
        assert title == "guide"

    def test_title_from_url_empty_path(self):
        title = HuggingFaceCrawler._title_from_url("https://example.com/")
        assert title == "untitled"

    def test_safe_filename_sanitization(self, crawler):
        assert crawler._safe_filename("hello world!@#") == "hello_world___"

    @pytest.mark.asyncio
    async def test_crawl_skips_invalid_urls(self, crawler, finepdfs_config):
        """Rows with missing or non-http URLs should be skipped."""
        fake_dataset = [
            {"url": None},
            {"url": ""},
            {"url": "ftp://bad-scheme.com/file.pdf"},
            {"url": 12345},
        ]

        with patch(
            "src.core.crawler.sources.huggingface_datasets.load_dataset"
        ) as mock_load:
            mock_load.return_value = iter(fake_dataset)

            result = await crawler.crawl(config=finepdfs_config, max_items=10)

            assert result.total_found == 0
            assert result.downloaded == 0

    @pytest.mark.asyncio
    async def test_crawl_validates_pdf_content(self, crawler, finepdfs_config):
        """Non-PDF responses should be skipped even if URL looks valid."""
        fake_dataset = [
            {"url": "https://example.com/not-a-pdf.pdf"},
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<html>Not a PDF</html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = MagicMock()

        with (
            patch(
                "src.core.crawler.sources.huggingface_datasets.load_dataset"
            ) as mock_load,
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_load.return_value = iter(fake_dataset)

            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)

            result = await crawler.crawl(config=finepdfs_config, max_items=1)

            assert result.total_found == 1
            assert result.downloaded == 0
            assert result.skipped == 1

    @pytest.mark.asyncio
    async def test_crawl_downloads_valid_pdf(self, crawler, finepdfs_config):
        """Valid PDF responses should be saved to disk."""
        fake_dataset = [
            {"url": "https://example.com/manual.pdf", "title": "Test Manual"},
        ]

        pdf_bytes = b"%PDF-1.4 fake pdf content for testing"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = pdf_bytes
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.raise_for_status = MagicMock()

        with (
            patch(
                "src.core.crawler.sources.huggingface_datasets.load_dataset"
            ) as mock_load,
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_load.return_value = iter(fake_dataset)

            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)

            result = await crawler.crawl(config=finepdfs_config, max_items=1)

            assert result.downloaded == 1
            assert len(result.files) == 1
            assert result.files[0].exists()
            assert result.files[0].read_bytes() == pdf_bytes

    @pytest.mark.asyncio
    async def test_crawl_skips_already_downloaded_urls(self, crawler, finepdfs_config):
        """URLs recorded in the ledger should be skipped on a second run."""
        fake_dataset = [
            {"url": "https://example.com/manual.pdf"},
        ]

        pdf_bytes = b"%PDF-1.4 fake pdf content"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = pdf_bytes
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.raise_for_status = MagicMock()

        with (
            patch(
                "src.core.crawler.sources.huggingface_datasets.load_dataset"
            ) as mock_load,
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)

            # First run — should download
            mock_load.return_value = iter(fake_dataset)
            r1 = await crawler.crawl(config=finepdfs_config, max_items=1)
            assert r1.downloaded == 1

            # Second run — same URL should be skipped via ledger
            mock_load.return_value = iter(fake_dataset)
            r2 = await crawler.crawl(config=finepdfs_config, max_items=1)
            assert r2.downloaded == 0
            assert r2.skipped == 1

    @pytest.mark.asyncio
    async def test_max_items_limits_rows_processed(self, crawler, finepdfs_config):
        """max_items should cap the number of rows iterated, not just downloads."""
        fake_dataset = [
            {"url": f"https://example.com/doc{i}.pdf"} for i in range(100)
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"%PDF-1.4 fake"
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.raise_for_status = MagicMock()

        with (
            patch(
                "src.core.crawler.sources.huggingface_datasets.load_dataset"
            ) as mock_load,
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_load.return_value = iter(fake_dataset)

            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)

            result = await crawler.crawl(config=finepdfs_config, max_items=5)

            # Should process at most 5 rows
            assert result.total_found <= 5
