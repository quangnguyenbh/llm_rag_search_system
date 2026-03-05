"""Tests for the Internet Archive crawler."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from src.core.crawler.sources.internet_archive import InternetArchiveCrawler


@pytest.fixture
def crawler(tmp_path):
    return InternetArchiveCrawler(output_dir=tmp_path, rate_limit_seconds=0)


@pytest.fixture
def mock_search_response():
    return {
        "response": {
            "docs": [
                {
                    "identifier": "test-manual-001",
                    "title": "Test Electronics Manual",
                    "creator": "Test Corp",
                    "date": "2020-01-01",
                },
            ]
        }
    }


@pytest.fixture
def mock_metadata_response():
    return {
        "metadata": {
            "creator": "Test Corp",
            "date": "2020-01-01",
            "description": "A test manual",
        },
        "files": [
            {
                "name": "test-manual.pdf",
                "format": "PDF",
                "size": "1024",
            }
        ],
    }


class TestInternetArchiveCrawler:
    def test_safe_filename(self, crawler):
        assert crawler._safe_filename("hello world!@#") == "hello_world___"
        assert crawler._safe_filename("normal-file_name.pdf") == "normal-file_name.pdf"

    def test_safe_filename_truncation(self, crawler):
        long_name = "a" * 300
        assert len(crawler._safe_filename(long_name)) == 200

    @pytest.mark.asyncio
    async def test_search_builds_correct_query(self, crawler):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": {"docs": []}}
            mock_resp.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_resp)

            results = await crawler._search("electronics manual", "manuals", "texts", 10)
            assert results == []
