"""Unit tests for BatchEmbedder with mocked Bedrock client."""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from src.core.ingestion.embedder import BatchEmbedder


def _make_bedrock_response(embedding: list[float]) -> dict:
    """Build a mock Bedrock invoke_model response."""
    body = MagicMock()
    body.read.return_value = json.dumps({"embedding": embedding}).encode()
    return {"body": body}


class TestBatchEmbedderInit:
    def test_defaults_from_settings(self):
        embedder = BatchEmbedder()
        assert embedder.provider == "bedrock"
        assert embedder.model_id == "amazon.titan-embed-text-v2:0"
        assert embedder.dimensions == 1024
        assert embedder.batch_size == 256

    def test_override_params(self):
        embedder = BatchEmbedder(
            provider="bedrock",
            model_id="custom-model",
            dimensions=512,
            batch_size=32,
            max_retries=5,
        )
        assert embedder.model_id == "custom-model"
        assert embedder.dimensions == 512
        assert embedder.batch_size == 32
        assert embedder.max_retries == 5


class TestEmbedBatch:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        embedder = BatchEmbedder()
        result = await embedder.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_single_text(self):
        embedder = BatchEmbedder(dimensions=4)
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_bedrock_response([0.1, 0.2, 0.3, 0.4])
        embedder._client = mock_client

        result = await embedder.embed_batch(["hello world"])

        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3, 0.4]
        mock_client.invoke_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_texts(self):
        embedder = BatchEmbedder(dimensions=2)
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = [
            _make_bedrock_response([0.1, 0.2]),
            _make_bedrock_response([0.3, 0.4]),
            _make_bedrock_response([0.5, 0.6]),
        ]
        embedder._client = mock_client

        result = await embedder.embed_batch(["a", "b", "c"])

        assert len(result) == 3
        assert result[0] == [0.1, 0.2]
        assert result[2] == [0.5, 0.6]
        assert mock_client.invoke_model.call_count == 3

    @pytest.mark.asyncio
    async def test_request_body_format(self):
        embedder = BatchEmbedder(dimensions=1024)
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_bedrock_response([0.0] * 1024)
        embedder._client = mock_client

        await embedder.embed_batch(["test text"])

        call_kwargs = mock_client.invoke_model.call_args
        body = json.loads(call_kwargs.kwargs.get("body") or call_kwargs[1].get("body"))
        assert body["inputText"] == "test text"
        assert body["dimensions"] == 1024
        assert body["normalize"] is True


class TestBatching:
    @pytest.mark.asyncio
    async def test_splits_into_batches(self):
        embedder = BatchEmbedder(dimensions=2, batch_size=2)
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = [
            _make_bedrock_response([0.1, 0.2]),
            _make_bedrock_response([0.3, 0.4]),
            _make_bedrock_response([0.5, 0.6]),
        ]
        embedder._client = mock_client

        result = await embedder.embed_batch(["a", "b", "c"])

        assert len(result) == 3
        # 3 texts with batch_size=2 → 2 batches, 3 individual calls
        assert mock_client.invoke_model.call_count == 3


class TestRetry:
    @pytest.mark.asyncio
    async def test_retries_on_failure_then_succeeds(self):
        embedder = BatchEmbedder(dimensions=2, max_retries=3)
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = [
            Exception("throttled"),
            _make_bedrock_response([0.1, 0.2]),
        ]
        embedder._client = mock_client

        # Patch sleep to avoid waiting
        with patch("src.core.ingestion.embedder.asyncio.sleep", new_callable=AsyncMock):
            result = await embedder.embed_batch(["test"])

        assert len(result) == 1
        assert result[0] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        embedder = BatchEmbedder(dimensions=2, max_retries=2)
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = Exception("persistent failure")
        embedder._client = mock_client

        with patch("src.core.ingestion.embedder.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="persistent failure"):
                await embedder.embed_batch(["test"])


class TestEmbedQuery:
    @pytest.mark.asyncio
    async def test_embed_query_returns_single_vector(self):
        embedder = BatchEmbedder(dimensions=3)
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_bedrock_response([0.1, 0.2, 0.3])
        embedder._client = mock_client

        result = await embedder.embed_query("what is pilgrimage tourism?")

        assert result == [0.1, 0.2, 0.3]


class TestUnsupportedProvider:
    @pytest.mark.asyncio
    async def test_unsupported_provider_raises(self):
        embedder = BatchEmbedder(provider="unknown")
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            await embedder.embed_batch(["test"])
