"""Monitoring and metrics placeholders."""


class Metrics:
    """Application metrics tracking."""

    @staticmethod
    def track_query(model: str, latency_ms: float, tokens: int) -> None:
        """Track a query execution."""
        # TODO: Prometheus counter/histogram
        pass

    @staticmethod
    def track_ingestion(document_id: str, chunks: int, duration_s: float) -> None:
        """Track document ingestion."""
        pass

    @staticmethod
    def track_crawl(source: str, documents: int, duration_s: float) -> None:
        """Track a crawl job."""
        pass
