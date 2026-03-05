"""Application-specific exceptions."""


class ManualAIError(Exception):
    """Base exception for the application."""
    pass


class DocumentNotFoundError(ManualAIError):
    pass


class IngestionError(ManualAIError):
    pass


class CrawlError(ManualAIError):
    pass


class QuotaExceededError(ManualAIError):
    pass


class RetrievalError(ManualAIError):
    pass
