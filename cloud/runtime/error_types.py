from __future__ import annotations

from typing import Any

from .observability import CorrelationContext


class SmritiError(Exception):
    """Base class for all Smriti errors."""

    def __init__(self, message: str, correlation_id: str | None = None):
        self.correlation_id = correlation_id or CorrelationContext.get()
        super().__init__(message)


class SmritiPipelineError(SmritiError):
    """JEPA pipeline computation failed."""

    def __init__(self, stage: str, message: str, **kwargs: Any) -> None:
        self.stage = stage
        self.details = dict(kwargs)
        super().__init__(f"{stage}: {message}")


class SmritiIngestionError(SmritiError):
    """Media ingestion failed."""

    def __init__(self, file_path: str, message: str, **kwargs: Any) -> None:
        self.file_path = file_path
        self.details = dict(kwargs)
        super().__init__(f"{file_path}: {message}")


class SmritiRecallError(SmritiError):
    """Recall query failed."""

    def __init__(self, query: str, message: str, **kwargs: Any) -> None:
        self.query = query
        self.details = dict(kwargs)
        super().__init__(f"{query}: {message}")


class SmritiSchemaError(SmritiError):
    """Database schema version conflict."""


class SmritiRateLimitError(SmritiError):
    """Request rejected by rate limiter."""

    def __init__(self, endpoint: str, retry_after_s: float) -> None:
        self.endpoint = endpoint
        self.retry_after_s = float(retry_after_s)
        super().__init__(f"Rate limited on {endpoint}; retry after {self.retry_after_s:.3f}s")


class SmritiUncertaintyError(SmritiError):
    """Raised by the ECGD gate when confidence is too low."""

    def __init__(self, region_id: str, consistency_score: float, uncertainty_map: "Any") -> None:
        self.region_id = region_id
        self.consistency_score = float(consistency_score)
        self.uncertainty_map = uncertainty_map
        super().__init__(f"Uncertain region {region_id}")
