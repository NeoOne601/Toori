from __future__ import annotations

import contextvars
import functools
import gc
import json
import logging
import os
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Generator

try:  # Optional. Tests must still pass when structlog is unavailable.
    import structlog  # type: ignore
except Exception:  # pragma: no cover - exercised when dependency is absent
    structlog = None
else:  # pragma: no cover - only active when structlog is installed
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


class CorrelationContext:
    """Request-scoped correlation id propagated through the runtime."""

    _current: contextvars.ContextVar[str | None] = contextvars.ContextVar("smriti_correlation_id", default=None)

    @classmethod
    def new(cls) -> str:
        correlation_id = f"smriti_{uuid.uuid4().hex[:12]}"
        cls._current.set(correlation_id)
        return correlation_id

    @classmethod
    def get(cls) -> str:
        value = cls._current.get()
        return value or "no-correlation"

    @classmethod
    def set(cls, correlation_id: str) -> None:
        cls._current.set(correlation_id)


class _JsonStructuredLogger:
    def __init__(self, component: str, context: dict[str, Any] | None = None) -> None:
        self._component = component
        self._context = context or {}
        self._logger = logging.getLogger("smriti")

    def bind(self, **kwargs: Any) -> "_JsonStructuredLogger":
        context = {**self._context, **kwargs}
        return _JsonStructuredLogger(self._component, context)

    def _emit(self, level: int, event: str, **kwargs: Any) -> None:
        payload = {
            "event": event,
            "component": self._component,
            **self._context,
            **kwargs,
        }
        self._logger.log(level, json.dumps(payload, default=_json_default, sort_keys=True))

    def debug(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.ERROR, event, **kwargs)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime,)):
        return value.astimezone(timezone.utc).isoformat()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def get_logger(component: str) -> Any:
    """Return a structured logger bound to the current correlation id."""

    correlation_id = CorrelationContext.get()
    if structlog is not None:
        return structlog.get_logger().bind(component=component, correlation_id=correlation_id)
    return _JsonStructuredLogger(component).bind(correlation_id=correlation_id)


@dataclass
class PipelineTrace:
    correlation_id: str
    stages: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    total_ms: float = 0.0

    def record(self, stage: str, duration_ms: float) -> None:
        self.stages[stage] = round(duration_ms, 3)
        self.total_ms = round(sum(self.stages.values()), 3)

    def record_error(self, stage: str, error: str) -> None:
        self.errors[stage] = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "stages_ms": self.stages,
            "errors": self.errors,
            "total_ms": round(sum(self.stages.values()), 3),
            "bottleneck": max(self.stages, key=self.stages.get) if self.stages else None,
        }


@contextmanager
def trace_stage(trace: PipelineTrace, stage: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        trace.record_error(stage, str(exc))
        raise
    finally:
        trace.record(stage, (time.perf_counter() - start) * 1000.0)


def with_fallback(fallback_value: Any, log_component: str = "unknown"):
    """Return fallback_value when the wrapped callable raises."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                log = get_logger(log_component)
                log.warning(
                    "stage_degraded",
                    function=func.__name__,
                    error=str(exc),
                    fallback=repr(fallback_value),
                )
                return fallback_value

        return wrapper

    return decorator


class TokenBucketRateLimiter:
    """Simple in-process token bucket rate limiter."""

    def __init__(self, rate_per_second: float, burst: int) -> None:
        self.rate_per_second = max(float(rate_per_second), 0.0)
        self.burst = max(int(burst), 1)
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last_refill = self._state.get(key, (self.burst, now))
            elapsed = max(now - last_refill, 0.0)
            if self.rate_per_second > 0:
                tokens = min(self.burst, tokens + (elapsed * self.rate_per_second))
            else:
                tokens = min(self.burst, tokens)
            if tokens < 1.0:
                self._state[key] = (tokens, now)
                return False
            self._state[key] = (tokens - 1.0, now)
            return True


class MemoryCeilingManager:
    """Monitor RSS and trigger GC when approaching a ceiling."""

    def __init__(self, ceiling_mb: int = 2048) -> None:
        self.ceiling_mb = max(int(ceiling_mb), 1)

    def check_and_gc(self) -> bool:
        rss = self.current_rss_mb()
        if rss < (self.ceiling_mb * 0.9):
            return False
        gc.collect()
        return True

    def current_rss_mb(self) -> float:
        proc_statm = Path("/proc/self/statm")
        if proc_statm.exists():
            try:
                pages = int(proc_statm.read_text().split()[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                return float((pages * page_size) / (1024 * 1024))
            except Exception:
                pass
        try:
            import resource
        except Exception:
            return 0.0
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = float(usage.ru_maxrss)
        if sys.platform == "darwin":
            return rss / (1024 * 1024)
        return rss / 1024.0


class SchemaVersionManager:
    """Track and apply SQLite schema migrations in order."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_table(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS smriti_schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def current_version(self) -> int:
        self._ensure_table()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT MAX(version) AS version FROM smriti_schema_version"
            ).fetchone()
        if row is None or row["version"] is None:
            return 0
        return int(row["version"])

    def apply_pending(self, migrations: list[tuple[int, str]]) -> None:
        if not migrations:
            return
        self._ensure_table()
        ordered = sorted(migrations, key=lambda item: item[0])
        current = self.current_version()
        expected = current + 1

        with self._lock, self._connect() as connection:
            existing = {
                int(row["version"])
                for row in connection.execute("SELECT version FROM smriti_schema_version").fetchall()
            }
            for version, sql in ordered:
                if version in existing or version <= current:
                    continue
                if version != expected:
                    from .error_types import SmritiSchemaError

                    raise SmritiSchemaError(
                        f"Schema migration gap detected: expected version {expected}, got {version}"
                    )
                connection.executescript(sql)
                connection.execute(
                    "INSERT INTO smriti_schema_version (version, applied_at) VALUES (?, ?)",
                    (version, datetime.now(timezone.utc).isoformat()),
                )
                existing.add(version)
                expected += 1
            connection.commit()
