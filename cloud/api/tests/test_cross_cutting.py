from __future__ import annotations

import asyncio
import logging

import numpy as np
from fastapi.testclient import TestClient

from cloud.runtime.app import create_app
from cloud.runtime.error_types import SmritiRateLimitError, SmritiUncertaintyError
from cloud.runtime.observability import (
    MemoryCeilingManager,
    SchemaVersionManager,
    TokenBucketRateLimiter,
    with_fallback,
)
from cloud.runtime.resilience import BackPressureQueue, FallbackChain, SmritiCircuitBreaker


def test_correlation_id_propagated_through_response_header():
    app = create_app()
    client = TestClient(app)

    response = client.get("/healthz", headers={"X-Correlation-ID": "corr-123"})

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "corr-123"


def test_correlation_id_present_in_structured_log_output(caplog):
    app = create_app()
    client = TestClient(app)
    caplog.set_level(logging.INFO)

    response = client.get("/healthz", headers={"X-Correlation-ID": "corr-log-456"})

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "corr-log-456"
    assert "corr-log-456" in caplog.text
    assert "http_request_completed" in caplog.text


def test_rate_limiter_rejects_after_burst_exceeded(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("cloud.runtime.observability.time.monotonic", lambda: clock[0])

    limiter = TokenBucketRateLimiter(rate_per_second=1.0, burst=2)

    assert limiter.allow("query")
    assert limiter.allow("query")
    assert not limiter.allow("query")


def test_rate_limiter_allows_after_token_refill(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("cloud.runtime.observability.time.monotonic", lambda: clock[0])

    limiter = TokenBucketRateLimiter(rate_per_second=1.0, burst=1)

    assert limiter.allow("ingest")
    assert not limiter.allow("ingest")
    clock[0] += 1.1
    assert limiter.allow("ingest")


def test_circuit_breaker_opens_after_threshold(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("cloud.runtime.resilience.time.monotonic", lambda: clock[0])

    breaker = SmritiCircuitBreaker("demo", failure_threshold=2, reset_timeout_s=10.0)

    def fail():
        raise ValueError("boom")

    for _ in range(2):
        try:
            breaker.call(fail)
        except ValueError:
            pass
        clock[0] += 0.1

    assert breaker.state == "open"


def test_circuit_breaker_half_opens_after_reset_timeout(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("cloud.runtime.resilience.time.monotonic", lambda: clock[0])

    breaker = SmritiCircuitBreaker("demo", failure_threshold=1, reset_timeout_s=5.0)

    def fail():
        raise ValueError("boom")

    try:
        breaker.call(fail)
    except ValueError:
        pass

    assert breaker.state == "open"
    clock[0] = 5.1
    assert breaker.state == "half-open"
    assert breaker.call(lambda: "ok") == "ok"
    assert breaker.state == "closed"


def test_fallback_chain_tries_all_handlers_before_raising():
    calls: list[str] = []

    def first():
        calls.append("first")
        raise ValueError("first")

    def second():
        calls.append("second")
        raise RuntimeError("second")

    chain = FallbackChain(("first", first), ("second", second))

    try:
        chain.execute()
    except RuntimeError as exc:
        assert str(exc) == "second"
    else:
        raise AssertionError("FallbackChain should have raised the last exception")

    assert calls == ["first", "second"]


def test_fallback_chain_returns_first_success():
    calls: list[str] = []

    def first():
        calls.append("first")
        raise ValueError("first")

    def second():
        calls.append("second")
        return "ok"

    def third():
        calls.append("third")
        return "should-not-run"

    chain = FallbackChain(("first", first), ("second", second), ("third", third))
    assert chain.execute() == "ok"
    assert calls == ["first", "second"]


def test_with_fallback_decorator_never_propagates_exception():
    calls: list[str] = []

    @with_fallback("fallback-value", log_component="test")
    def explode():
        calls.append("called")
        raise RuntimeError("boom")

    assert explode() == "fallback-value"
    assert calls == ["called"]


def test_smriti_uncertainty_error_returns_200_not_500(tmp_path):
    app = create_app(data_dir=str(tmp_path))

    @app.get("/uncertain")
    def uncertain():
        raise SmritiUncertaintyError("region-1", 0.12, np.zeros((2, 2), dtype=np.float32))

    client = TestClient(app)
    response = client.get("/uncertain")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "uncertain"
    assert body["region_id"] == "region-1"
    assert body["uncertainty_map"] == [[0.0, 0.0], [0.0, 0.0]]


def test_smriti_rate_limit_error_returns_429_with_retry_after(tmp_path):
    app = create_app(data_dir=str(tmp_path))

    @app.get("/limited")
    def limited():
        raise SmritiRateLimitError("/limited", 2.5)

    client = TestClient(app)
    response = client.get("/limited")

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "2"
    body = response.json()
    assert body["error"] == "rate_limited"
    assert body["retry_after_s"] == 2.5


def test_schema_version_manager_applies_migrations_in_order(tmp_path):
    db_path = tmp_path / "schema.sqlite3"
    manager = SchemaVersionManager(str(db_path))
    migrations = [
        (
            1,
            """
            CREATE TABLE IF NOT EXISTS smriti_test_a (
                id INTEGER PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT INTO smriti_test_a (id, value) VALUES (1, 'one');
            """,
        ),
        (
            2,
            """
            INSERT INTO smriti_test_a (id, value) VALUES (2, 'two');
            """,
        ),
    ]

    manager.apply_pending(migrations)
    assert manager.current_version() == 2


def test_schema_version_manager_skips_already_applied(tmp_path):
    db_path = tmp_path / "schema.sqlite3"
    manager = SchemaVersionManager(str(db_path))
    first_pass = [
        (
            1,
            """
            CREATE TABLE IF NOT EXISTS smriti_test_b (
                id INTEGER PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT INTO smriti_test_b (id, value) VALUES (1, 'one');
            """,
        ),
        (
            2,
            """
            INSERT INTO smriti_test_b (id, value) VALUES (2, 'two');
            """,
        ),
    ]
    manager.apply_pending(first_pass)
    manager.apply_pending(first_pass + [(3, "INSERT INTO smriti_test_b (id, value) VALUES (3, 'three');")])

    assert manager.current_version() == 3


def test_back_pressure_queue_drops_when_full_in_latest_drop_mode():
    async def run() -> tuple[bool, bool, int, str]:
        queue = BackPressureQueue(maxsize=1, policy="LATEST_DROP")
        first = await queue.put("first")
        second = await queue.put("second")
        item = await queue.get()
        return first, second, queue.qsize(), item

    first, second, size, item = asyncio.run(run())

    assert first is True
    assert second is False
    assert size == 0
    assert item == "first"


def test_memory_ceiling_manager_reports_current_rss():
    manager = MemoryCeilingManager()

    assert manager.current_rss_mb() > 0.0
