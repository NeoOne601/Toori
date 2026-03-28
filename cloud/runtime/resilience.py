from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any, Callable

from .observability import get_logger


class SmritiCircuitBreaker:
    """
    Named circuit breaker with configurable threshold and reset behavior.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        reset_timeout_s: float = 30.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self.name = name
        self.failure_threshold = max(int(failure_threshold), 1)
        self.reset_timeout_s = max(float(reset_timeout_s), 0.0)
        self.half_open_max_calls = max(int(half_open_max_calls), 1)
        self._state = "closed"
        self._failure_count = 0
        self._opened_at: float | None = None
        self._last_failure_at: float | None = None
        self._half_open_calls = 0
        self._lock = threading.Lock()
        self._log = get_logger("resilience").bind(circuit_breaker=name)

    def _refresh_state(self) -> None:
        if self._state != "open" or self._opened_at is None:
            return
        if (time.monotonic() - self._opened_at) >= self.reset_timeout_s:
            self._state = "half-open"
            self._half_open_calls = 0

    @property
    def state(self) -> str:
        with self._lock:
            self._refresh_state()
            return self._state

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            self._refresh_state()
            if self._state == "open":
                raise RuntimeError(f"Circuit breaker {self.name} is open")
            if self._state == "half-open":
                if self._half_open_calls >= self.half_open_max_calls:
                    raise RuntimeError(f"Circuit breaker {self.name} is half-open and at capacity")
                self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            with self._lock:
                now = time.monotonic()
                self._last_failure_at = now
                self._failure_count += 1
                if self._state == "half-open" or self._failure_count >= self.failure_threshold:
                    self._state = "open"
                    self._opened_at = now
                    self._half_open_calls = 0
                else:
                    self._state = "closed"
            self._log.warning("circuit_breaker_call_failed", error=str(exc))
            raise
        else:
            with self._lock:
                if self._state == "half-open":
                    self._state = "closed"
                    self._failure_count = 0
                    self._opened_at = None
                    self._half_open_calls = 0
                else:
                    self._failure_count = 0
            self._log.info("circuit_breaker_call_succeeded")
            return result

    def to_health_dict(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_state()
            return {
                "name": self.name,
                "state": self._state,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "reset_timeout_s": self.reset_timeout_s,
                "half_open_max_calls": self.half_open_max_calls,
                "opened_at": self._opened_at,
                "last_failure_at": self._last_failure_at,
                "half_open_calls": self._half_open_calls,
            }


class FallbackChain:
    """Tries handlers in sequence until one succeeds."""

    def __init__(self, *handlers: tuple[str, Callable]) -> None:
        self.handlers = list(handlers)
        self._log = get_logger("resilience")

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for name, handler in self.handlers:
            try:
                return handler(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                self._log.warning("fallback_handler_failed", handler=name, error=str(exc))
        if last_exc is None:
            raise RuntimeError("FallbackChain requires at least one handler")
        raise last_exc


class BackPressureQueue:
    """
    Bounded async queue with back-pressure signaling.
    """

    def __init__(
        self,
        maxsize: int,
        policy: str = "LATEST_DROP",
    ) -> None:
        self.maxsize = max(int(maxsize), 1)
        self.policy = policy.upper()
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=self.maxsize)

    async def put(self, item: Any) -> bool:
        if self.policy == "WAIT":
            await self._queue.put(item)
            return True
        if not self._queue.full():
            self._queue.put_nowait(item)
            return True
        if self.policy == "OLDEST_DROP":
            _ = self._queue.get_nowait()
            self._queue.put_nowait(item)
            return True
        return False

    async def get(self) -> Any:
        return await self._queue.get()

    def qsize(self) -> int:
        return self._queue.qsize()

    def utilization(self) -> float:
        return min(self.qsize() / float(self.maxsize), 1.0)
