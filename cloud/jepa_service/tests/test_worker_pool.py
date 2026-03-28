from __future__ import annotations

import asyncio
import contextlib

import numpy as np

from cloud.runtime.error_types import SmritiRateLimitError
from cloud.runtime.jepa_worker import JEPAWorkItem, JEPAWorkerPool
from cloud.runtime.service import ProgressiveComputationScheduler


def _frame() -> np.ndarray:
    rows = np.repeat(np.arange(224, dtype=np.uint8)[:, None], 224, axis=1)
    cols = np.repeat(np.arange(224, dtype=np.uint8)[None, :], 224, axis=0)
    return np.stack(
        [
            (cols * 2) % 255,
            (rows * 3) % 255,
            (rows + cols) % 255,
        ],
        axis=-1,
    )


def _work_item(session_id: str, correlation_id: str, *, priority: int = 3, delay_s: float = 0.0) -> JEPAWorkItem:
    frame = _frame()
    return JEPAWorkItem(
        correlation_id=correlation_id,
        session_id=session_id,
        frame_array=frame.tobytes(),
        frame_shape=frame.shape,
        frame_dtype=str(frame.dtype),
        priority=priority,
        observation_id=f"obs-{correlation_id}",
        delay_s=delay_s,
    )


def test_worker_pool_processes_work_item():
    async def run() -> None:
        pool = JEPAWorkerPool(num_workers=1, queue_maxsize=2)
        try:
            result = await pool.submit(_work_item("session-a", "corr-a"))
            assert result.error is None
            assert result.correlation_id == "corr-a"
            assert result.jepa_tick_dict["energy_map"]
            assert len(result.jepa_tick_dict["energy_map"]) == 14
            assert result.state_vector is not None
            assert len(result.state_vector) == 384
        finally:
            await pool.shutdown()

    asyncio.run(run())


def test_worker_pool_returns_result_with_matching_correlation_id():
    async def run() -> None:
        pool = JEPAWorkerPool(num_workers=1, queue_maxsize=2)
        try:
            result = await pool.submit(_work_item("session-b", "corr-b"))
            assert result.correlation_id == "corr-b"
            assert result.pipeline_trace["correlation_id"] == "corr-b"
        finally:
            await pool.shutdown()

    asyncio.run(run())


def test_full_queue_raises_rate_limit_error_not_blocks():
    async def run() -> None:
        pool = JEPAWorkerPool(num_workers=1, queue_maxsize=1)
        try:
            first = asyncio.create_task(pool.submit(_work_item("session-c", "slow", delay_s=0.25)))
            await asyncio.sleep(0.05)
            try:
                await pool.submit(_work_item("session-c", "fast"))
            except SmritiRateLimitError:
                pass
            else:  # pragma: no cover - defensive
                raise AssertionError("Expected SmritiRateLimitError")
            await first
        finally:
            await pool.shutdown()

    asyncio.run(run())


def test_session_affinity_reuses_same_engine_for_same_session():
    async def run() -> None:
        pool = JEPAWorkerPool(num_workers=2, queue_maxsize=2)
        try:
            first = await pool.submit(_work_item("shared-session", "corr-1"))
            second = await pool.submit(_work_item("shared-session", "corr-2"))
            assert first.worker_id == second.worker_id
            stats = pool.get_worker_stats()
            worker_stats = stats[first.worker_id or 0]
            assert worker_stats["sessions"] == ["shared-session"]
            assert worker_stats["submitted"] >= 2
        finally:
            await pool.shutdown()

    asyncio.run(run())


def test_worker_pool_shuts_down_cleanly():
    async def run() -> None:
        pool = JEPAWorkerPool(num_workers=1, queue_maxsize=2)
        await pool.submit(_work_item("shutdown-session", "corr-shutdown"))
        await pool.shutdown()
        assert all(not item["alive"] for item in pool.get_worker_stats())

    asyncio.run(run())


def test_progressive_scheduler_skips_sag_on_stable_scene():
    scheduler = ProgressiveComputationScheduler()
    assert scheduler.should_run_sag(frame_count=3, energy_variance=0.0, anchor_cache_age_frames=1) is False
    assert scheduler.should_run_cwma(frame_count=10, anchor_matches_changed=False) is False


def test_progressive_scheduler_forces_full_on_user_query():
    scheduler = ProgressiveComputationScheduler()
    scheduler.force_full_pipeline()
    assert scheduler.should_run_sag(frame_count=1, energy_variance=0.0, anchor_cache_age_frames=0) is True
    assert scheduler.should_run_cwma(frame_count=1, anchor_matches_changed=False) is True


def test_jepa_never_runs_on_event_loop_thread():
    async def run() -> None:
        pool = JEPAWorkerPool(num_workers=1, queue_maxsize=2)
        heartbeat = 0

        async def ticker() -> None:
            nonlocal heartbeat
            while True:
                heartbeat += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(ticker())
        try:
            result = await pool.submit(_work_item("blocking-session", "corr-block", delay_s=0.25))
            assert result.error is None
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await pool.shutdown()

        assert heartbeat > 5

    asyncio.run(run())
