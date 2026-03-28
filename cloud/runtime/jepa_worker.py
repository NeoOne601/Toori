from __future__ import annotations

import asyncio
import hashlib
import heapq
import multiprocessing as mp
import queue as queue_module
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from cloud.runtime.error_types import SmritiRateLimitError
from cloud.runtime.observability import PipelineTrace, get_logger, trace_stage


@dataclass(slots=True)
class JEPAWorkItem:
    """Submitted to the worker pool."""

    correlation_id: str
    session_id: str
    frame_array: bytes
    frame_shape: tuple[int, ...]
    frame_dtype: str
    priority: int
    observation_id: str | None
    delay_s: float = 0.0


@dataclass(slots=True)
class JEPAWorkResult:
    """Returned from the worker pool."""

    correlation_id: str
    session_id: str
    observation_id: str | None
    jepa_tick_dict: dict[str, Any]
    pipeline_trace: dict[str, Any]
    error: str | None
    state_vector: list[float] | None = None
    worker_id: int | None = None


@dataclass(slots=True)
class _WorkerHandle:
    worker_id: int
    process: mp.Process
    request_queue: mp.Queue
    result_queue: mp.Queue
    shutdown_event: mp.Event
    pending: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Future[JEPAWorkResult]]]
    pending_order: deque[str]
    sessions: set[str]
    submitted: int = 0
    completed: int = 0
    last_error: str | None = None
    lock: threading.Lock = threading.Lock()


class JEPAWorkerPool:
    """
    Manages a pool of isolated JEPA worker processes.
    """

    def __init__(
        self,
        num_workers: int | None = None,
        queue_maxsize: int = 4,
    ) -> None:
        self._ctx = mp.get_context("spawn")
        self._num_workers = self._resolve_worker_count(num_workers)
        self._queue_maxsize = max(int(queue_maxsize), 1)
        self._log = get_logger("jepa_worker_pool")
        self._closed = False
        self._session_to_worker: dict[str, int] = {}
        self._workers: list[_WorkerHandle] = []
        self._dispatcher_threads: list[threading.Thread] = []
        self._dispatcher_stop = threading.Event()
        for worker_id in range(self._num_workers):
            request_queue = self._ctx.Queue(maxsize=self._queue_maxsize)
            result_queue = self._ctx.Queue(maxsize=self._queue_maxsize * 2)
            shutdown_event = self._ctx.Event()
            process = self._ctx.Process(
                target=_worker_process_main,
                args=(worker_id, request_queue, result_queue, shutdown_event),
                daemon=True,
            )
            handle = _WorkerHandle(
                worker_id=worker_id,
                process=process,
                request_queue=request_queue,
                result_queue=result_queue,
                shutdown_event=shutdown_event,
                pending={},
                pending_order=deque(),
                sessions=set(),
                lock=threading.Lock(),
            )
            self._workers.append(handle)
            process.start()
            dispatcher = threading.Thread(
                target=self._result_dispatcher,
                args=(handle,),
                name=f"jepa-worker-dispatcher-{worker_id}",
                daemon=True,
            )
            dispatcher.start()
            self._dispatcher_threads.append(dispatcher)

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            if not self._closed:
                self._shutdown_sync(1.0)
        except Exception:
            pass

    def _resolve_worker_count(self, num_workers: int | None) -> int:
        if num_workers is not None:
            return max(int(num_workers), 1)
        return 1

    def _worker_index_for_session(self, session_id: str) -> int:
        if session_id in self._session_to_worker:
            return self._session_to_worker[session_id]
        digest = hashlib.blake2b(session_id.encode("utf-8"), digest_size=4).digest()
        index = int.from_bytes(digest, byteorder="big", signed=False) % self._num_workers
        self._session_to_worker[session_id] = index
        self._workers[index].sessions.add(session_id)
        return index

    async def submit(self, work_item: JEPAWorkItem) -> JEPAWorkResult:
        """
        Submit work, await result. Non-blocking to event loop.
        Raises SmritiRateLimitError if queue is full.
        """

        if self._closed:
            raise RuntimeError("JEPAWorkerPool is shut down")

        worker = self._workers[self._worker_index_for_session(work_item.session_id)]
        loop = asyncio.get_running_loop()
        future: asyncio.Future[JEPAWorkResult] = loop.create_future()

        with worker.lock:
            pending_count = len(worker.pending_order)
            if pending_count >= self._queue_maxsize:
                raise SmritiRateLimitError(endpoint="jepa_worker_pool", retry_after_s=0.05)
            worker.pending[work_item.correlation_id] = (loop, future)
            worker.pending_order.append(work_item.correlation_id)
            worker.submitted += 1

        try:
            worker.request_queue.put_nowait(work_item)
        except queue_module.Full as exc:
            with worker.lock:
                worker.pending.pop(work_item.correlation_id, None)
                try:
                    worker.pending_order.remove(work_item.correlation_id)
                except ValueError:
                    pass
            raise SmritiRateLimitError(endpoint="jepa_worker_pool", retry_after_s=0.05) from exc

        return await future

    async def submit_fire_and_forget(
        self,
        work_item: JEPAWorkItem,
        callback: Callable[[JEPAWorkResult], None],
    ) -> None:
        """
        Submit without awaiting. Result delivered via callback.
        """

        async def _run() -> None:
            result = await self.submit(work_item)
            callback(result)

        asyncio.create_task(_run())

    def get_worker_stats(self) -> list[dict[str, Any]]:
        stats: list[dict[str, Any]] = []
        for worker in self._workers:
            try:
                queue_size = worker.request_queue.qsize()
            except (NotImplementedError, OSError):
                queue_size = -1
            stats.append(
                {
                    "worker_id": worker.worker_id,
                    "pid": worker.process.pid,
                    "alive": worker.process.is_alive(),
                    "queue_size": queue_size,
                    "pending": len(worker.pending_order),
                    "submitted": worker.submitted,
                    "completed": worker.completed,
                    "sessions": sorted(worker.sessions),
                }
            )
        return stats

    async def shutdown(self, timeout_s: float = 10.0) -> None:
        await asyncio.to_thread(self._shutdown_sync, timeout_s)

    def _shutdown_sync(self, timeout_s: float) -> None:
        if self._closed:
            return
        self._closed = True
        deadline = time.time() + max(float(timeout_s), 0.0)
        while time.time() < deadline:
            if all(not worker.pending_order for worker in self._workers):
                break
            time.sleep(0.05)
        self._dispatcher_stop.set()
        for worker in self._workers:
            worker.shutdown_event.set()
            try:
                worker.request_queue.put_nowait(None)
            except Exception:
                pass
        for worker in self._workers:
            worker.process.join(timeout=max(deadline - time.time(), 0.0))
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=1.0)

    def _result_dispatcher(self, worker: _WorkerHandle) -> None:
        while not self._dispatcher_stop.is_set() or worker.pending_order:
            try:
                result = worker.result_queue.get(timeout=0.1)
            except queue_module.Empty:
                continue
            if result is None:
                continue
            if not isinstance(result, JEPAWorkResult):
                continue
            with worker.lock:
                pending = worker.pending.pop(result.correlation_id, None)
                if pending is None:
                    worker.completed += 1
                    worker.last_error = result.error
                    continue
                try:
                    worker.pending_order.remove(result.correlation_id)
                except ValueError:
                    pass
                loop, future = pending
                worker.completed += 1
                worker.last_error = result.error
            if future.cancelled():
                continue
            loop.call_soon_threadsafe(future.set_result, result)


def _deserialize_frame(work_item: JEPAWorkItem) -> np.ndarray:
    array = np.frombuffer(work_item.frame_array, dtype=np.dtype(work_item.frame_dtype))
    return array.reshape(work_item.frame_shape)


def _process_work_item(
    worker_id: int,
    request_queue: mp.Queue,
    result_queue: mp.Queue,
    shutdown_event: mp.Event,
) -> None:
    from cloud.jepa_service.engine import ImmersiveJEPAEngine

    engines: OrderedDict[str, ImmersiveJEPAEngine] = OrderedDict()
    pending: list[tuple[int, int, JEPAWorkItem]] = []
    sequence = 0
    logger = get_logger("jepa_worker").bind(worker_id=worker_id)

    def get_engine(session_id: str) -> ImmersiveJEPAEngine:
        engine = engines.get(session_id)
        if engine is None:
            engine = ImmersiveJEPAEngine(device="cpu")
            engines[session_id] = engine
        engines.move_to_end(session_id)
        while len(engines) > 8:
            engines.popitem(last=False)
        return engine

    while True:
        if shutdown_event.is_set() and not pending:
            break
        try:
            work_item = request_queue.get(timeout=0.1)
        except queue_module.Empty:
            if pending:
                pass
            continue
        if work_item is None:
            shutdown_event.set()
            continue
        pending.append((int(work_item.priority), sequence, work_item))
        heapq.heapify(pending)
        sequence += 1

        while pending:
            priority, _, current = heapq.heappop(pending)
            _ = priority
            try:
                if current.delay_s > 0:
                    time.sleep(float(current.delay_s))
                trace = PipelineTrace(correlation_id=current.correlation_id)
                with trace_stage(trace, "tick"):
                    frame = _deserialize_frame(current)
                    engine = get_engine(current.session_id)
                    tick = engine.tick(
                        frame,
                        session_id=current.session_id,
                        observation_id=current.observation_id,
                    )
                payload = tick.to_payload().model_dump(mode="json")
                state_vector = (
                    np.asarray(engine._last_state, dtype=np.float32).reshape(-1).tolist()
                    if getattr(engine, "_last_state", None) is not None
                    else None
                )
                result_queue.put(
                    JEPAWorkResult(
                        correlation_id=current.correlation_id,
                        session_id=current.session_id,
                        observation_id=current.observation_id,
                        jepa_tick_dict=payload,
                        pipeline_trace=trace.to_dict(),
                        error=None,
                        state_vector=state_vector,
                        worker_id=worker_id,
                    )
                )
            except Exception as exc:  # pragma: no cover - exercised via worker failure paths
                trace = PipelineTrace(correlation_id=current.correlation_id)
                trace.record_error("tick", str(exc))
                logger.warning("jepa_worker_failed", error=str(exc), session_id=current.session_id)
                result_queue.put(
                    JEPAWorkResult(
                        correlation_id=current.correlation_id,
                        session_id=current.session_id,
                        observation_id=current.observation_id,
                        jepa_tick_dict={},
                        pipeline_trace=trace.to_dict(),
                        error=str(exc),
                        state_vector=None,
                        worker_id=worker_id,
                    )
                )
            try:
                while True:
                    queued = request_queue.get_nowait()
                    if queued is None:
                        shutdown_event.set()
                        continue
                    pending.append((int(queued.priority), sequence, queued))
                    heapq.heapify(pending)
                    sequence += 1
            except queue_module.Empty:
                pass
            if shutdown_event.is_set() and not pending:
                break


def _worker_process_main(
    worker_id: int,
    request_queue: mp.Queue,
    result_queue: mp.Queue,
    shutdown_event: mp.Event,
) -> None:
    """
    Runs in a subprocess. Owns its own ImmersiveJEPAEngine.
    """

    _process_work_item(worker_id, request_queue, result_queue, shutdown_event)
