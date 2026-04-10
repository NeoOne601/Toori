from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import queue as queue_module
import signal
import subprocess
import sys
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from cloud.runtime.error_types import SmritiRateLimitError
from cloud.runtime.observability import PipelineTrace, get_logger, trace_stage


@dataclass(slots=True)
class JEPAWorkItem:
    correlation_id: str
    session_id: str
    frame_array: bytes
    frame_shape: tuple[int, ...]
    frame_dtype: str
    priority: int
    observation_id: str | None
    delay_s: float = 0.0
    precomputed_patch_tokens: bytes | None = None
    precomputed_mask_results_json: str | None = None


@dataclass(slots=True)
class JEPAWorkResult:
    correlation_id: str
    session_id: str
    observation_id: str | None
    jepa_tick_dict: dict[str, Any]
    pipeline_trace: dict[str, Any]
    error: str | None
    state_vector: list[float] | None = None
    worker_id: int | None = None


@dataclass(slots=True)
class JEPAWorkerPreflightResult:
    ok: bool
    stage: str
    device: str = "cpu"
    n_frames: int = 0
    model_id: str = ""
    error: str | None = None
    crash_fingerprint: str | None = None
    exit_code: int | None = None
    signal: int | None = None


@dataclass(slots=True)
class _WorkerHandle:
    worker_id: int
    process: subprocess.Popen[str]
    pending: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Future[JEPAWorkResult]]]
    pending_order: deque[str]
    sessions: set[str]
    submitted: int = 0
    completed: int = 0
    last_error: str | None = None
    last_exit_code: int | None = None
    last_signal: int | None = None
    last_stderr: deque[str] = field(default_factory=lambda: deque(maxlen=32))
    lock: threading.Lock = field(default_factory=threading.Lock)
    stdin_lock: threading.Lock = field(default_factory=threading.Lock)


class JEPAWorkerPool:
    """
    Runs isolated JEPA workers in console python3.11 subprocesses.

    Native crashes are terminal for the current pool. Workers are not
    automatically respawned after a hard failure; the runtime must explicitly
    switch to degraded continuity and, if desired later, create a brand-new pool.
    """

    def __init__(
        self,
        num_workers: int | None = None,
        queue_maxsize: int = 4,
        disable_vjepa2: bool = False,
    ) -> None:
        self._num_workers = self._resolve_worker_count(num_workers)
        self._queue_maxsize = max(int(queue_maxsize), 1)
        self._disable_vjepa2 = bool(disable_vjepa2)
        self._log = get_logger("jepa_worker_pool")
        self._closed = False
        self._pool_lock = threading.Lock()
        self._session_to_worker: dict[str, int] = {}
        self._workers: list[_WorkerHandle] = []
        self._reader_threads: list[threading.Thread] = []
        self._stderr_threads: list[threading.Thread] = []
        for worker_id in range(self._num_workers):
            self._workers.append(self._spawn_worker_handle(worker_id))

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

    def _spawn_worker_handle(self, worker_id: int, *, sessions: set[str] | None = None) -> _WorkerHandle:
        process = _launch_worker_subprocess(worker_id=worker_id, disable_vjepa2=self._disable_vjepa2)
        handle = _WorkerHandle(
            worker_id=worker_id,
            process=process,
            pending={},
            pending_order=deque(),
            sessions=set(sessions or ()),
        )
        reader = threading.Thread(
            target=self._stdout_dispatcher,
            args=(handle,),
            name=f"jepa-worker-reader-{worker_id}",
            daemon=True,
        )
        reader.start()
        self._reader_threads.append(reader)
        stderr_reader = threading.Thread(
            target=self._stderr_drain,
            args=(handle,),
            name=f"jepa-worker-stderr-{worker_id}",
            daemon=True,
        )
        stderr_reader.start()
        self._stderr_threads.append(stderr_reader)
        return handle

    def _resolve_worker(self, worker_id: int) -> _WorkerHandle:
        return self._workers[worker_id]

    def _worker_alive(self, worker: _WorkerHandle) -> bool:
        return worker.process.poll() is None

    def _record_exit_state(self, worker: _WorkerHandle) -> None:
        exit_code = worker.process.poll()
        worker.last_exit_code = exit_code
        worker.last_signal = -exit_code if exit_code is not None and exit_code < 0 else None

    def _close_worker_resources(self, worker: _WorkerHandle, *, terminate_timeout_s: float = 0.5) -> None:
        with worker.stdin_lock:
            try:
                if worker.process.stdin is not None and not worker.process.stdin.closed:
                    worker.process.stdin.close()
            except Exception:
                pass
        if self._worker_alive(worker):
            worker.process.terminate()
            try:
                worker.process.wait(timeout=max(terminate_timeout_s, 0.0))
            except subprocess.TimeoutExpired:
                worker.process.kill()
                worker.process.wait(timeout=1.0)
        self._record_exit_state(worker)
        try:
            if worker.process.stdout is not None:
                worker.process.stdout.close()
        except Exception:
            pass
        try:
            if worker.process.stderr is not None:
                worker.process.stderr.close()
        except Exception:
            pass

    def _cleanup_pending_submission(self, worker: _WorkerHandle, correlation_id: str) -> None:
        with worker.lock:
            worker.pending.pop(correlation_id, None)
            try:
                worker.pending_order.remove(correlation_id)
            except ValueError:
                pass

    def _worker_failure_reason(self, worker: _WorkerHandle, default_reason: str) -> str:
        self._record_exit_state(worker)
        stderr = " ".join(worker.last_stderr).strip()
        if stderr:
            return stderr.splitlines()[-1].strip()
        if worker.last_signal is not None:
            return f"{default_reason} (signal {worker.last_signal})"
        if worker.last_exit_code not in (None, 0):
            return f"{default_reason} (exit {worker.last_exit_code})"
        return default_reason

    def _fail_pending_work(self, worker: _WorkerHandle, reason: str) -> None:
        with worker.lock:
            pending = list(worker.pending.values())
            worker.pending.clear()
            worker.pending_order.clear()
            worker.last_error = reason
        for loop, future in pending:
            def _set_exception(target: asyncio.Future[JEPAWorkResult] = future, error: str = reason) -> None:
                if target.cancelled() or target.done():
                    return
                target.set_exception(RuntimeError(f"JEPA worker unavailable: {error}"))

            loop.call_soon_threadsafe(_set_exception)

    def _ensure_worker_alive(self, worker_id: int, *, reason: str) -> _WorkerHandle:
        worker = self._resolve_worker(worker_id)
        if self._worker_alive(worker):
            return worker
        failure_reason = self._worker_failure_reason(worker, reason)
        self._fail_pending_work(worker, failure_reason)
        raise RuntimeError(failure_reason)

    def recycle_session_worker(self, session_id: str, reason: str) -> None:
        if self._closed:
            return
        worker_id = self._session_to_worker.get(session_id)
        if worker_id is None:
            return
        worker = self._resolve_worker(worker_id)
        worker.last_error = reason
        self._close_worker_resources(worker)
        self._fail_pending_work(worker, reason)

    def _worker_index_for_session(self, session_id: str) -> int:
        if session_id in self._session_to_worker:
            index = self._session_to_worker[session_id]
            worker = self._ensure_worker_alive(index, reason="assigned JEPA worker exited")
            worker.sessions.add(session_id)
            return index
        digest = hashlib.blake2b(session_id.encode("utf-8"), digest_size=4).digest()
        index = int.from_bytes(digest, byteorder="big", signed=False) % self._num_workers
        self._session_to_worker[session_id] = index
        self._ensure_worker_alive(index, reason="assigned JEPA worker unavailable").sessions.add(session_id)
        return index

    async def submit(self, work_item: JEPAWorkItem) -> JEPAWorkResult:
        if self._closed:
            raise RuntimeError("JEPAWorkerPool is shut down")

        worker = self._ensure_worker_alive(
            self._worker_index_for_session(work_item.session_id),
            reason="JEPA worker unavailable before submit",
        )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[JEPAWorkResult] = loop.create_future()

        with worker.lock:
            if len(worker.pending_order) >= self._queue_maxsize:
                raise SmritiRateLimitError(endpoint="jepa_worker_pool", retry_after_s=0.05)
            worker.pending[work_item.correlation_id] = (loop, future)
            worker.pending_order.append(work_item.correlation_id)
            worker.submitted += 1

        try:
            await asyncio.to_thread(self._write_request, worker, work_item)
        except Exception as exc:
            self._cleanup_pending_submission(worker, work_item.correlation_id)
            raise RuntimeError(f"JEPA worker queue unavailable: {exc}") from exc

        try:
            return await future
        except asyncio.CancelledError:
            self._cleanup_pending_submission(worker, work_item.correlation_id)
            future.cancel()
            raise

    async def submit_fire_and_forget(
        self,
        work_item: JEPAWorkItem,
        callback: Callable[[JEPAWorkResult], None],
    ) -> None:
        async def _run() -> None:
            result = await self.submit(work_item)
            callback(result)

        asyncio.create_task(_run())

    def _write_request(self, worker: _WorkerHandle, work_item: JEPAWorkItem) -> None:
        if not self._worker_alive(worker):
            raise RuntimeError(self._worker_failure_reason(worker, "worker process exited without returning a result"))
        payload = {
            "correlation_id": work_item.correlation_id,
            "session_id": work_item.session_id,
            "observation_id": work_item.observation_id,
            "priority": int(work_item.priority),
            "delay_s": float(work_item.delay_s),
            "frame_shape": list(work_item.frame_shape),
            "frame_dtype": work_item.frame_dtype,
            "frame_base64": base64.b64encode(work_item.frame_array).decode("utf-8"),
            "precomputed_patch_tokens_b64": (
                base64.b64encode(work_item.precomputed_patch_tokens).decode("utf-8")
                if work_item.precomputed_patch_tokens is not None
                else None
            ),
            "precomputed_mask_results_json": (
                work_item.precomputed_mask_results_json
            ),
        }
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        with worker.stdin_lock:
            try:
                if worker.process.stdin is None or worker.process.stdin.closed:
                    raise RuntimeError("worker stdin unavailable")
                worker.process.stdin.write(line)
                worker.process.stdin.flush()
            except (BrokenPipeError, OSError, RuntimeError) as exc:
                raise RuntimeError(self._worker_failure_reason(worker, f"worker stdin failed: {exc}")) from exc

    def get_worker_stats(self) -> list[dict[str, Any]]:
        stats: list[dict[str, Any]] = []
        for worker in self._workers:
            self._record_exit_state(worker)
            stats.append(
                {
                    "worker_id": worker.worker_id,
                    "pid": worker.process.pid,
                    "alive": self._worker_alive(worker),
                    "queue_size": len(worker.pending_order),
                    "pending": len(worker.pending_order),
                    "submitted": worker.submitted,
                    "completed": worker.completed,
                    "sessions": sorted(worker.sessions),
                    "last_exit_code": worker.last_exit_code,
                    "last_signal": worker.last_signal,
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
        for worker in self._workers:
            self._fail_pending_work(worker, "worker pool shutting down")
            self._close_worker_resources(worker, terminate_timeout_s=max(deadline - time.time(), 0.0))

    def _stderr_drain(self, worker: _WorkerHandle) -> None:
        stream = worker.process.stderr
        if stream is None:
            return
        try:
            for raw_line in stream:
                line = raw_line.strip()
                if not line:
                    continue
                worker.last_stderr.append(line)
        except Exception:
            return

    def _stdout_dispatcher(self, worker: _WorkerHandle) -> None:
        stream = worker.process.stdout
        if stream is None:
            reason = "worker stdout unavailable"
            worker.last_error = reason
            self._fail_pending_work(worker, reason)
            return
        try:
            for raw_line in stream:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    worker.last_stderr.append(f"invalid worker json: {line[:200]}")
                    continue
                result = _deserialize_result_payload(payload, worker.worker_id)
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
        finally:
            if self._closed:
                return
            reason = self._worker_failure_reason(worker, "worker process exited without returning a result")
            self._log.warning("jepa_worker_unavailable", worker_id=worker.worker_id, reason=reason)
            self._fail_pending_work(worker, reason)


def _crash_fingerprint(stage: str, error: str) -> str:
    payload = f"{stage}:{error}".encode("utf-8", "ignore")
    return hashlib.blake2b(payload, digest_size=6).hexdigest()


def _apply_worker_env_defaults(*, disable_vjepa2: bool = False) -> None:
    os.environ["TOORI_VJEPA2_DEVICE"] = "cpu"
    os.environ["TOORI_DINOV2_DEVICE"] = "cpu"
    if disable_vjepa2:
        os.environ["TOORI_VJEPA2_DISABLE"] = "1"
    else:
        os.environ.pop("TOORI_VJEPA2_DISABLE", None)
    if not os.environ.get("TOORI_VJEPA2_FRAMES", "").strip() and os.environ.get("TOORI_VJEPA2_DEVICE", "").lower() == "cpu":
        os.environ["TOORI_VJEPA2_FRAMES"] = "4"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _worker_env(*, disable_vjepa2: bool = False) -> dict[str, str]:
    env = dict(os.environ)
    repo_root = str(Path.cwd())
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = repo_root if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    env["TOORI_VJEPA2_DEVICE"] = "cpu"
    env["TOORI_DINOV2_DEVICE"] = "cpu"
    if disable_vjepa2:
        env["TOORI_VJEPA2_DISABLE"] = "1"
    else:
        env.pop("TOORI_VJEPA2_DISABLE", None)
    if not env.get("TOORI_VJEPA2_FRAMES", "").strip() and env.get("TOORI_VJEPA2_DEVICE", "").lower() == "cpu":
        env["TOORI_VJEPA2_FRAMES"] = "4"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return env


def _python3_11_executable() -> str:
    candidates = [
        sys.executable,
        "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11",
        "python3.11",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = str(candidate)
        if "/" in path:
            try:
                if Path(path).exists():
                    return path
            except Exception:
                continue
        else:
            return path
    return sys.executable


def _launch_worker_subprocess(*, worker_id: int, disable_vjepa2: bool) -> subprocess.Popen[str]:
    command = [
        _python3_11_executable(),
        str(Path(__file__).resolve()),
        "--serve-worker",
        str(worker_id),
    ]
    if disable_vjepa2:
        command.append("--disable-vjepa2")
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(Path.cwd()),
        env=_worker_env(disable_vjepa2=disable_vjepa2),
    )


def run_jepa_worker_preflight(
    *,
    timeout_s: float = 30.0,
    disable_vjepa2: bool = False,
) -> JEPAWorkerPreflightResult:
    command = [
        _python3_11_executable(),
        "-c",
        (
            "import json, numpy as np; "
            "from cloud.perception.vjepa2_encoder import get_vjepa2_encoder; "
            "encoder = get_vjepa2_encoder(); "
            "encoder.ensure_loaded(); "
            "dummy = np.zeros((256,256,3), dtype=np.uint8); "
            "encoder.encode(dummy); "
            "print(json.dumps({"
            "\"ok\": True, "
            "\"stage\": \"ready\", "
            "\"device\": str(encoder.device), "
            "\"n_frames\": int(getattr(encoder, 'n_frames', 0)), "
            "\"model_id\": str(getattr(encoder, 'model_id', ''))"
            "}))"
        ),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(Path.cwd()),
        env=_worker_env(disable_vjepa2=disable_vjepa2),
    )
    try:
        stdout, stderr = process.communicate(timeout=max(float(timeout_s), 0.1))
        exit_code = process.returncode
        if exit_code == 0:
            payload: dict[str, Any] | None = None
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    decoded = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(decoded, dict):
                    payload = decoded
                    break
            if payload is None:
                error = "preflight returned malformed json"
                return JEPAWorkerPreflightResult(
                    ok=False,
                    stage="failed",
                    error=error,
                    crash_fingerprint=_crash_fingerprint("preflight", error),
                    exit_code=exit_code,
                )
            return JEPAWorkerPreflightResult(
                ok=bool(payload.get("ok", True)),
                stage=str(payload.get("stage", "ready")),
                device=str(payload.get("device", "cpu")),
                n_frames=int(payload.get("n_frames", 0) or 0),
                model_id=str(payload.get("model_id", "")),
                exit_code=exit_code,
            )
        signal_number = -exit_code if exit_code is not None and exit_code < 0 else None
        error_output = " ".join(part.strip() for part in (stderr, stdout) if part and part.strip())
        error = error_output or "worker process exited during preflight"
        return JEPAWorkerPreflightResult(
            ok=False,
            stage="failed",
            error=error,
            crash_fingerprint=_crash_fingerprint("preflight", error),
            exit_code=exit_code,
            signal=signal_number,
        )
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        error = "preflight timed out"
        return JEPAWorkerPreflightResult(
            ok=False,
            stage="failed",
            error=error,
            crash_fingerprint=_crash_fingerprint("preflight", error),
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def _deserialize_frame(work_item: JEPAWorkItem) -> np.ndarray:
    array = np.frombuffer(work_item.frame_array, dtype=np.dtype(work_item.frame_dtype))
    return array.reshape(work_item.frame_shape)


def _serialize_result_payload(result: JEPAWorkResult) -> dict[str, Any]:
    return {
        "correlation_id": result.correlation_id,
        "session_id": result.session_id,
        "observation_id": result.observation_id,
        "jepa_tick_dict": result.jepa_tick_dict,
        "pipeline_trace": result.pipeline_trace,
        "error": result.error,
        "state_vector": result.state_vector,
        "worker_id": result.worker_id,
    }


def _deserialize_result_payload(payload: dict[str, Any], worker_id: int) -> JEPAWorkResult:
    return JEPAWorkResult(
        correlation_id=str(payload.get("correlation_id") or ""),
        session_id=str(payload.get("session_id") or ""),
        observation_id=payload.get("observation_id"),
        jepa_tick_dict=dict(payload.get("jepa_tick_dict") or {}),
        pipeline_trace=dict(payload.get("pipeline_trace") or {}),
        error=str(payload.get("error")) if payload.get("error") is not None else None,
        state_vector=[float(value) for value in (payload.get("state_vector") or [])] if payload.get("state_vector") is not None else None,
        worker_id=int(payload.get("worker_id") or worker_id),
    )


def _decode_work_item(payload: dict[str, Any]) -> JEPAWorkItem:
    precomputed_patch_tokens_b64 = payload.get(
        "precomputed_patch_tokens_b64"
    )
    precomputed_patch_tokens: bytes | None = (
        base64.b64decode(
            precomputed_patch_tokens_b64.encode("utf-8")
        )
        if precomputed_patch_tokens_b64
        else None
    )
    precomputed_mask_results_json: str | None = payload.get(
        "precomputed_mask_results_json"
    )
    return JEPAWorkItem(
        correlation_id=str(payload["correlation_id"]),
        session_id=str(payload["session_id"]),
        frame_array=base64.b64decode(str(payload["frame_base64"]).encode("utf-8")),
        frame_shape=tuple(int(value) for value in payload["frame_shape"]),
        frame_dtype=str(payload["frame_dtype"]),
        priority=int(payload.get("priority", 0)),
        observation_id=payload.get("observation_id"),
        delay_s=float(payload.get("delay_s", 0.0) or 0.0),
        precomputed_patch_tokens=precomputed_patch_tokens,
        precomputed_mask_results_json=precomputed_mask_results_json,
    )


def _serve_worker_subprocess_main(worker_id: int, *, disable_vjepa2: bool = False) -> int:
    from cloud.jepa_service.engine import ImmersiveJEPAEngine

    _apply_worker_env_defaults(disable_vjepa2=disable_vjepa2)
    engines: OrderedDict[str, ImmersiveJEPAEngine] = OrderedDict()
    logger = get_logger("jepa_worker").bind(worker_id=worker_id)

    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    def get_engine(session_id: str) -> ImmersiveJEPAEngine:
        engine = engines.get(session_id)
        if engine is None:
            engine = ImmersiveJEPAEngine(device="cpu")
            engines[session_id] = engine
        engines.move_to_end(session_id)
        while len(engines) > 8:
            engines.popitem(last=False)
        return engine

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            work_item = _decode_work_item(json.loads(line))
        except Exception as exc:
            logger.warning("jepa_worker_decode_failed", error=str(exc))
            continue

        try:
            if work_item.delay_s > 0:
                time.sleep(float(work_item.delay_s))
            trace = PipelineTrace(correlation_id=work_item.correlation_id)
            with trace_stage(trace, "tick"):
                frame = _deserialize_frame(work_item)
                engine = get_engine(work_item.session_id)
                precomputed_patch_tokens = None
                if work_item.precomputed_patch_tokens is not None:
                    precomputed_patch_tokens = np.frombuffer(
                        work_item.precomputed_patch_tokens, dtype=np.float32
                    ).reshape(196, 384).copy()

                precomputed_mask_results = None
                if work_item.precomputed_mask_results_json is not None:
                    try:
                        precomputed_mask_results = json.loads(
                            work_item.precomputed_mask_results_json
                        )
                    except Exception:
                        precomputed_mask_results = None

                tick = engine.tick(
                    frame,
                    session_id=work_item.session_id,
                    observation_id=work_item.observation_id,
                    precomputed_patch_tokens=precomputed_patch_tokens,
                    precomputed_mask_results=precomputed_mask_results,
                )
            payload = tick.to_payload().model_dump(mode="json")
            state_vector = (
                np.asarray(engine._last_state, dtype=np.float32).reshape(-1).tolist()
                if getattr(engine, "_last_state", None) is not None
                else None
            )
            result = JEPAWorkResult(
                correlation_id=work_item.correlation_id,
                session_id=work_item.session_id,
                observation_id=work_item.observation_id,
                jepa_tick_dict=payload,
                pipeline_trace=trace.to_dict(),
                error=None,
                state_vector=state_vector,
                worker_id=worker_id,
            )
        except Exception as exc:  # pragma: no cover - native crashes exit before this path
            trace = PipelineTrace(correlation_id=work_item.correlation_id)
            trace.record_error("tick", str(exc))
            logger.warning("jepa_worker_failed", error=str(exc), session_id=work_item.session_id)
            result = JEPAWorkResult(
                correlation_id=work_item.correlation_id,
                session_id=work_item.session_id,
                observation_id=work_item.observation_id,
                jepa_tick_dict={},
                pipeline_trace=trace.to_dict(),
                error=str(exc),
                state_vector=None,
                worker_id=worker_id,
            )
        sys.stdout.write(json.dumps(_serialize_result_payload(result), separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


def _main(argv: list[str]) -> int:
    if "--serve-worker" not in argv:
        return 0
    index = argv.index("--serve-worker")
    worker_id = int(argv[index + 1]) if index + 1 < len(argv) else 0
    disable_vjepa2 = "--disable-vjepa2" in argv
    return _serve_worker_subprocess_main(worker_id, disable_vjepa2=disable_vjepa2)


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess use
    raise SystemExit(_main(sys.argv[1:]))
