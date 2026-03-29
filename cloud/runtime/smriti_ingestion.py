"""
Smriti Ingestion Daemon.

Watches a configured folder and progressively indexes media
through the full VL-JEPA + Setu-2 pipeline.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from cloud.runtime.error_types import SmritiIngestionError
from cloud.runtime.observability import CorrelationContext, get_logger
from cloud.runtime.resilience import BackPressureQueue

log = get_logger("ingestion")

SUPPORTED_EXTENSIONS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".heic",
    ".heif",
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".gif",
}

VIDEO_EXTENSIONS: set[str] = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


@dataclass
class IngestionJob:
    file_path: str
    file_hash: str
    media_type: str
    correlation_id: str
    priority: int = 5


def _compute_file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read(512 * 1024))
    return digest.hexdigest()[:32]


def _classify_media_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    name_lower = Path(path).name.lower()
    if "screenshot" in name_lower or "screen shot" in name_lower:
        return "screenshot"
    return "image"


class SmritiIngestionDaemon:
    def __init__(
        self,
        smriti_db,
        jepa_pool,
        on_progress: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._db = smriti_db
        self._pool = jepa_pool
        self._on_progress = on_progress
        self._queue: BackPressureQueue = BackPressureQueue(maxsize=32, policy="OLDEST_DROP")
        self._worker_task: Optional[asyncio.Task] = None
        self._watcher = None
        self._watched_folders: list[str] = []
        self._watch_handles: dict[str, object] = {}
        self._last_event_at: dict[str, datetime] = {}
        self._watch_errors: dict[str, str] = {}
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stats = {
            "queued": 0,
            "processed": 0,
            "failed": 0,
            "skipped_duplicate": 0,
        }

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._worker_task = asyncio.create_task(self._worker_loop(), name="smriti_ingestion_worker")
        log.info("ingestion_daemon_started")

    async def stop(self) -> None:
        self._running = False
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher.join()
            self._watcher = None
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await asyncio.wait_for(self._worker_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._worker_task = None
        log.info("ingestion_daemon_stopped", stats=self._stats)

    async def watch_folder(self, folder_path: str) -> int:
        folder = Path(folder_path).expanduser().resolve()
        if not folder.is_dir():
            self._watch_errors[str(folder)] = "Not a directory"
            raise SmritiIngestionError(folder_path, "Not a directory")

        if str(folder) not in self._watched_folders:
            self._watched_folders.append(str(folder))
            self._start_watchdog(str(folder))
        self._watch_errors.pop(str(folder), None)

        queued = 0
        for root, _, files in os.walk(str(folder)):
            for filename in files:
                file_path = os.path.join(root, filename)
                if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                queued += int(await self._enqueue_file(file_path, priority=8))

        log.info("folder_watch_started", folder=str(folder), queued=queued)
        return queued

    async def unwatch_folder(self, folder_path: str) -> bool:
        folder = str(Path(folder_path).expanduser().resolve())
        if folder in self._watched_folders:
            self._watched_folders.remove(folder)
        handle = self._watch_handles.pop(folder, None)
        if handle is not None and self._watcher is not None:
            try:
                self._watcher.unschedule(handle)
            except Exception:
                pass
        self._last_event_at.pop(folder, None)
        self._watch_errors.pop(folder, None)
        return True

    def schedule_watch_folder(self, folder_path: str) -> None:
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.watch_folder(folder_path), self._loop)

    def schedule_unwatch_folder(self, folder_path: str) -> None:
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.unwatch_folder(folder_path), self._loop)

    def is_watch_active(self, folder_path: str) -> bool:
        return str(Path(folder_path).expanduser().resolve()) in self._watched_folders

    def get_last_event_at(self, folder_path: str) -> Optional[datetime]:
        return self._last_event_at.get(str(Path(folder_path).expanduser().resolve()))

    def get_watch_error(self, folder_path: str) -> Optional[str]:
        return self._watch_errors.get(str(Path(folder_path).expanduser().resolve()))

    async def ingest_file(self, file_path: str) -> bool:
        return await self._enqueue_file(file_path, priority=1)

    def get_stats(self) -> dict:
        return {
            **self._stats,
            "queue_depth": self._queue.qsize(),
            "queue_utilization": self._queue.utilization(),
            "watched_folders": self._watched_folders,
        }

    async def _enqueue_file(self, file_path: str, priority: int = 5) -> bool:
        try:
            file_hash = _compute_file_hash(file_path)
        except OSError:
            return False

        existing = self._db.get_smriti_media_by_hash(file_hash)
        if existing is not None:
            self._stats["skipped_duplicate"] += 1
            return False

        job = IngestionJob(
            file_path=file_path,
            file_hash=file_hash,
            media_type=_classify_media_type(file_path),
            correlation_id=CorrelationContext.new(),
            priority=priority,
        )

        accepted = await self._queue.put(job)
        if accepted:
            self._stats["queued"] += 1
        return accepted

    async def _worker_loop(self) -> None:
        while self._running:
            try:
                job: IngestionJob = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise

            CorrelationContext.set(job.correlation_id)
            try:
                await self._process_job(job)
                self._stats["processed"] += 1
            except Exception as exc:
                self._stats["failed"] += 1
                log.warning("ingestion_job_failed", file=job.file_path, error=str(exc))
                self._db.mark_smriti_media_failed(job.file_hash, str(exc))

            if self._on_progress is not None:
                self._on_progress(self.get_stats())

    async def _process_job(self, job: IngestionJob) -> None:
        import numpy as np
        from PIL import Image

        from cloud.runtime.jepa_worker import JEPAWorkItem

        media = self._db.create_smriti_media(
            file_path=job.file_path,
            file_hash=job.file_hash,
            media_type=job.media_type,
            ingestion_status="processing",
        )

        if job.media_type == "video":
            frame = await self._extract_video_keyframe(job.file_path)
        else:
            loop = asyncio.get_running_loop()
            frame = await loop.run_in_executor(
                None,
                lambda: np.asarray(Image.open(job.file_path).convert("RGB")),
            )

        if frame is None:
            raise SmritiIngestionError(job.file_path, "Could not load frame")

        work_item = JEPAWorkItem(
            correlation_id=job.correlation_id,
            session_id=f"smriti_ingest_{media.id}",
            frame_array=frame.tobytes(),
            frame_shape=frame.shape,
            frame_dtype=str(frame.dtype),
            priority=job.priority,
            observation_id=media.id,
        )

        result = await self._resolve_pool().submit(work_item)
        if result.error:
            raise SmritiIngestionError(job.file_path, result.error)

        tick_dict = result.jepa_tick_dict
        embedding = tick_dict.get("session_fingerprint", [])[:128]
        self._db.update_smriti_media(
            media_id=media.id,
            depth_strata=tick_dict.get("depth_strata"),
            anchor_matches=tick_dict.get("anchor_matches") or [],
            setu_descriptions=[
                item.get("description", item)
                for item in (tick_dict.get("setu_descriptions") or [])
                if isinstance(item, dict)
            ],
            alignment_loss=tick_dict.get("alignment_loss", 0.0),
            hallucination_risk=self._compute_mean_hallucination_risk(tick_dict),
            ingestion_status="complete",
            embedding=embedding if len(embedding) == 128 else None,
            observation_id=media.observation_id,
        )
        log.info("ingestion_complete", file=job.file_path, media_id=media.id)

    def _compute_mean_hallucination_risk(self, tick_dict: dict) -> float:
        descriptions = tick_dict.get("setu_descriptions", []) or []
        risks = [item.get("description", {}).get("hallucination_risk", 0.5) for item in descriptions]
        return float(sum(risks) / max(len(risks), 1))

    async def _extract_video_keyframe(self, file_path: str):
        import numpy as np
        from PIL import Image

        av = None
        try:
            av = importlib.import_module("av")
        except ImportError:
            log.warning(
                "pyav_not_installed",
                message="Install av>=12.0.0 for video support: pip install av",
            )

        if av is not None:
            try:
                loop = asyncio.get_running_loop()

                def _decode():
                    container = av.open(file_path)
                    try:
                        video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
                        if video_stream is None:
                            return None
                        video_stream.codec_context.skip_frame = "NONKEY"
                        for packet in container.demux(video_stream):
                            for frame in packet.decode():
                                image = frame.to_image().convert("RGB")
                                return np.asarray(image, dtype=np.uint8)
                        return None
                    finally:
                        container.close()

                frame = await loop.run_in_executor(None, _decode)
                if frame is not None:
                    return frame
            except Exception as exc:
                log.warning("pyav_decode_failed", file=file_path, error=str(exc))

        try:
            image = Image.open(file_path).convert("RGB")
            return np.asarray(image)
        except Exception as exc:
            log.warning("pil_video_fallback_failed", file=file_path, error=str(exc))
            return None

    def _resolve_pool(self):
        return self._pool() if callable(self._pool) else self._pool

    def _start_watchdog(self, folder: str) -> None:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except Exception:
            self._watch_errors[folder] = "watchdog not installed"
            log.warning("watchdog_not_installed", message="File watching disabled")
            return

        daemon_ref = self

        class SmritiHandler(FileSystemEventHandler):
            def on_created(self, event):
                if event.is_directory:
                    return
                path = event.src_path
                if Path(path).suffix.lower() not in SUPPORTED_EXTENSIONS:
                    return
                daemon_ref._last_event_at[folder] = datetime.now(timezone.utc)
                if daemon_ref._loop is not None and daemon_ref._loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        daemon_ref._enqueue_file(path, priority=2),
                        daemon_ref._loop,
                    )

        if self._watcher is None:
            self._watcher = Observer()
        handle = self._watcher.schedule(SmritiHandler(), folder, recursive=True)
        self._watch_handles[folder] = handle
        if not self._watcher.is_alive():
            self._watcher.start()
