from __future__ import annotations

import asyncio
import base64
import json
import io
import hashlib
import os
from collections import defaultdict, deque
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw

from .atlas import EpistemicAtlas
from .config import resolve_data_dir, resolve_smriti_storage
from .error_types import SmritiError, SmritiPipelineError
from .events import EventBus
from .jepa_worker import JEPAWorkItem, JEPAWorkerPool
from .models import (
    ActionToken,
    AnalyzeRequest,
    AnalyzeResponse,
    Answer,
    ChallengeEvaluateRequest,
    ChallengeRun,
    EntityTrack,
    GroundedAffordance,
    GroundedEntity,
    JEPAForecastResponse,
    JEPATick,
    JEPATickPayload,
    LivingLensTickRequest,
    LivingLensTickResponse,
    Observation,
    ObservationsResponse,
    PlanningRolloutRequest,
    PlanningRolloutResponse,
    ProofReportResponse,
    RecoveryBenchmarkRun,
    RecoveryBenchmarkRunRequest,
    RecoveryScenario,
    WorldModelConfig,
    WorldModelConfigUpdate,
    WorldModelStatus,
    ProviderHealthResponse,
    SmritiMigrationRequest,
    SmritiMigrationResult,
    QueryRequest,
    QueryResponse,
    ReasoningTraceEntry,
    RuntimeSettings,
    SceneState,
    ShareObservationResponse,
    SmritiRecallFeedback,
    SmritiRecallFeedbackResult,
    SmritiRecallItem,
    SmritiPruneRequest,
    SmritiPruneResult,
    SmritiRecallRequest,
    SmritiRecallResponse,
    SmritiStorageConfig,
    SmritiTagPersonRequest,
    SmritiTagPersonResponse,
    StorageUsageReport,
    ToolStateObserveRequest,
    ToolStateObserveResponse,
    TalkerEvent,
    WatchFolderStatus,
    WorldStateResponse,
)
from .observability import CorrelationContext, get_logger
from .providers import ProviderRegistry
from .smriti_storage import SmetiDB
from .storage import ObservationStore
from .talker import SelectiveTalker
from .world_model import (
    build_baseline_comparison,
    build_challenge_run,
    build_object_summary,
    build_recovery_benchmark_run,
    build_rollout_comparison,
    build_scene_state,
    default_candidate_actions,
    derive_grounded_entities,
)



class ProgressiveComputationScheduler:
    """
    Manages the tiered computation schedule.
    """

    def __init__(self) -> None:
        self._force_full_pipeline = False

    def should_run_sag(
        self,
        frame_count: int,
        energy_variance: float,
        anchor_cache_age_frames: int,
    ) -> bool:
        if self._force_full_pipeline:
            return True
        if frame_count <= 0 or frame_count % 3 != 0:
            return False
        return energy_variance > 0.01 or anchor_cache_age_frames >= 3

    def should_run_cwma(
        self,
        frame_count: int,
        anchor_matches_changed: bool,
    ) -> bool:
        if self._force_full_pipeline:
            self._force_full_pipeline = False
            return True
        if frame_count <= 0 or frame_count % 10 != 0:
            return False
        return anchor_matches_changed

    def force_full_pipeline(self) -> None:
        self._force_full_pipeline = True


def _jepa_tick_from_payload(payload: JEPATickPayload) -> JEPATick:
    return JEPATick(
        energy_map=np.asarray(payload.energy_map, dtype=np.float32),
        entity_tracks=[EntityTrack.model_validate(track) for track in payload.entity_tracks],
        talker_event=payload.talker_event,
        sigreg_loss=float(payload.sigreg_loss),
        forecast_errors={int(key): float(value) for key, value in payload.forecast_errors.items()},
        session_fingerprint=np.asarray(payload.session_fingerprint, dtype=np.float32),
        planning_time_ms=float(payload.planning_time_ms),
        caption_score=float(payload.caption_score),
        retrieval_score=float(payload.retrieval_score),
        timestamp_ms=int(payload.timestamp_ms),
        warmup=bool(payload.warmup),
        mask_results=payload.mask_results,
        mean_energy=float(payload.mean_energy),
        energy_std=float(payload.energy_std),
        guard_active=bool(payload.guard_active),
        ema_tau=float(payload.ema_tau),
        depth_strata=payload.depth_strata,
        anchor_matches=payload.anchor_matches,
        setu_descriptions=payload.setu_descriptions,
        alignment_loss=float(payload.alignment_loss),
        l2_embedding=payload.l2_embedding,
        predicted_next_embedding=payload.predicted_next_embedding,
        prediction_error=payload.prediction_error,
        epistemic_uncertainty=payload.epistemic_uncertainty,
        aleatoric_uncertainty=payload.aleatoric_uncertainty,
        surprise_score=payload.surprise_score,
        audio_embedding=payload.audio_embedding,
        audio_energy=payload.audio_energy,
        world_model_version=payload.world_model_version,
        configured_encoder=payload.configured_encoder,
        last_tick_encoder_type=payload.last_tick_encoder_type,
        degraded=payload.degraded,
        degrade_reason=payload.degrade_reason,
        degrade_stage=payload.degrade_stage,
    )


class RuntimeContainer:
    def __init__(self, data_dir: str | Path | None = None) -> None:
        from cloud.jepa_service.engine import ImmersiveJEPAEngine, JEPAEngine

        self.data_dir = resolve_data_dir(data_dir)
        self.store = ObservationStore(self.data_dir)
        settings = self.store.load_settings() or RuntimeSettings()
        self._smriti_storage = resolve_smriti_storage(settings, str(self.data_dir))
        self.smriti_db = self._create_smriti_db(self._smriti_storage)
        self.providers = ProviderRegistry()
        self.events = EventBus()
        self.engine = JEPAEngine()
        self._immersive_engine_factory = ImmersiveJEPAEngine
        self._immersive_engines: dict[str, ImmersiveJEPAEngine] = {}
        self._jepa_ticks_by_session: dict[str, deque[JEPATick]] = defaultdict(lambda: deque(maxlen=240))
        self._jepa_pool: JEPAWorkerPool | None = None
        self._latest_proof_report: Optional[Path] = None
        self.talker = SelectiveTalker()
        self.atlas = EpistemicAtlas()
        self._atlases: dict[str, EpistemicAtlas] = {}
        self._previous_tracks: list = []
        self._progressive_scheduler = ProgressiveComputationScheduler()
        self._jepa_energy_ema: dict[str, float] = defaultdict(float)
        self.smriti_daemon = None
        self._sag_templates_path = self._smriti_storage.templates_path or str(Path(self._smriti_storage.data_dir or self.data_dir) / "sag_templates.json")
        self._setu2_bridge = None

    def _create_smriti_db(self, storage_config: SmritiStorageConfig) -> SmetiDB:
        data_dir = Path(storage_config.data_dir or self.data_dir).expanduser().resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        smriti_db = SmetiDB(data_dir)
        if storage_config.frames_dir:
            smriti_db.frames_dir = Path(storage_config.frames_dir).expanduser().resolve()
            smriti_db.frames_dir.mkdir(parents=True, exist_ok=True)
        if storage_config.thumbs_dir:
            smriti_db.thumbs_dir = Path(storage_config.thumbs_dir).expanduser().resolve()
            smriti_db.thumbs_dir.mkdir(parents=True, exist_ok=True)
        if storage_config.templates_path:
            Path(storage_config.templates_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        return smriti_db

    def _configure_smriti_storage(self, storage_config: SmritiStorageConfig) -> SmritiStorageConfig:
        resolved = storage_config.resolve_paths(str(self.data_dir))
        for dir_path in (resolved.data_dir, resolved.frames_dir, resolved.thumbs_dir):
            if dir_path:
                Path(dir_path).expanduser().resolve().mkdir(parents=True, exist_ok=True)
        if resolved.templates_path:
            Path(resolved.templates_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._smriti_storage = resolved
        self.smriti_db = self._create_smriti_db(resolved)
        self._sag_templates_path = resolved.templates_path or str(Path(resolved.data_dir or self.data_dir) / "sag_templates.json")
        daemon = getattr(self, "smriti_daemon", None)
        if daemon is not None:
            daemon._db = self.smriti_db
        return resolved

    @staticmethod
    def _human_readable_bytes(num_bytes: int) -> str:
        value = float(max(num_bytes, 0))
        for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
            if value < 1024.0:
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return f"{value:.1f} PB"

    def get_smriti_storage_config(self) -> SmritiStorageConfig:
        settings = self.get_settings()
        resolved = resolve_smriti_storage(settings, str(self.data_dir))
        self._smriti_storage = resolved
        return resolved

    async def restore_smriti_watch_folders(self) -> None:
        daemon = getattr(self, "smriti_daemon", None)
        if daemon is None:
            return
        resolved = self.get_smriti_storage_config()
        for folder_path in resolved.watch_folders:
            try:
                await daemon.watch_folder(folder_path)
            except Exception as exc:
                get_logger("runtime").warning("watch_folder_restore_failed", path=folder_path, error=str(exc))

    def update_smriti_storage_config(self, new_config: SmritiStorageConfig) -> SmritiStorageConfig:
        settings = self.get_settings()
        normalized_updates: dict[str, Any] = {}
        for field_name in ("data_dir", "frames_dir", "thumbs_dir", "templates_path"):
            value = getattr(new_config, field_name, None)
            if value is None:
                continue
            path_value = Path(value).expanduser().resolve()
            if not path_value.parent.exists():
                raise SmritiError(f"Parent directory does not exist for {field_name}: {path_value.parent}")
            normalized_updates[field_name] = str(path_value)

        normalized_watch_folders: list[str] = []
        for folder_path in new_config.watch_folders:
            normalized_watch_folders.append(str(Path(folder_path).expanduser().resolve()))
        normalized_updates["watch_folders"] = normalized_watch_folders

        normalized_config = new_config.model_copy(update=normalized_updates)
        updated_settings = settings.model_copy(update={"smriti_storage": normalized_config})
        self.update_settings(updated_settings)
        resolved = resolve_smriti_storage(updated_settings, str(self.data_dir))
        return self._configure_smriti_storage(resolved)

    def get_storage_usage(self) -> StorageUsageReport:
        resolved = self.get_smriti_storage_config()

        def _dir_size_bytes(path: str) -> int:
            target = Path(path)
            if not target.exists():
                return 0
            total = 0
            for item in target.rglob("*"):
                if item.is_file():
                    try:
                        total += item.stat().st_size
                    except OSError:
                        continue
            return total

        def _file_size_bytes(path: Path) -> int:
            try:
                return path.stat().st_size if path.exists() else 0
            except OSError:
                return 0

        stats = self.smriti_db.get_ingestion_stats()
        frames_bytes = _dir_size_bytes(resolved.frames_dir or "")
        thumbs_bytes = _dir_size_bytes(resolved.thumbs_dir or "")
        db_path = Path(getattr(self.smriti_db, "db_path", Path(resolved.data_dir or self.data_dir) / "runtime.sqlite3"))
        faiss_path = Path(getattr(self.smriti_db, "_faiss_index_path", Path(resolved.data_dir or self.data_dir) / "smriti_faiss.index"))
        templates_path = Path(resolved.templates_path) if resolved.templates_path else Path()
        smriti_db_bytes = _file_size_bytes(db_path)
        faiss_index_bytes = _file_size_bytes(faiss_path)
        templates_bytes = _file_size_bytes(templates_path)
        total_bytes = frames_bytes + thumbs_bytes + smriti_db_bytes + faiss_index_bytes + templates_bytes
        budget_bytes = resolved.max_storage_gb * (1024 ** 3)
        budget_pct = (total_bytes / budget_bytes * 100.0) if budget_bytes > 0 else 0.0
        watch_folder_stats = [
            self.get_watch_folder_status(folder_path).model_dump(mode="json")
            for folder_path in resolved.watch_folders
        ]

        return StorageUsageReport(
            smriti_data_dir=resolved.data_dir or str(self.data_dir),
            total_media_count=int(stats.get("total", 0)),
            indexed_count=int(stats.get("complete", 0)),
            pending_count=int(stats.get("pending", 0) + stats.get("processing", 0)),
            failed_count=int(stats.get("failed", 0)),
            frames_bytes=frames_bytes,
            thumbs_bytes=thumbs_bytes,
            smriti_db_bytes=smriti_db_bytes,
            faiss_index_bytes=faiss_index_bytes,
            templates_bytes=templates_bytes,
            total_bytes=total_bytes,
            total_human=self._human_readable_bytes(total_bytes),
            max_storage_gb=resolved.max_storage_gb,
            budget_pct=round(budget_pct, 1),
            budget_warning=budget_pct > 85.0,
            budget_critical=budget_pct > 95.0,
            watch_folder_stats=watch_folder_stats,
        )

    def get_watch_folder_status(self, folder_path: str) -> WatchFolderStatus:
        folder = Path(folder_path).expanduser().resolve()
        exists = folder.exists()
        accessible = False
        media_total = 0
        media_indexed = 0
        supported_extensions = {
            ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif",
            ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".gif",
        }

        if exists and folder.is_dir():
            try:
                media_total = sum(
                    1 for item in folder.rglob("*")
                    if item.is_file() and item.suffix.lower() in supported_extensions
                )
                accessible = True
            except PermissionError:
                accessible = False

        try:
            media_indexed = self.smriti_db.count_media_in_folder(str(folder))
        except Exception:
            media_indexed = 0

        daemon = getattr(self, "smriti_daemon", None)
        watchdog_active = bool(daemon and daemon.is_watch_active(str(folder)))
        last_event_at = daemon.get_last_event_at(str(folder)) if daemon else None
        error = daemon.get_watch_error(str(folder)) if daemon else None

        return WatchFolderStatus(
            path=str(folder),
            exists=exists,
            is_accessible=accessible,
            media_count_total=media_total,
            media_count_indexed=media_indexed,
            media_count_pending=max(0, media_total - media_indexed),
            watchdog_active=watchdog_active,
            last_event_at=last_event_at,
            error=error,
        )

    def add_watch_folder(self, folder_path: str) -> WatchFolderStatus:
        folder = Path(folder_path).expanduser().resolve()
        if not folder.is_dir():
            raise SmritiError(f"Not a directory: {folder_path}")

        settings = self.get_settings()
        current_folders = [str(Path(item).expanduser().resolve()) for item in settings.smriti_storage.watch_folders]
        if str(folder) not in current_folders:
            current_folders.append(str(folder))
            updated_storage = settings.smriti_storage.model_copy(update={"watch_folders": current_folders})
            self.update_settings(settings.model_copy(update={"smriti_storage": updated_storage}))

        daemon = getattr(self, "smriti_daemon", None)
        if daemon is not None:
            daemon.schedule_watch_folder(str(folder))

        return self.get_watch_folder_status(str(folder))

    def remove_watch_folder(self, folder_path: str) -> None:
        folder = Path(folder_path).expanduser().resolve()
        settings = self.get_settings()
        current_folders = [
            str(Path(item).expanduser().resolve())
            for item in settings.smriti_storage.watch_folders
            if Path(item).expanduser().resolve() != folder
        ]
        updated_storage = settings.smriti_storage.model_copy(update={"watch_folders": current_folders})
        self.update_settings(settings.model_copy(update={"smriti_storage": updated_storage}))
        daemon = getattr(self, "smriti_daemon", None)
        if daemon is not None:
            daemon.schedule_unwatch_folder(str(folder))

    def prune_smriti_storage(self, request: SmritiPruneRequest) -> SmritiPruneResult:
        if request.clear_all and request.confirm_clear_all != "CONFIRM_CLEAR_ALL":
            raise SmritiError("clear_all requires confirm_clear_all='CONFIRM_CLEAR_ALL'")

        removed_records = 0
        removed_bytes = 0
        errors: list[str] = []

        try:
            if request.clear_all:
                removed_records, removed_bytes = self.smriti_db.clear_all_smriti_data()
            else:
                if request.older_than_days is not None:
                    records, bytes_removed = self.smriti_db.prune_older_than(request.older_than_days)
                    removed_records += records
                    removed_bytes += bytes_removed
                if request.remove_missing_files:
                    records, bytes_removed = self.smriti_db.prune_missing_files()
                    removed_records += records
                    removed_bytes += bytes_removed
                if request.remove_failed:
                    records, bytes_removed = self.smriti_db.prune_failed()
                    removed_records += records
                    removed_bytes += bytes_removed
        except Exception as exc:
            errors.append(str(exc))
            if not removed_records and not removed_bytes:
                raise

        return SmritiPruneResult(
            removed_media_records=removed_records,
            removed_bytes=removed_bytes,
            removed_bytes_human=self._human_readable_bytes(removed_bytes),
            errors=errors,
        )

    def migrate_smriti_data(self, request: SmritiMigrationRequest) -> SmritiMigrationResult:
        """
        Migrate Smriti data to a new directory.
        Config is updated after a verified migration succeeds.
        """
        from cloud.runtime.smriti_migration import migrate_smriti_data as _migrate

        settings = self.get_settings()
        current_config = settings.smriti_storage

        def _progress(progress) -> None:
            self.events.publish("smriti.migration_progress", progress.model_dump(mode="json"))

        result = _migrate(
            request=request,
            current_config=current_config,
            base_data_dir=str(self.data_dir),
            on_progress=_progress,
        )

        if result.success and not result.dry_run:
            new_storage = current_config.model_copy(
                update={
                    "data_dir": request.target_data_dir,
                    "frames_dir": request.target_frames_dir,
                    "thumbs_dir": request.target_thumbs_dir,
                    "templates_path": request.target_templates_path,
                }
            )
            self.update_smriti_storage_config(new_storage)

        return result

    async def smriti_recall(self, payload: SmritiRecallRequest) -> SmritiRecallResponse:
        started = perf_counter()
        query_embedding = self._embed_smriti_query(payload.query)
        results = await asyncio.to_thread(
            self.smriti_db.hybrid_search,
            query_embedding,
            payload.query,
            max(payload.top_k * 4, payload.top_k),
        )
        results = self._personalize_smriti_results(payload.query, results)
        filtered = [
            result
            for result in results
            if self._smriti_result_matches_filters(
                result,
                person_filter=payload.person_filter,
                location_filter=payload.location_filter,
                time_start=payload.time_start,
                time_end=payload.time_end,
                min_confidence=payload.min_confidence,
            )
        ][: payload.top_k]
        items = [
            SmritiRecallItem(
                media_id=result.media_id,
                file_path=result.file_path,
                thumbnail_path=result.thumbnail_path,
                setu_score=float(result.setu_score),
                hybrid_score=float(result.hybrid_score),
                primary_description=result.primary_description,
                anchor_basis=result.anchor_basis,
                depth_stratum=result.depth_stratum,
                hallucination_risk=max(0.0, min(1.0, 1.0 - float(result.hybrid_score))),
                created_at=result.created_at,
                person_names=result.person_names,
                location_name=result.location_name,
                depth_strata_data=(self.smriti_db.get_smriti_media(result.media_id).depth_strata if self.smriti_db.get_smriti_media(result.media_id) else None),
                anchor_matches=[
                    {
                        "template_name": match.name,
                        "confidence": match.confidence,
                        "patch_indices": match.patch_indices,
                        "depth_stratum": result.depth_stratum,
                    }
                    for match in (self.smriti_db.get_smriti_media(result.media_id).anchor_matches if self.smriti_db.get_smriti_media(result.media_id) else [])
                ],
                setu_descriptions=[
                    {
                        "description": {
                            "text": description.text,
                            "confidence": description.confidence,
                            "anchor_basis": description.anchor_basis,
                            "hallucination_risk": max(0.0, min(1.0, float(self.smriti_db.get_smriti_media(result.media_id).hallucination_risk if self.smriti_db.get_smriti_media(result.media_id) else 0.5))),
                        }
                    }
                    for description in (self.smriti_db.get_smriti_media(result.media_id).setu_descriptions if self.smriti_db.get_smriti_media(result.media_id) else [])
                ],
            )
            for result in filtered
        ]
        return SmritiRecallResponse(
            query=payload.query,
            results=items,
            total_searched=len(results),
            setu_ms=round((perf_counter() - started) * 1000.0, 3),
        )

    def smriti_recall_feedback(
        self,
        feedback: SmritiRecallFeedback,
    ) -> SmritiRecallFeedbackResult:
        """
        Update Setu-2 W-matrix based on user confirmation or rejection.
        """
        from cloud.runtime.setu2 import Setu2Bridge

        query_embedding = self._embed_smriti_query(feedback.query, dim=384)

        media_embedding: np.ndarray | None = None
        try:
            media = self.smriti_db.get_smriti_media(feedback.media_id)
        except Exception:
            media = None

        if media is not None and media.embedding is not None:
            media_embedding = self._normalize_setu2_media_embedding(media.embedding)
        elif media is not None and media.observation_id:
            observation = self.store.get_observation(media.observation_id)
            if observation is not None:
                media_embedding = self._normalize_setu2_media_embedding(observation.embedding)
        else:
            observation = self.store.get_observation(feedback.media_id)
            if observation is not None:
                media_embedding = self._normalize_setu2_media_embedding(observation.embedding)

        if media_embedding is None:
            return SmritiRecallFeedbackResult(
                updated=False,
                w_mean=0.0,
                message="Media embedding not found — feedback ignored",
            )

        if self._setu2_bridge is None:
            self._setu2_bridge = Setu2Bridge()
        bridge = self._setu2_bridge

        if feedback.confirmed:
            bridge.update_metric_w(
                positive_pairs=[(query_embedding, media_embedding)],
                negative_pairs=[],
                learning_rate=0.005,
            )
        else:
            bridge.update_metric_w(
                positive_pairs=[],
                negative_pairs=[(query_embedding, media_embedding)],
                learning_rate=0.005,
            )

        return SmritiRecallFeedbackResult(
            updated=True,
            w_mean=float(bridge._W.mean()),
            message=(
                f"W-matrix updated. Mean metric weight: {bridge._W.mean():.4f}. "
                f"{'Confirmed' if feedback.confirmed else 'Rejected'} result "
                "will influence future recall ordering."
            ),
        )

    async def smriti_tag_person(self, payload: SmritiTagPersonRequest) -> SmritiTagPersonResponse:
        media = self.smriti_db.get_smriti_media(payload.media_id)
        seed_embedding = np.asarray(media.embedding if media and media.embedding else self._embed_smriti_query(payload.person_name), dtype=np.float32)
        person = self.smriti_db.get_person_by_name(payload.person_name)
        if person is None:
            person = self.smriti_db.create_person(payload.person_name, seed_embedding)
        if media is not None and payload.confirmed:
            self.smriti_db.link_person_to_media(person.id, media.id, 1.0)
        propagated_to = self.smriti_db.propagate_person_tag(person.id) if media is not None else 0
        return SmritiTagPersonResponse(person_id=person.id, propagated_to=propagated_to)

    def smriti_person_journal(self, person_name: str) -> dict[str, Any]:
        person = self.smriti_db.get_person_by_name(person_name)
        if person is None:
            return {"person_name": person_name, "entries": [], "count": 0, "atlas": self.atlas.to_dict()}
        with self.smriti_db._connect() as connection:
            rows = connection.execute(
                """
                SELECT m.id, m.file_path, m.ingested_at
                FROM smriti_person_media pm
                JOIN smriti_media m ON m.id = pm.media_id
                WHERE pm.person_id = ?
                ORDER BY m.ingested_at DESC
                """,
                (person.id,),
            ).fetchall()
            co_rows = connection.execute(
                """
                SELECT p.id, p.name, COUNT(*) AS shared_count
                FROM smriti_person_media target_pm
                JOIN smriti_person_media other_pm
                    ON other_pm.media_id = target_pm.media_id
                   AND other_pm.person_id != target_pm.person_id
                JOIN smriti_persons p ON p.id = other_pm.person_id
                WHERE target_pm.person_id = ?
                GROUP BY p.id, p.name
                ORDER BY shared_count DESC, p.name ASC
                """,
                (person.id,),
            ).fetchall()
        entries = [
            {
                "media_id": row["id"],
                "file_path": row["file_path"],
                "ingested_at": row["ingested_at"],
            }
            for row in rows
        ]
        atlas = self.atlas.to_dict()
        if not atlas.get("nodes") and co_rows:
            atlas = {
                "nodes": [
                    {"entity_id": person.id, "label": person.name, "track_length": max(len(entries), 1)},
                    *[
                        {
                            "entity_id": str(row["id"]),
                            "label": str(row["name"]),
                            "track_length": int(row["shared_count"] or 0),
                        }
                        for row in co_rows
                    ],
                ],
                "edges": [
                    {
                        "source_id": person.id,
                        "target_id": str(row["id"]),
                        "co_occurrence_count": int(row["shared_count"] or 0),
                        "spatial_proximity": min(int(row["shared_count"] or 0) / max(len(entries), 1), 1.0),
                        "status": "active",
                    }
                    for row in co_rows
                ],
            }
        return {
            "person_name": person_name,
            "entries": entries,
            "count": len(entries),
            "atlas": atlas,
        }

    def smriti_media_detail(self, media_id: str) -> dict[str, Any] | None:
        media = self.smriti_db.get_smriti_media(media_id)
        if media is None:
            return None
        return {
            "id": media.id,
            "observation_id": media.observation_id,
            "file_path": media.file_path,
            "file_hash": media.file_hash,
            "media_type": media.media_type,
            "depth_strata": media.depth_strata,
            "anchor_matches": [match.to_dict() for match in media.anchor_matches],
            "setu_descriptions": [description.to_dict() for description in media.setu_descriptions],
            "hallucination_risk": media.hallucination_risk,
            "ingestion_status": media.ingestion_status,
            "visual_cluster_id": media.visual_cluster_id,
            "location_id": media.location_id,
            "original_created_at": media.original_created_at.isoformat() if media.original_created_at else None,
            "ingested_at": media.ingested_at.isoformat() if media.ingested_at else None,
            "alignment_loss": media.alignment_loss,
            "error_message": media.error_message,
            "embedding": media.embedding,
        }

    def smriti_media_neighbors(self, media_id: str, top_k: int = 6) -> dict[str, Any]:
        media = self.smriti_db.get_smriti_media(media_id)
        embedding = None
        if media is not None and media.embedding is not None:
            embedding = self._normalize_setu2_media_embedding(media.embedding)
        elif media is not None and media.observation_id:
            observation = self.store.get_observation(media.observation_id)
            if observation is not None:
                embedding = self._normalize_setu2_media_embedding(observation.embedding)
        elif self.store.get_observation(media_id) is not None:
            embedding = self._normalize_setu2_media_embedding(self.store.get_observation(media_id).embedding)
        if embedding is None:
            return {"neighbors": []}

        from cloud.runtime.setu2 import Setu2Bridge

        if self._setu2_bridge is None:
            self._setu2_bridge = Setu2Bridge()
        corpus = self.smriti_db.get_all_embeddings(limit=500)
        matrix = np.asarray(corpus["embeddings"], dtype=np.float32)
        if matrix.shape[0] <= 1:
            return {"neighbors": []}

        energies = self._setu2_bridge.project_query(embedding, matrix)
        ranked = np.argsort(energies)
        neighbors: list[dict[str, Any]] = []
        for index in ranked:
            candidate_id = corpus["media_ids"][int(index)]
            if candidate_id == media_id:
                continue
            neighbors.append(
                {
                    "media_id": candidate_id,
                    "setu_score": float(1.0 / (1.0 + max(float(energies[int(index)]), 0.0))),
                    "thumbnail_path": corpus.get("thumbnails", {}).get(candidate_id, ""),
                }
            )
            if len(neighbors) >= top_k:
                break
        return {"neighbors": neighbors}

    def smriti_clusters(self) -> dict[str, Any]:
        return self.smriti_db.get_mandala_data()

    def smriti_metrics(self) -> dict[str, Any]:
        worker_stats = self._jepa_pool.get_worker_stats() if self._jepa_pool is not None else []
        pending_media = self.smriti_db.get_pending_media(limit=100)
        return {
            "workers": worker_stats,
            "pending_media": len(pending_media),
            "recent_sessions": sorted(self._jepa_ticks_by_session.keys())[-8:],
            "energy_ema": {session_id: float(value) for session_id, value in self._jepa_energy_ema.items()},
        }

    def get_settings(self) -> RuntimeSettings:
        settings = self.store.load_settings()
        if settings is None:
            raise RuntimeError("Runtime settings are unavailable")
        return settings

    def update_settings(self, settings: RuntimeSettings) -> RuntimeSettings:
        previous = self.store.load_settings()
        saved = self.store.save_settings(settings)
        self._write_settings_mirror(saved)
        self._reset_vjepa2_if_config_changed(previous, saved)
        self.providers.reset_circuits()
        self.events.publish(
            "provider.changed",
            {"settings": saved.model_dump(mode="json")},
        )
        return saved

    def _settings_mirror_path(self) -> Path:
        return Path(self.data_dir).expanduser().resolve() / "settings.json"

    def _write_settings_mirror(self, settings: RuntimeSettings) -> None:
        try:
            path = self._settings_mirror_path()
            path.write_text(json.dumps(settings.model_dump(mode="json"), indent=2, sort_keys=True))
        except Exception as exc:
            get_logger("runtime").warning("settings_mirror_write_failed", error=str(exc))

    def _reset_vjepa2_if_config_changed(
        self,
        previous: RuntimeSettings | None,
        current: RuntimeSettings,
    ) -> None:
        if previous is None:
            return
        if (
            previous.vjepa2_model_path == current.vjepa2_model_path
            and previous.vjepa2_n_frames == current.vjepa2_n_frames
        ):
            return
        try:
            from cloud.perception.vjepa2_encoder import reset_encoder_singleton

            reset_encoder_singleton()
        except Exception as exc:
            get_logger("runtime").warning("vjepa2_reset_failed", error=str(exc))

    def get_vjepa2_settings(self) -> WorldModelConfig:
        settings = self.get_settings()
        effective_model = self._effective_vjepa2_model_id(settings)

        return WorldModelConfig(
            model_path=settings.vjepa2_model_path or "",
            n_frames=settings.vjepa2_n_frames or 0,
            effective_model=effective_model,
            cache_dir=str(Path.home() / ".cache" / "huggingface" / "hub"),
            download_url="https://huggingface.co/facebook/vjepa2-vitl-fpc64-256",
        )

    def update_vjepa2_settings(self, model_path: str, n_frames: int) -> WorldModelConfig:
        settings = self.get_settings()
        updated = settings.model_copy(
            update={
                "vjepa2_model_path": model_path,
                "vjepa2_n_frames": n_frames,
            }
        )
        self.update_settings(updated)
        return self.get_vjepa2_settings()

    def _effective_vjepa2_model_id(self, settings: RuntimeSettings) -> str:
        env_override = os.environ.get("TOORI_VJEPA2_MODEL", "").strip()
        if env_override:
            if env_override.startswith(("~", "/")):
                resolved = Path(env_override).expanduser()
                if resolved.exists():
                    return str(resolved.resolve())
            else:
                return env_override

        configured = str(settings.vjepa2_model_path or "").strip()
        if configured:
            if configured.startswith(("~", "/")):
                resolved = Path(configured).expanduser()
                if resolved.exists():
                    return str(resolved.resolve())
            else:
                return configured

        from cloud.perception.vjepa2_encoder import _resolve_model_id

        return _resolve_model_id()

    def get_world_model_status(self) -> WorldModelStatus:
        from cloud.perception.vjepa2_encoder import _is_test_environment, get_vjepa2_encoder

        settings = self.get_settings()
        test_mode = _is_test_environment()
        encoder_type = "surrogate"
        model_id = "mobilenetv2-12.onnx"
        model_loaded = True
        device = "cpu"
        n_frames = 0
        configured_encoder = "surrogate" if test_mode else "vjepa2"
        last_tick_encoder_type = "surrogate" if test_mode else "not_loaded"
        degraded = False
        degrade_reason = None
        degrade_stage = None

        if not test_mode:
            try:
                encoder = get_vjepa2_encoder()
                encoder_type = encoder.encoder_type
                model_id = encoder.model_id
                model_loaded = encoder.is_loaded
                device = str(encoder.device)
                n_frames = encoder.n_frames
            except Exception as exc:
                get_logger("runtime").warning("world_model_status_fallback", error=str(exc))
                degraded = True
                degrade_reason = str(exc)
                degrade_stage = "status"
        else:
            model_id = "mobilenetv2-12.onnx"

        latest_tick: JEPATick | None = None
        latest_ts = -1
        for ticks in self._jepa_ticks_by_session.values():
            if not ticks:
                continue
            candidate = ticks[-1]
            if candidate.timestamp_ms > latest_ts:
                latest_tick = candidate
                latest_ts = candidate.timestamp_ms

        if latest_tick is not None:
            last_tick_encoder_type = latest_tick.last_tick_encoder_type or latest_tick.world_model_version
            degraded = bool(latest_tick.degraded)
            degrade_reason = latest_tick.degrade_reason
            degrade_stage = latest_tick.degrade_stage
            if latest_tick.world_model_version == "vjepa2":
                encoder_type = "vjepa2"
            if latest_tick.world_model_version == "surrogate" and not test_mode:
                encoder_type = "surrogate"

        all_prediction_errors: list[float] = []
        for engine in self._immersive_engines.values():
            windows = getattr(engine, "_surprise_windows", {})
            for window in windows.values():
                all_prediction_errors.extend(float(score) for score in window)

        mean_prediction_error = (
            float(sum(all_prediction_errors) / len(all_prediction_errors))
            if all_prediction_errors
            else None
        )
        mean_surprise_score = (
            float(
                sum((score / (score + 1.0)) for score in all_prediction_errors) / len(all_prediction_errors)
            )
            if all_prediction_errors
            else None
        )
        total_ticks = int(sum(int(getattr(engine, "_tick_count", 0)) for engine in self._immersive_engines.values()))

        return WorldModelStatus(
            encoder_type=encoder_type,
            model_id=model_id,
            model_loaded=model_loaded,
            device=device,
            n_frames=n_frames,
            test_mode=test_mode,
            total_ticks=total_ticks,
            mean_prediction_error=mean_prediction_error,
            mean_surprise_score=mean_surprise_score,
            configured_encoder=configured_encoder,
            last_tick_encoder_type=last_tick_encoder_type,
            degraded=degraded,
            degrade_reason=degrade_reason,
            degrade_stage=degrade_stage,
        )

    def provider_health(self) -> ProviderHealthResponse:
        return ProviderHealthResponse(providers=self.providers.health_snapshot(self.get_settings()))

    def list_observations(self, *, session_id: Optional[str], limit: int = 50) -> ObservationsResponse:
        return ObservationsResponse(observations=self.store.list_observations(session_id=session_id, limit=limit))

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        response, _, _ = self._analyze_with_world_model(request)
        return response

    async def living_lens_tick(self, request: LivingLensTickRequest) -> LivingLensTickResponse:
        response, scene_state, tracks = await asyncio.to_thread(self._analyze_with_world_model, request, True)
        observations = list(reversed(self.store.recent_observations(session_id=request.session_id, limit=8)))
        history = list(reversed(self.store.recent_scene_states(session_id=request.session_id, limit=8)))
        baseline = build_baseline_comparison(observations, history) if request.proof_mode in {"both", "baseline"} else None
        if request.query:
            self._progressive_scheduler.force_full_pipeline()
        frame = await asyncio.to_thread(self._load_frame_array, response.observation.image_path)
        correlation_id = CorrelationContext.get()
        if correlation_id == "no-correlation":
            correlation_id = CorrelationContext.new()
        work_item = JEPAWorkItem(
            correlation_id=correlation_id,
            session_id=request.session_id,
            frame_array=frame.tobytes(),
            frame_shape=frame.shape,
            frame_dtype=str(frame.dtype),
            priority=0 if request.query else 3,
            observation_id=response.observation.id,
        )
        jepa_pool = self._ensure_jepa_pool()
        result = await jepa_pool.submit(work_item)
        if result.error is not None:
            raise SmritiPipelineError(stage="jepa_worker", message=result.error)
        payload = JEPATickPayload.model_validate(result.jepa_tick_dict)
        jepa_tick = _jepa_tick_from_payload(payload)
        self._jepa_ticks_by_session[request.session_id].append(jepa_tick)
        self._prime_forecast_engine(request.session_id, result.state_vector, payload.forecast_errors)
        previous_ema = self._jepa_energy_ema[request.session_id]
        self._jepa_energy_ema[request.session_id] = (0.05 * float(payload.mean_energy)) + (0.95 * previous_ema)

        immersive_tracks = jepa_tick.entity_tracks[:12]
        self.store.save_entity_tracks(immersive_tracks)
        talker_event = self._talker_event_from_jepa(jepa_tick, immersive_tracks)
        self._previous_tracks = [track.model_copy() for track in immersive_tracks]

        self.atlas.update(
            entity_tracks=immersive_tracks,
            scene_state=scene_state,
            energy_map=jepa_tick.energy_map,
        )
        self._atlas_for_session(request.session_id).update(
            entity_tracks=immersive_tracks,
            scene_state=scene_state,
            energy_map=jepa_tick.energy_map,
        )

        scene_state.metrics.surprise_score = round(float(min(jepa_tick.mean_energy, 1.0)), 4)
        scene_state.metrics.prediction_consistency = round(float(max(1.0 - jepa_tick.mean_energy, 0.0)), 4)

        threshold = float(self._jepa_energy_ema[request.session_id] + (2.0 * jepa_tick.energy_std))
        should_talk = bool(jepa_tick.talker_event)
        self.events.publish(
            "jepa.energy_map",
            {
                "grid": [14, 14],
                "values": np.asarray(jepa_tick.energy_map, dtype=np.float32).ravel().tolist(),
                "mean_energy": jepa_tick.mean_energy,
                "threshold": threshold,
                "should_talk": should_talk,
                "sigreg_loss": jepa_tick.sigreg_loss,
            },
        )
        self.events.publish(
            "jepa_tick",
            {"payload": jepa_tick.to_payload().model_dump(mode="json")},
        )
        if payload.degraded:
            self.events.publish(
                "world_model.degraded",
                {
                    "session_id": request.session_id,
                    "configured_encoder": payload.configured_encoder,
                    "last_tick_encoder_type": payload.last_tick_encoder_type,
                    "degrade_reason": payload.degrade_reason,
                    "degrade_stage": payload.degrade_stage,
                },
            )
        if talker_event is not None:
            self.events.publish("jepa.talker_event", talker_event.model_dump(mode="json"))

        return LivingLensTickResponse(
            **response.model_dump(),
            scene_state=scene_state,
            entity_tracks=immersive_tracks,
            baseline_comparison=baseline,
            talker_event=talker_event,
            jepa_tick=jepa_tick.to_payload(),
        )

    async def shutdown(self) -> None:
        if self._jepa_pool is not None:
            await self._jepa_pool.shutdown()
            self._jepa_pool = None
        self._atlases.clear()

    def get_world_state(self, session_id: str) -> WorldStateResponse:
        history = self.store.recent_scene_states(session_id=session_id, limit=24)
        tracks = self.store.list_entity_tracks(session_id=session_id, limit=32)
        challenges = self.store.recent_challenge_runs(session_id=session_id, limit=8)
        benchmarks = self.store.recent_recovery_benchmark_runs(session_id=session_id, limit=8)
        return WorldStateResponse(
            session_id=session_id,
            current=history[0] if history else None,
            history=history,
            entity_tracks=tracks,
            challenges=challenges,
            benchmarks=benchmarks,
            atlas=self._atlas_for_session(session_id).to_dict(),
        )

    def observe_tool_state(self, request: ToolStateObserveRequest) -> ToolStateObserveResponse:
        settings = self.get_settings()
        self.store.prune(settings.retention_days)

        image_bytes, image = self._load_tool_state_image(request)
        summary_seed = self._tool_state_summary(request)
        if request.screenshot_base64 or request.file_path:
            embedding, provider_name, confidence, provider_metadata = self.providers.perceive(settings, image)
            if provider_name != "basic":
                _, _, basic_metadata = self.providers.basic.perceive(image)
                provider_metadata = {**basic_metadata, **provider_metadata}
        else:
            embedding = self._embed_smriti_query(summary_seed, dim=128).tolist()
            provider_name = "local"
            confidence = 0.78
            provider_metadata = {
                "top_label": request.view_id or request.current_url or "tool state",
                "source": "tool_state",
            }

        previous = self.store.recent_observations(session_id=request.session_id, limit=1)
        novelty = 1.0
        if previous:
            novelty = max(0.0, 1.0 - self._cosine_similarity(embedding, previous[0].embedding))

        grounded_entities = [entity.model_dump(mode="json") for entity in request.visible_entities]
        affordances = [affordance.model_dump(mode="json") for affordance in request.affordances]
        tags = [
            *[entity.label for entity in request.visible_entities if entity.label],
            *[affordance.label for affordance in request.affordances if affordance.label],
            *request.error_banners,
        ]
        metadata = {
            "perception": provider_metadata,
            "observation_kind": "tool_state",
            "state_domain": request.state_domain,
            "current_url": request.current_url,
            "view_id": request.view_id,
            "grounded_entities": grounded_entities,
            "affordances": affordances,
            "focused_target": request.focused_target,
            "error_banners": request.error_banners,
            "triggering_action": request.triggering_action.model_dump(mode="json") if request.triggering_action else None,
            "summary_candidates": [entity["label"] for entity in grounded_entities if entity.get("label")][:6],
            "primary_object_label": request.visible_entities[0].label if request.visible_entities else (request.view_id or None),
        }

        observation = self.store.create_observation(
            image=image,
            raw_bytes=image_bytes,
            embedding=list(embedding),
            session_id=request.session_id,
            confidence=confidence,
            novelty=novelty,
            source_query=request.current_url or request.view_id or summary_seed,
            tags=[tag for tag in tags if tag],
            providers=[provider_name],
            metadata=metadata,
            observation_kind="tool_state",
        )
        observation = self.store.update_observation(
            observation.id,
            summary=summary_seed,
            metadata={
                **observation.metadata,
                "summary_source": "tool_state",
                "summary_candidates": metadata["summary_candidates"],
                "primary_object_label": metadata["primary_object_label"],
            },
        )
        hits = self.store.search_by_vector(
            observation.embedding,
            top_k=request.top_k or settings.top_k,
            session_id=request.session_id,
            exclude_id=observation.id,
        )
        previous_state = self.store.latest_scene_state(request.session_id)
        recent_observations = [
            item
            for item in self.store.recent_observations(session_id=request.session_id, limit=6)
            if item.id != observation.id
        ]
        existing_tracks = self.store.list_entity_tracks(session_id=request.session_id, limit=32)
        scene_state, entity_tracks = build_scene_state(
            observation=observation,
            hits=hits,
            previous_state=previous_state,
            recent_observations=recent_observations,
            existing_tracks=existing_tracks,
        )
        self.store.save_scene_state(scene_state)
        self.store.save_entity_tracks(entity_tracks)
        self._atlas_for_session(request.session_id).update(
            entity_tracks=entity_tracks[:12],
            scene_state=scene_state,
            energy_map=np.zeros((14, 14), dtype=np.float32),
        )
        observation = self.store.update_observation(
            observation.id,
            world_state_id=scene_state.id,
            metadata={
                **observation.metadata,
                "world_model": scene_state.metrics.model_dump(mode="json"),
                "scene_state_id": scene_state.id,
            },
        )

        self.events.publish("observation.created", {"observation": observation.model_dump(mode="json")})
        self.events.publish("tool_state.observed", {"scene_state": scene_state.model_dump(mode="json")})
        self.events.publish("world_state.updated", {"scene_state": scene_state.model_dump(mode="json")})
        return ToolStateObserveResponse(
            observation=observation,
            hits=hits,
            provider_health=self.providers.health_snapshot(settings),
            reasoning_trace=[],
            scene_state=scene_state,
            entity_tracks=entity_tracks,
        )

    def plan_rollout(self, request: PlanningRolloutRequest) -> PlanningRolloutResponse:
        scene_state = (
            self.store.get_scene_state(request.current_state_id)
            if request.current_state_id
            else self.store.latest_scene_state(request.session_id)
        )
        if scene_state is None:
            raise SmritiError("No world state available for rollout planning")
        observation = self.store.get_observation(scene_state.observation_id)
        if request.candidate_actions:
            candidate_actions = request.candidate_actions
        elif scene_state.prediction_window.candidate_actions:
            candidate_actions = scene_state.prediction_window.candidate_actions
        elif observation is not None:
            state_domain, grounded_entities, affordances = derive_grounded_entities(
                observation,
                self.store.list_entity_tracks(session_id=request.session_id, limit=16),
                scene_state.proposal_boxes,
            )
            candidate_actions = default_candidate_actions(
                observation=observation,
                state_domain=request.state_domain or state_domain,
                grounded_entities=grounded_entities,
                affordances=affordances,
            )
        else:
            candidate_actions = [
                ActionToken(
                    id=f"fallback:{scene_state.id}",
                    verb="query_memory",
                    target_kind="memory",
                    target_id=scene_state.id,
                    target_label="continuity memory",
                    parameters={"state_domain": "memory"},
                )
            ]

        comparison = build_rollout_comparison(
            scene_state=scene_state.model_copy(update={"state_domain": request.state_domain or scene_state.state_domain}),
            candidate_actions=candidate_actions,
            horizon=request.horizon,
        )
        updated_scene_state = scene_state.model_copy(
            update={
                "conditioned_rollouts": comparison,
                "prediction_window": scene_state.prediction_window.model_copy(
                    update={
                        "candidate_actions": candidate_actions,
                        "predicted_branches": comparison.ranked_branches,
                        "chosen_branch_id": comparison.chosen_branch_id,
                    }
                ),
            }
        )
        self.store.save_scene_state(updated_scene_state)
        self.events.publish("planning.rollout", {"comparison": comparison.model_dump(mode="json")})
        return PlanningRolloutResponse(scene_state=updated_scene_state, comparison=comparison)

    def run_recovery_benchmark(self, request: RecoveryBenchmarkRunRequest) -> RecoveryBenchmarkRun:
        scene_states = self.store.recent_scene_states(session_id=request.session_id, limit=12)
        if not scene_states:
            raise SmritiError("No world states available for recovery benchmarking")
        comparison = scene_states[0].conditioned_rollouts
        if comparison is None:
            comparison = self.plan_rollout(
                PlanningRolloutRequest(session_id=request.session_id, current_state_id=scene_states[0].id, horizon=2)
            ).comparison
            scene_states = self.store.recent_scene_states(session_id=request.session_id, limit=12)
        benchmark = build_recovery_benchmark_run(
            session_id=request.session_id,
            scene_states=scene_states,
            comparison=comparison,
        )
        self.store.save_recovery_benchmark_run(benchmark)
        self.events.publish("recovery.benchmark", {"benchmark": benchmark.model_dump(mode="json")})
        return benchmark

    def get_recovery_benchmark(self, benchmark_id: str) -> Optional[RecoveryBenchmarkRun]:
        return self.store.get_recovery_benchmark_run(benchmark_id)

    def forecast_jepa(self, session_id: str, k: int) -> JEPAForecastResponse:
        engine = self._immersive_engines.get(session_id)
        if engine is None:
            return JEPAForecastResponse(session_id=session_id, k=k, prediction=[], forecast_error=None, ready=False)
        prediction, forecast_error = engine.forecast_last_state(k)
        return JEPAForecastResponse(
            session_id=session_id,
            k=k,
            prediction=np.asarray(prediction, dtype=np.float32).tolist(),
            forecast_error=forecast_error,
            ready=bool(prediction.size),
        )

    def generate_proof_report(self, session_id: str, chart_b64: str | None = None) -> ProofReportResponse:
        from .proof_report import generate_proof_report

        ticks = list(self._jepa_ticks_by_session.get(session_id, deque()))
        path = generate_proof_report(ticks=ticks, session_id=session_id, chart_b64=chart_b64)
        self._latest_proof_report = Path(path)
        return ProofReportResponse(session_id=session_id, path=str(path), generated=True)

    def build_observation_share(self, session_id: str, observation_id: str | None = None) -> ShareObservationResponse:
        settings = self.get_settings()
        public_url = settings.public_url
        observation = self.store.get_observation(observation_id) if observation_id else None
        if observation is None:
            recent = self.store.recent_observations(session_id=session_id, limit=1)
            observation = recent[0] if recent else None
        if observation is None or observation.session_id != session_id:
            raise KeyError(observation_id or session_id)

        scene_state: SceneState | None = None
        if observation.world_state_id:
            scene_state = self.store.get_scene_state(observation.world_state_id)
        else:
            latest_state = self.store.latest_scene_state(session_id)
            if latest_state is not None and latest_state.observation_id == observation.id:
                scene_state = latest_state

        nearest_memory = self.store.search_by_vector(
            observation.embedding,
            top_k=1,
            session_id=session_id,
            exclude_id=observation.id,
        )
        memory_match = nearest_memory[0] if nearest_memory else None
        summary_source = (
            observation.summary
            if observation.summary
            else scene_state.observed_state_summary
            if scene_state and scene_state.observed_state_summary
            else "A live scene captured with continuity-aware memory."
        )
        summary = " ".join(str(summary_source).split())
        if len(summary) > 180:
            summary = f"{summary[:177].rsplit(' ', 1)[0]}..."

        tracked_entities = len(scene_state.entity_track_ids) if scene_state else 0
        persistence_confidence = scene_state.metrics.persistence_confidence if scene_state else None
        memory_match_score = memory_match.score if memory_match else None

        detail_clauses: list[str] = []
        if tracked_entities > 0:
            noun = "entity" if tracked_entities == 1 else "entities"
            if persistence_confidence is not None:
                detail_clauses.append(
                    f"kept {tracked_entities} tracked {noun} active at {round(persistence_confidence * 100)}% persistence"
                )
            else:
                detail_clauses.append(f"kept {tracked_entities} tracked {noun} active")
        if memory_match_score is not None:
            detail_clauses.append(f"found a {round(memory_match_score * 100)}% memory match")

        share_text = f"I just used Toori to analyze a live scene: {summary.rstrip('.!?')}."
        if detail_clauses:
            share_text += f" It {' and '.join(detail_clauses)}."
        share_text += f" Try it: {public_url}"

        if tracked_entities > 0:
            title = f"Tracked {tracked_entities} live {'entity' if tracked_entities == 1 else 'entities'}"
        elif memory_match_score is not None:
            title = "Matched this scene to memory"
        else:
            title = "Live scene analysis"

        return ShareObservationResponse(
            session_id=session_id,
            observation_id=observation.id,
            title=title,
            summary=summary,
            share_text=share_text,
            share_url=public_url,
            tracked_entities=tracked_entities,
            persistence_confidence=persistence_confidence,
            memory_match_score=memory_match_score,
        )

    def record_observation_share_event(self, session_id: str, observation_id: str, event_type: str) -> None:
        observation = self.store.get_observation(observation_id)
        if observation is None or observation.session_id != session_id:
            raise KeyError(observation_id)
        get_logger("runtime").info(
            "observation_share_event",
            session_id=session_id,
            observation_id=observation_id,
            event_type=event_type,
        )

    def latest_proof_report(self) -> Optional[Path]:
        return self._latest_proof_report

    def _engine_for_session(self, session_id: str):
        if session_id not in self._immersive_engines:
            engine = self._immersive_engine_factory()
            self._load_sag_templates_into_engine(engine)
            self._immersive_engines[session_id] = engine
        return self._immersive_engines[session_id]

    def _atlas_for_session(self, session_id: str) -> EpistemicAtlas:
        atlas = self._atlases.get(session_id)
        if atlas is None:
            atlas = EpistemicAtlas()
            self._atlases[session_id] = atlas
        return atlas

    def _load_sag_templates_into_engine(self, engine: Any) -> None:
        from cloud.jepa_service.anchor_graph import SemanticAnchorGraph

        if not hasattr(engine, "_sag") or not isinstance(engine._sag, SemanticAnchorGraph):
            return
        try:
            engine._sag.load_learned_templates(self._sag_templates_path)
        except Exception as exc:
            get_logger("runtime").warning("sag_template_load_failed", error=str(exc))

    def _load_sag_templates(self) -> None:
        for engine in self._immersive_engines.values():
            self._load_sag_templates_into_engine(engine)

    def _save_sag_templates(self) -> None:
        from cloud.jepa_service.anchor_graph import SemanticAnchorGraph

        for engine in self._immersive_engines.values():
            if hasattr(engine, "_sag") and isinstance(engine._sag, SemanticAnchorGraph):
                try:
                    engine._sag.save_learned_templates(self._sag_templates_path)
                except Exception as exc:
                    get_logger("runtime").warning("sag_template_save_failed", error=str(exc))

    def _prime_forecast_engine(
        self,
        session_id: str,
        state_vector: list[float] | None,
        forecast_errors: dict[int, float],
    ) -> None:
        if state_vector is None:
            return
        engine = self._engine_for_session(session_id)
        engine._last_state = np.asarray(state_vector, dtype=np.float32)
        engine._last_forecast_errors = {int(key): float(value) for key, value in forecast_errors.items()}

    def _ensure_jepa_pool(self) -> JEPAWorkerPool:
        if self._jepa_pool is None:
            self._jepa_pool = JEPAWorkerPool()
        return self._jepa_pool

    def _load_frame_array(self, image_path: str) -> np.ndarray:
        with Image.open(image_path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _talker_event_from_jepa(self, tick: JEPATick, entity_tracks: list) -> TalkerEvent | None:
        if not tick.talker_event:
            return None
        event_type = tick.talker_event
        names = [track.label for track in entity_tracks[:2]]
        name = names[0] if names else "entity"
        descriptions = {
            "ENTITY_APPEARED": "Something new just entered",
            "ENTITY_DISAPPEARED": f"{name} left - I'll remember it",
            "OCCLUSION_START": f"I'm still tracking {name} even though I can't see it",
            "OCCLUSION_END": f"{name} came back - exactly where I expected",
            "PREDICTION_VIOLATION": "That wasn't supposed to happen",
            "SCENE_STABLE": "Everything is as expected",
        }
        return TalkerEvent(
            event_type=event_type,  # type: ignore[arg-type]
            confidence=max(0.0, min(1.0, tick.mean_energy)),
            entity_ids=[track.id for track in entity_tracks[:4]],
            energy_summary=tick.mean_energy,
            description=descriptions.get(event_type, "Unexpected change detected"),
        )

    def _emit_reasoning_latency(self, trace: list[ReasoningTraceEntry]) -> None:
        for entry in trace:
            if entry.provider not in {"ollama", "mlx"}:
                continue
            if not entry.attempted or entry.latency_ms is None:
                continue
            self.events.publish(
                "llm_latency",
                {
                    "provider": entry.provider,
                    "ms": float(entry.latency_ms),
                    "success": entry.success,
                },
            )

    def evaluate_challenge(self, request: ChallengeEvaluateRequest) -> ChallengeRun:
        if request.observation_ids:
            observations = self.store.get_observations_by_ids(request.observation_ids)
        else:
            observations = list(
                reversed(self.store.recent_observations(session_id=request.session_id, limit=request.limit))
            )
        world_states = [
            self.store.get_scene_state(observation.world_state_id)
            for observation in observations
            if observation.world_state_id
        ]
        scene_states = [state for state in world_states if state is not None]
        if not scene_states:
            scene_states = list(
                reversed(self.store.recent_scene_states(session_id=request.session_id, limit=request.limit))
            )
        challenge = build_challenge_run(
            session_id=request.session_id,
            challenge_set=request.challenge_set,
            proof_mode=request.proof_mode,
            observations=observations,
            scene_states=scene_states,
        )
        self.store.save_challenge_run(challenge)
        self.events.publish(
            "challenge.updated",
            {"challenge": challenge.model_dump(mode="json")},
        )
        return challenge

    def query(self, request: QueryRequest) -> QueryResponse:
        settings = self.get_settings()
        answer: Optional[Answer] = None
        reasoning_trace: list[ReasoningTraceEntry] = []

        if request.image_base64 or request.file_path:
            image_bytes, image = self._load_image(request.image_base64, request.file_path)
            embedding, _, _, _ = self.providers.perceive(settings, image)
            hits = self.store.search_by_vector(
                embedding,
                top_k=request.top_k,
                session_id=request.session_id,
                time_window_s=request.time_window_s,
                exclude_id=None,
            )
            if request.query:
                context = [
                    self.store.get_observation(hit.observation_id)
                    for hit in hits
                    if self.store.get_observation(hit.observation_id) is not None
                ]
                outcome = self.providers.reason(
                    settings,
                    prompt=request.query,
                    image_bytes=image_bytes,
                    image_path=None,
                    context=[item for item in context if item is not None],
                )
                answer = outcome.answer
                reasoning_trace = outcome.trace
                self._emit_reasoning_latency(reasoning_trace)
        else:
            hits = self.store.search_by_text(
                request.query or "",
                top_k=request.top_k,
                session_id=request.session_id,
                time_window_s=request.time_window_s,
            )
            if request.query and hits:
                context = [
                    self.store.get_observation(hit.observation_id)
                    for hit in hits
                    if self.store.get_observation(hit.observation_id) is not None
                ]
                outcome = self.providers.reason(
                    settings,
                    prompt=request.query,
                    image_bytes=None,
                    image_path=None,
                    context=[item for item in context if item is not None],
                )
                answer = outcome.answer
                reasoning_trace = outcome.trace
                self._emit_reasoning_latency(reasoning_trace)

        return QueryResponse(
            hits=hits,
            answer=answer,
            provider_health=self.providers.health_snapshot(settings),
            reasoning_trace=reasoning_trace,
        )

    def _analyze_with_world_model(
        self,
        request: AnalyzeRequest,
        _is_tick: bool = False,
    ) -> tuple[AnalyzeResponse, "SceneState", list]:
        settings = self.get_settings()
        self.store.prune(settings.retention_days)

        image_bytes, image = self._load_image(request.image_base64, request.file_path)
        embedding, provider_name, confidence, provider_metadata = self.providers.perceive(settings, image)
        if provider_name != "basic":
            _, _, basic_metadata = self.providers.basic.perceive(image)
            provider_metadata = {**basic_metadata, **provider_metadata}

        previous = self.store.recent_observations(session_id=request.session_id, limit=1)
        novelty = 1.0
        if previous:
            novelty = max(0.0, 1.0 - self._cosine_similarity(embedding, previous[0].embedding))

        observation = self.store.create_observation(
            image=image,
            raw_bytes=image_bytes,
            embedding=embedding,
            session_id=request.session_id,
            confidence=confidence,
            novelty=novelty,
            source_query=request.query,
            tags=request.tags + self._fallback_tags(provider_metadata),
            providers=[provider_name],
            metadata={"perception": provider_metadata},
        )

        proposal_boxes = self.providers.object_proposals(
            settings,
            image,
            provider_name=provider_name,
        )

        hits = self.store.search_by_vector(
            embedding,
            top_k=request.top_k or settings.top_k,
            session_id=request.session_id,
            exclude_id=observation.id,
            time_window_s=request.time_window_s,
        )

        answer, reasoning_trace = self._maybe_reason(
            settings=settings,
            request=request,
            observation=observation,
            image_bytes=image_bytes,
            _is_tick=_is_tick,
        )
        summary_text, summary_metadata = build_object_summary(
            observation,
            provider_metadata,
            proposal_boxes,
            answer_text=answer.text if answer is not None else None,
            query=request.query,
        )
        if answer is not None:
            observation = self.store.update_observation(
                observation.id,
                summary=summary_text,
                providers=observation.providers + [answer.provider],
                metadata={
                    **observation.metadata,
                    "answer": answer.model_dump(mode="json"),
                    "reasoning_trace": [entry.model_dump(mode="json") for entry in reasoning_trace],
                    "object_proposals": [box.model_dump(mode="json") for box in proposal_boxes],
                    "proposal_boxes": [box.model_dump(mode="json") for box in proposal_boxes],
                    **summary_metadata,
                },
            )
        else:
            observation = self.store.update_observation(
                observation.id,
                summary=summary_text,
                metadata={
                    **observation.metadata,
                    "reasoning_trace": [entry.model_dump(mode="json") for entry in reasoning_trace],
                    "object_proposals": [box.model_dump(mode="json") for box in proposal_boxes],
                    "proposal_boxes": [box.model_dump(mode="json") for box in proposal_boxes],
                    **summary_metadata,
                },
            )

        previous_state = self.store.latest_scene_state(request.session_id)
        recent_observations = [
            item
            for item in self.store.recent_observations(session_id=request.session_id, limit=6)
            if item.id != observation.id
        ]
        existing_tracks = self.store.list_entity_tracks(session_id=request.session_id, limit=32)
        scene_state, entity_tracks = build_scene_state(
            observation=observation,
            hits=hits,
            previous_state=previous_state,
            recent_observations=recent_observations,
            existing_tracks=existing_tracks,
        )
        self.store.save_scene_state(scene_state)
        self.store.save_entity_tracks(entity_tracks)
        observation = self.store.update_observation(
            observation.id,
            world_state_id=scene_state.id,
            metadata={
                **observation.metadata,
                "world_model": scene_state.metrics.model_dump(mode="json"),
                "scene_state_id": scene_state.id,
                "proposal_boxes": [box.model_dump(mode="json") for box in proposal_boxes],
            },
        )

        self.events.publish(
            "observation.created",
            {"observation": observation.model_dump(mode="json")},
        )
        if answer is not None:
            self.events.publish(
                "answer.ready",
                {"observation_id": observation.id, "answer": answer.model_dump(mode="json")},
            )
        self.events.publish(
            "search.ready",
            {
                "observation_id": observation.id,
                "hits": [hit.model_dump(mode="json") for hit in hits],
            },
        )
        self.events.publish(
            "world_state.updated",
            {"scene_state": scene_state.model_dump(mode="json")},
        )
        for track in entity_tracks[:8]:
            self.events.publish(
                "entity_track.updated",
                {"track": track.model_dump(mode="json")},
            )

        response = AnalyzeResponse(
            observation=observation,
            hits=hits,
            answer=answer,
            provider_health=self.providers.health_snapshot(settings),
            reasoning_trace=reasoning_trace,
        )
        return response, scene_state, entity_tracks

    def _maybe_reason(
        self,
        *,
        settings: RuntimeSettings,
        request: AnalyzeRequest,
        observation: Observation,
        image_bytes: bytes,
        _is_tick: bool = False,
    ) -> tuple[Optional[Answer], list[ReasoningTraceEntry]]:
        # During living_lens_tick, suppress reasoning — talker handles output
        if _is_tick and not request.query and request.decode_mode != "force":
            return None, []
        should_reason = request.decode_mode == "force" or bool(request.query)
        should_reason = should_reason or (
            request.decode_mode == "auto" and observation.novelty >= settings.decode_auto_threshold
        )
        if not should_reason:
            return None, []
        context = self.store.recent_observations(session_id=observation.session_id, limit=5)
        prompt = request.query or "Describe the live scene, focus on actionable objects, activities, and visual changes."
        outcome = self.providers.reason(
            settings,
            prompt=prompt,
            image_bytes=image_bytes,
            image_path=observation.image_path,
            context=context,
        )
        self._emit_reasoning_latency(outcome.trace)
        return outcome.answer, outcome.trace

    def _load_image(self, image_base64: Optional[str], file_path: Optional[str]) -> tuple[bytes, Image.Image]:
        if image_base64:
            raw_bytes = base64.b64decode(image_base64)
        elif file_path:
            raw_bytes = Path(file_path).expanduser().read_bytes()
        else:  # pragma: no cover - validated by Pydantic
            raise ValueError("image input missing")
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return raw_bytes, image

    def _load_tool_state_image(self, request: ToolStateObserveRequest) -> tuple[bytes, Image.Image]:
        if request.screenshot_base64 or request.file_path:
            return self._load_image(request.screenshot_base64, request.file_path)

        canvas = Image.new("RGB", (960, 540), color=(247, 249, 252))
        draw = ImageDraw.Draw(canvas)
        headline = request.view_id or request.current_url or f"{request.state_domain} state"
        detail_lines = [
            headline[:80],
            f"focused: {request.focused_target or 'none'}",
            f"errors: {', '.join(request.error_banners[:3]) or 'none'}",
            f"entities: {', '.join(entity.label for entity in request.visible_entities[:4]) or 'none'}",
            f"affordances: {', '.join(affordance.label for affordance in request.affordances[:4]) or 'none'}",
        ]
        draw.rounded_rectangle((32, 32, 928, 508), radius=24, outline=(60, 94, 130), width=3, fill=(255, 255, 255))
        for index, line in enumerate(detail_lines):
            draw.text((56, 60 + (index * 64)), line, fill=(22, 39, 61))
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        return buffer.getvalue(), canvas

    def _tool_state_summary(self, request: ToolStateObserveRequest) -> str:
        entity_labels = [entity.label for entity in request.visible_entities[:4] if entity.label]
        affordance_labels = [affordance.label for affordance in request.affordances[:4] if affordance.label]
        target = request.view_id or request.current_url or f"{request.state_domain} state"
        parts = [f"{request.state_domain} state at {target}"]
        if entity_labels:
            parts.append(f"entities: {', '.join(entity_labels)}")
        if affordance_labels:
            parts.append(f"affordances: {', '.join(affordance_labels)}")
        if request.error_banners:
            parts.append(f"errors: {', '.join(request.error_banners[:2])}")
        return " | ".join(parts)

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        left_vec = np.array(left, dtype=np.float32)
        right_vec = np.array(right, dtype=np.float32)
        denom = (np.linalg.norm(left_vec) or 1.0) * (np.linalg.norm(right_vec) or 1.0)
        return float(np.dot(left_vec, right_vec) / denom)

    def _fallback_summary(self, observation: Observation, metadata: dict, query: Optional[str]) -> str:
        top_label = str(metadata.get("top_label", "")).replace("_", " ").strip()
        dominant = metadata.get("dominant_color")
        brightness = metadata.get("brightness_label")
        edge = metadata.get("edge_label")
        parts = []
        if top_label:
            parts.append(top_label)
        if dominant:
            parts.append(f"{dominant} dominant")
        if brightness:
            parts.append(brightness)
        if edge:
            parts.append(edge)
        summary = "Local observation"
        if parts:
            summary = f"{' '.join(parts)} scene"
        if query:
            summary = f"{summary}; prompt: {query}"
        return summary

    def _fallback_tags(self, metadata: dict) -> list[str]:
        tags = []
        top_label = str(metadata.get("top_label", "")).replace("_", " ").strip()
        if top_label:
            tags.append(top_label)
        for key in ("dominant_color", "brightness_label", "edge_label"):
            value = metadata.get(key)
            if value:
                tags.append(str(value))
        return tags

    def _embed_smriti_query(self, query: str, dim: int = 128) -> np.ndarray:
        seed_bytes = hashlib.blake2b(query.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(seed_bytes, byteorder="big", signed=False)
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(dim).astype(np.float32)
        norm = float(np.linalg.norm(vector)) or 1.0
        return (vector / norm).astype(np.float32)

    def _normalize_setu2_media_embedding(self, embedding: list[float] | np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size > 128:
            vector = vector[:128]
        elif vector.size < 128:
            vector = np.pad(vector, (0, 128 - vector.size))
        norm = float(np.linalg.norm(vector)) or 1.0
        return (vector / norm).astype(np.float32)

    def _embed_setu2_query(self, query: str) -> np.ndarray:
        return self._embed_smriti_query(query, dim=384)

    def _lookup_setu2_media_embedding(self, media_id: str) -> np.ndarray | None:
        media = self.smriti_db.get_smriti_media(media_id)
        if media is not None and media.embedding is not None:
            return self._normalize_setu2_media_embedding(media.embedding)
        if media is not None and media.observation_id:
            observation = self.store.get_observation(media.observation_id)
            if observation is not None:
                return self._normalize_setu2_media_embedding(observation.embedding)
        observation = self.store.get_observation(media_id)
        if observation is not None:
            return self._normalize_setu2_media_embedding(observation.embedding)
        return None

    def _personalize_smriti_results(self, query: str, results: list[Any]) -> list[Any]:
        if not results:
            return results
        from dataclasses import replace
        from cloud.runtime.setu2 import Setu2Bridge

        if self._setu2_bridge is None:
            self._setu2_bridge = Setu2Bridge()

        candidate_pairs: list[tuple[Any, np.ndarray]] = []
        for result in results:
            embedding = self._lookup_setu2_media_embedding(result.media_id)
            if embedding is None:
                continue
            candidate_pairs.append((result, embedding))

        if not candidate_pairs:
            return results

        corpus = np.stack([embedding for _, embedding in candidate_pairs], axis=0)
        energies = self._setu2_bridge.project_query(self._embed_setu2_query(query), corpus)
        updated_results = []
        for (result, _), energy in zip(candidate_pairs, energies):
            personalized_setu = float(1.0 / (1.0 + max(float(energy), 0.0)))
            personalized_hybrid = float((0.6 * float(result.hybrid_score)) + (0.4 * personalized_setu))
            updated_results.append(
                replace(
                    result,
                    setu_score=round(personalized_setu, 6),
                    hybrid_score=round(personalized_hybrid, 6),
                )
            )

        untouched = [result for result in results if all(result.media_id != updated.media_id for updated in updated_results)]
        reranked = updated_results + untouched
        reranked.sort(key=lambda item: item.hybrid_score, reverse=True)
        return reranked

    def _smriti_result_matches_filters(
        self,
        result,
        *,
        person_filter: str | None,
        location_filter: str | None,
        time_start,
        time_end,
        min_confidence: float,
    ) -> bool:
        if person_filter and person_filter not in result.person_names:
            return False
        if location_filter and result.location_name != location_filter:
            return False
        if time_start and result.created_at < time_start:
            return False
        if time_end and result.created_at > time_end:
            return False
        confidence = max(0.0, min(1.0, float(result.hybrid_score)))
        if confidence < min_confidence:
            return False
        return True
