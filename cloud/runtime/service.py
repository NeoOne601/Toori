from __future__ import annotations

import asyncio
import base64
import json
import io
import hashlib
import os
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
import time
from time import perf_counter
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw

from .atlas import EpistemicAtlas
from .config import resolve_data_dir, resolve_smriti_storage
from .error_types import SmritiError, SmritiRateLimitError
from .events import EventBus
from .jepa_worker import JEPAWorkItem, JEPAWorkResult, JEPAWorkerPool, run_jepa_worker_preflight
from .mlx_adapter import make_mlx_bridge_call
from .models import (
    ActionToken,
    AnalyzeRequest,
    AnalyzeResponse,
    Answer,
    BoundingBox,
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
    ObservationSummary,
    ObservationsResponse,
    ObservationSummariesResponse,
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
    RuntimeFeatureStatus,
    RuntimeSettings,
    SceneGraphEdge,
    SceneGraphNode,
    SceneGraphPayload,
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
    RuntimeSnapshotResponse,
    WorldStateResponse,
)
from .observability import CorrelationContext, PipelineTrace, get_logger, trace_stage
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
        gemma4_alert=payload.gemma4_alert,
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
        self._fallback_engines: dict[str, ImmersiveJEPAEngine] = {}
        self._jepa_ticks_by_session: dict[str, deque[JEPATick]] = defaultdict(lambda: deque(maxlen=240))
        self._jepa_pool: JEPAWorkerPool | None = None
        self._jepa_circuit_reason: str | None = None
        self._jepa_safe_fallback_reason: str | None = None
        self._jepa_native_ready = False
        self._jepa_preflight_status = "not_run"
        self._jepa_preflight_device = "cpu"
        self._jepa_preflight_n_frames = 0
        self._jepa_preflight_model_id = ""
        self._jepa_last_failure_at: datetime | None = None
        self._jepa_crash_fingerprint: str | None = None
        self._jepa_native_process_state = "idle"
        self._jepa_last_native_exit_code: int | None = None
        self._jepa_last_native_signal: int | None = None
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
        self._pending_enrichment_task: asyncio.Task | None = None

    @staticmethod
    def _clean_label(value: object) -> str:
        return " ".join(str(value or "").replace("_", " ").split()).strip()

    @classmethod
    def _is_absurd_composite_label(cls, value: object) -> bool:
        normalized = cls._clean_label(value).lower()
        if not normalized:
            return True
        absurd = {
            "hot dog near velvet",
            "hand plane near windsor tie",
            "lemon near bra",
            "rgb histogram+edge histogram",
            "rgb histogram edge histogram",
            "dominant color",
            "brightness label",
            "edge label",
        }
        if normalized in absurd:
            return True
        if any(phrase in normalized for phrase in (" near ", " behind ", " left of ", " right of ", " above ", " below ", " next to ", " beside ", " with ")):
            return True
        if "histogram" in normalized or "descriptor" in normalized:
            return True
        return bool(re.match(r"^(?:entity|proposal|candidate|tracked(?: region| object)?|object)\s*[-_ ]*\d*$", normalized))

    @classmethod
    def _preferred_object_label(cls, *candidates: object) -> str:
        for candidate in candidates:
            cleaned = cls._clean_label(candidate).lower()
            if not cleaned:
                continue
            if cls._is_absurd_composite_label(cleaned):
                continue
            return cleaned
        return ""

    def _fallback_anchor_label(self, label: str, depth_stratum: str = "unknown") -> str:
        normalized = self._clean_label(label)
        key = normalized.lower().replace(" ", "_")
        stratum = str(depth_stratum).lower().strip()

        if key == "cylindrical_object":
            return "object" if stratum == "foreground" else "cylindrical object"

        mapping = {
            "screen_display": "screen",
            "desk_surface": "surface",
            "background_plane": "background",
            "chair_seated": "chair",
            "hand_region": "hand",
            "person_torso": "person",
            "spherical_object": "round object",
            "telescope": "telescope",
            "neck_brace": "neck brace",
            "tie": "tie",
            "wall_clock": "wall clock",
            "poster": "poster",
            "switch": "switch",
            "lamp": "lamp",
            "sunglasses": "sunglasses",
            "lab_coat": "lab coat",
            "barrette": "barrette",
            "unknown": "object",
        }
        if key in mapping:
            return mapping[key]

        if re.match(r"^entity-?\d+$", key) or key in {"object", "proposal_box", "grounded_entity", "unknown_object"}:
            if stratum in {"foreground", "midground", "background"}:
                return f"{stratum} region"
            return "localized region"
        return normalized or "object"

    def _latest_jepa_tick(self) -> JEPATick | None:
        latest_tick: JEPATick | None = None
        latest_ts = -1
        for ticks in self._jepa_ticks_by_session.values():
            if not ticks:
                continue
            candidate = ticks[-1]
            if candidate.timestamp_ms > latest_ts:
                latest_tick = candidate
                latest_ts = candidate.timestamp_ms
        return latest_tick

    @staticmethod
    def _observation_summary(observation: Observation) -> ObservationSummary:
        return ObservationSummary(
            id=observation.id,
            session_id=observation.session_id,
            created_at=observation.created_at,
            world_state_id=observation.world_state_id,
            observation_kind=observation.observation_kind,
            image_path=observation.image_path,
            thumbnail_path=observation.thumbnail_path,
            width=observation.width,
            height=observation.height,
            summary=observation.summary,
            source_query=observation.source_query,
            tags=list(observation.tags),
            confidence=observation.confidence,
            novelty=observation.novelty,
            providers=list(observation.providers),
            metadata=dict(observation.metadata or {}),
        )

    def _record_jepa_native_failure(self, reason: str, *, stage: str) -> None:
        normalized = " ".join(str(reason or "").split()).strip() or "JEPA worker unavailable"
        self._jepa_native_ready = False
        self._jepa_preflight_status = "failed" if stage == "preflight" else "quarantined"
        self._jepa_last_failure_at = datetime.now(timezone.utc)
        self._jepa_crash_fingerprint = hashlib.blake2b(
            f"{stage}:{normalized}".encode("utf-8", "ignore"),
            digest_size=6,
        ).hexdigest()
        self._jepa_native_process_state = "crashed" if stage == "preflight" else "quarantined"

    def _scene_graph_payload(
        self,
        *,
        session_id: str,
        scene_state: SceneState | None,
        tracks: list[EntityTrack],
        latest_tick: JEPATick | None,
    ) -> SceneGraphPayload:
        nodes: list[SceneGraphNode] = []
        edges: list[SceneGraphEdge] = []
        seen_ids: set[str] = set()

        def add_node(node: SceneGraphNode) -> None:
            if node.id in seen_ids:
                return
            seen_ids.add(node.id)
            nodes.append(node)

        if scene_state is not None:
            for entity in scene_state.grounded_entities[:16]:
                bbox_payload = None
                raw_bbox = (entity.properties or {}).get("bbox")
                if isinstance(raw_bbox, dict):
                    try:
                        bbox_payload = BoundingBox.model_validate(raw_bbox)
                    except Exception:
                        bbox_payload = None
                add_node(
                    SceneGraphNode(
                        id=entity.id,
                        label=entity.label,
                        depth_stratum=str((entity.properties or {}).get("depth_stratum") or "unknown"),
                        depth_confidence=float((entity.properties or {}).get("depth_confidence") or 0.0),
                        bbox=bbox_payload,
                        source="grounded_entity",
                        confidence=float(entity.confidence or 0.0),
                        status=str(entity.status or "visible"),
                        label_source=str((entity.properties or {}).get("label_source") or "grounded_entity"),
                        label_evidence=(entity.properties or {}).get("label_evidence"),
                    )
                )

        for track in tracks[:16]:
            metadata = track.metadata or {}
            bbox_payload = None
            raw_bbox = metadata.get("bbox") or metadata.get("ghost_bbox_pixels")
            if isinstance(raw_bbox, dict):
                try:
                    bbox_payload = BoundingBox.model_validate(raw_bbox)
                except Exception:
                    bbox_payload = None
            add_node(
                SceneGraphNode(
                    id=str(track.id),
                    label=self._preferred_object_label(track.label, metadata.get("caption"), metadata.get("top_label")) or str(track.id),
                    depth_stratum=str(metadata.get("depth_stratum") or "unknown"),
                    depth_confidence=float(metadata.get("depth_confidence") or 0.0),
                    bbox=bbox_payload,
                    source="track",
                    confidence=float(track.persistence_confidence or track.last_similarity or 0.0),
                    status=str(track.status or "visible"),
                    label_source=str(metadata.get("label_source") or "track"),
                    label_evidence=metadata.get("label_evidence"),
                )
            )

        for anchor in ((latest_tick.anchor_matches or []) if latest_tick is not None else [])[:16]:
            if not isinstance(anchor, dict):
                continue
            patch_indices = anchor.get("patch_indices") or []
            bbox_payload = anchor.get("bbox_normalized")
            if not bbox_payload and isinstance(patch_indices, list) and patch_indices:
                rows = [int(index) // 14 for index in patch_indices if str(index).isdigit()]
                cols = [int(index) % 14 for index in patch_indices if str(index).isdigit()]
                if rows and cols:
                    bbox_payload = {
                        "x": min(cols) / 14,
                        "y": min(rows) / 14,
                        "width": (max(cols) - min(cols) + 1) / 14,
                        "height": (max(rows) - min(rows) + 1) / 14,
                    }
            bbox_model = None
            if isinstance(bbox_payload, dict):
                try:
                    bbox_model = BoundingBox.model_validate(bbox_payload)
                except Exception:
                    bbox_model = None
            label = self._preferred_object_label(
                anchor.get("open_vocab_label"),
                anchor.get("template_name"),
                anchor.get("name"),
            ) or "anchor"
            anchor_id = str(anchor.get("template_name") or anchor.get("name") or f"anchor-{len(nodes)+1}")
            add_node(
                SceneGraphNode(
                    id=anchor_id,
                    label=label,
                    depth_stratum=str(anchor.get("depth_stratum") or "unknown"),
                    depth_confidence=float(anchor.get("depth_confidence", 0.0) or 0.0),
                    bbox=bbox_model,
                    source="anchor",
                    confidence=float(anchor.get("confidence", 0.0) or 0.0),
                    status="visible",
                    label_source=str(anchor.get("label_source") or "anchor"),
                    label_evidence=anchor.get("label_evidence"),
                )
            )

        primary_id = nodes[0].id if nodes else None
        for node in nodes[1:]:
            if primary_id:
                edges.append(
                    SceneGraphEdge(
                        source=primary_id,
                        target=node.id,
                        relation="co_present",
                        weight=max(float(node.confidence), 0.15),
                    )
                )

        atlas = self._atlas_for_session(session_id).to_dict()
        for edge in atlas.get("edges", [])[:24] if isinstance(atlas, dict) else []:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source_id") or "")
            target = str(edge.get("target_id") or "")
            if not source or not target:
                continue
            edges.append(
                SceneGraphEdge(
                    source=source,
                    target=target,
                    relation="co_occurs",
                    weight=float(edge.get("interaction_energy", 0.0) or 0.0),
                )
            )

        return SceneGraphPayload(nodes=nodes[:24], edges=edges[:40])

    @staticmethod
    def _normalized_anchor_bbox(anchor: dict[str, Any]) -> dict[str, float] | None:
        bbox = anchor.get("bbox_normalized")
        if isinstance(bbox, dict):
            try:
                return {
                    "x": float(bbox.get("x", 0.0)),
                    "y": float(bbox.get("y", 0.0)),
                    "width": float(bbox.get("width", 0.0)),
                    "height": float(bbox.get("height", 0.0)),
                }
            except Exception:
                return None
        patch_indices = anchor.get("patch_indices") or []
        if not isinstance(patch_indices, list) or not patch_indices:
            return None
        numeric_indices: list[int] = []
        for item in patch_indices:
            try:
                numeric_indices.append(int(item))
            except Exception:
                continue
        if not numeric_indices:
            return None
        rows = [index // 14 for index in numeric_indices]
        cols = [index % 14 for index in numeric_indices]
        return {
            "x": min(cols) / 14,
            "y": min(rows) / 14,
            "width": (max(cols) - min(cols) + 1) / 14,
            "height": (max(rows) - min(rows) + 1) / 14,
        }

    def _anchor_crop_base64(
        self,
        *,
        observation: Observation | None,
        anchor: dict[str, Any],
    ) -> str | None:
        if observation is None or not observation.image_path:
            return None
        bbox = self._normalized_anchor_bbox(anchor)
        if bbox is None:
            return None
        try:
            with Image.open(observation.image_path) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                x0 = max(0, int((bbox["x"] - 0.04) * width))
                y0 = max(0, int((bbox["y"] - 0.04) * height))
                x1 = min(width, int((bbox["x"] + bbox["width"] + 0.04) * width))
                y1 = min(height, int((bbox["y"] + bbox["height"] + 0.04) * height))
                if x1 <= x0 or y1 <= y0:
                    return None
                crop = rgb.crop((x0, y0, x1, y1))
                crop.thumbnail((384, 384))
                buffer = io.BytesIO()
                crop.save(buffer, format="JPEG", quality=88)
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception:
            return None

    @staticmethod
    def _anchor_tvlc_context(
        description_records: list[dict[str, Any]],
        index: int,
    ) -> tuple[str | None, str | None]:
        if index >= len(description_records):
            return None, None
        record = description_records[index]
        if not isinstance(record, dict):
            return None, None
        payload = record.get("description")
        if not isinstance(payload, dict):
            return None, None
        context = str(payload.get("tvlc_context") or "").strip() or None
        connector_type = str(payload.get("connector_type") or "").strip() or None
        if connector_type != "tvlc_trained":
            return None, None
        return context, connector_type

    def _mark_jepa_native_ready(self, *, device: str, n_frames: int, model_id: str) -> None:
        self._jepa_native_ready = True
        self._jepa_preflight_status = "ready"
        self._jepa_preflight_device = device
        self._jepa_preflight_n_frames = n_frames
        self._jepa_preflight_model_id = model_id
        self._jepa_safe_fallback_reason = None
        self._jepa_circuit_reason = None
        self._jepa_last_failure_at = None
        self._jepa_crash_fingerprint = None
        self._jepa_native_process_state = "ready"
        self._jepa_last_native_exit_code = 0
        self._jepa_last_native_signal = None

    def _mlx_health(self, settings: RuntimeSettings):
        mlx_config = settings.providers.get("mlx")
        if mlx_config is None:
            return None, None
        return self.providers.mlx.health(mlx_config), mlx_config

    def _feature_health_snapshot(self, settings: RuntimeSettings) -> list[RuntimeFeatureStatus]:
        features = settings.live_features
        latest_tick = self._latest_jepa_tick()
        mlx_health, mlx_config = self._mlx_health(settings)
        circuit_reason = self._jepa_circuit_reason
        safe_fallback_reason = self._jepa_safe_fallback_reason
        active_world_model = (
            latest_tick.last_tick_encoder_type
            if latest_tick is not None and latest_tick.last_tick_encoder_type
            else ("dinov2-vits14-onnx" if safe_fallback_reason else "vjepa2")
        )

        energy_map = np.asarray(getattr(latest_tick, "energy_map", []), dtype=np.float32)
        has_energy_tick = bool(energy_map.size)
        heatmap_active = False
        world_model_degraded = bool((latest_tick is not None and latest_tick.degraded) or circuit_reason)
        if has_energy_tick:
            flat = energy_map.reshape(-1)
            p70 = float(np.percentile(flat, 70)) if flat.size else 0.0
            heatmap_active = bool(np.any(flat >= p70)) and float(np.max(flat)) > 0.0

        heatmap_message = "heatmap disabled"
        heatmap_healthy = False
        if features.energy_heatmap_enabled:
            if circuit_reason:
                heatmap_message = f"world model unavailable: {circuit_reason}"
            elif world_model_degraded:
                heatmap_message = (
                    "native JEPA quarantined; rendering degraded energy overlay"
                    if active_world_model == "dinov2-vits14-onnx"
                    else f"world model degraded: {latest_tick.degrade_reason or 'worker fallback active'}"
                )
            elif latest_tick is None or not has_energy_tick:
                heatmap_message = "awaiting first JEPA tick"
            elif latest_tick.warmup:
                heatmap_message = "warmup active"
            elif not heatmap_active:
                heatmap_message = "no high-energy patches this tick"
            else:
                heatmap_message = "spectrum energy heatmap active"
                heatmap_healthy = True

        tvlc_enabled = bool(features.tvlc_enabled)
        tvlc_trained = False
        if not tvlc_enabled:
            tvlc_status = RuntimeFeatureStatus(
                name="tvlc-connector",
                enabled=False,
                healthy=False,
                message="TVLC disabled",
            )
        else:
            try:
                from cloud.perception.tvlc_connector import TVLCConnector

                if not TVLCConnector.is_available():
                    tvlc_status = RuntimeFeatureStatus(
                        name="tvlc-connector",
                        enabled=True,
                        healthy=False,
                        message="weights missing",
                    )
                else:
                    connector = TVLCConnector(TVLCConnector.default_weights_path())
                    tvlc_trained = bool(connector.is_trained)
                    tvlc_status = RuntimeFeatureStatus(
                        name="tvlc-connector",
                        enabled=True,
                        healthy=tvlc_trained,
                        message="trained connector ready" if tvlc_trained else "present but untrained",
                    )
            except Exception as exc:
                tvlc_status = RuntimeFeatureStatus(
                    name="tvlc-connector",
                    enabled=True,
                    healthy=False,
                    message=f"unavailable: {str(exc).splitlines()[0]}",
                )

        open_vocab_enabled = bool(features.open_vocab_labels_enabled)
        open_vocab_healthy = bool(open_vocab_enabled and tvlc_trained and mlx_health is not None and mlx_health.healthy)
        if not open_vocab_enabled:
            open_vocab_message = "open-vocab labels disabled"
        elif not tvlc_enabled:
            open_vocab_message = "TVLC disabled; deterministic aliases in use"
        elif not tvlc_trained:
            open_vocab_message = "TVLC present but untrained; deterministic aliases in use"
        elif mlx_health is None or not mlx_health.healthy:
            open_vocab_message = "MLX unavailable; deterministic aliases in use"
        else:
            open_vocab_message = "trained TVLC + MLX relabeling ready"

        proof_enabled = bool(mlx_config is not None and mlx_config.enabled)
        proof_healthy = bool(mlx_health is not None and mlx_health.healthy)
        if not proof_enabled:
            proof_message = "Gemma proof narration disabled"
        elif proof_healthy:
            proof_message = "Gemma proof narration ready"
        else:
            proof_message = "proof narration will use fallback text"

        live_lens_enabled = bool(features.live_lens_use_jepa_tick)
        if not live_lens_enabled:
            live_lens_message = "Live Lens uses legacy analyze mode"
            live_lens_healthy = False
        elif circuit_reason:
            live_lens_message = f"Native JEPA quarantined after worker failure: {circuit_reason}"
            live_lens_healthy = False
        elif safe_fallback_reason and latest_tick is None:
            live_lens_message = f"Native JEPA quarantined; {active_world_model} ready and waiting for first tick"
            live_lens_healthy = False
        elif world_model_degraded:
            live_lens_message = (
                f"Native JEPA quarantined; using {active_world_model}"
                if active_world_model == "dinov2-vits14-onnx"
                else f"world model degraded: {latest_tick.degrade_reason or 'worker fallback active'}"
            )
            live_lens_healthy = False
        elif latest_tick is None:
            live_lens_message = "JEPA-backed capture enabled; waiting for first tick"
            live_lens_healthy = False
        elif latest_tick.warmup:
            live_lens_message = (
                f"{active_world_model} capture warming up"
                if active_world_model != "vjepa2"
                else "JEPA-backed capture warming up"
            )
            live_lens_healthy = True
        else:
            live_lens_message = (
                f"Live Lens uses {active_world_model} tick capture"
                if active_world_model != "vjepa2"
                else "Live Lens uses JEPA-backed tick capture"
            )
            live_lens_healthy = True

        return [
            RuntimeFeatureStatus(
                name="live-lens-jepa",
                enabled=live_lens_enabled,
                healthy=live_lens_healthy,
                message=live_lens_message,
            ),
            RuntimeFeatureStatus(
                name="energy-heatmap",
                enabled=bool(features.energy_heatmap_enabled),
                healthy=heatmap_healthy,
                message=heatmap_message,
            ),
            RuntimeFeatureStatus(
                name="open-vocab-labels",
                enabled=open_vocab_enabled,
                healthy=open_vocab_healthy,
                message=open_vocab_message,
                source_provider="mlx",
            ),
            tvlc_status,
            RuntimeFeatureStatus(
                name="proof-report-gemma",
                enabled=proof_enabled,
                healthy=proof_healthy,
                message=proof_message,
                source_provider="mlx",
            ),
        ]

    def _preferred_anchor_summary(self, tick_dict: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        anchors = tick_dict.get("anchor_matches")
        descriptions = tick_dict.get("setu_descriptions")
        if not isinstance(anchors, list) or not anchors:
            return None, {}

        description_records = descriptions if isinstance(descriptions, list) else []
        candidates: list[tuple[float, str]] = []
        for index, anchor in enumerate(anchors):
            if not isinstance(anchor, dict):
                continue
            gate = description_records[index].get("gate", {}) if index < len(description_records) and isinstance(description_records[index], dict) else {}
            if gate and not bool(gate.get("passes", False)):
                continue
            confidence = float(anchor.get("confidence", 0.0) or 0.0)
            if confidence < 0.55:
                continue
            label = self._preferred_object_label(
                anchor.get("open_vocab_label")
                or anchor.get("template_name")
                or anchor.get("name")
            )
            if not label or label == "object":
                continue
            candidates.append((confidence, label))

        if not candidates:
            return None, {}

        candidates.sort(key=lambda item: item[0], reverse=True)
        primary = candidates[0][1]
        secondary = next((label for _, label in candidates[1:] if label != primary), "")
        summary = primary
        metadata = {
            "primary_object_label": primary,
            "secondary_object_labels": [label for _, label in candidates[1:4] if label != primary],
            "spatial_relation_candidates": ["co_present"] if secondary else [],
            "summary_candidates": [label for _, label in candidates[:6]],
            "summary_source": "anchor_matches",
            "open_vocab_labels": [label for _, label in candidates[:6]],
        }
        return summary, metadata

    @staticmethod
    def _extract_tvlc_prototype_label(
        description_records: list[dict[str, Any]],
        index: int,
    ) -> str | None:
        """Extract the top TVLC prototype match label from setu_descriptions.

        Returns the label string when the connector is trained and a
        high-confidence prototype match exists, otherwise None.
        """
        if index >= len(description_records):
            return None
        record = description_records[index]
        if not isinstance(record, dict):
            return None
        payload = record.get("description")
        if not isinstance(payload, dict):
            return None
        connector_type = str(payload.get("connector_type") or "").strip()
        if connector_type != "tvlc_trained":
            return None
        context = str(payload.get("tvlc_context") or "").strip().lower()
        if "prototype_matches=" not in context:
            return None
        # Parse "prototype_matches=label1:score1;label2:score2; ..."
        fragment = context.split("prototype_matches=", 1)[1]
        fragment = fragment.split("slot_activations=", 1)[0]
        for entry in fragment.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            label, _, score_str = entry.partition(":")
            normalized = " ".join(label.replace("_", " ").split()).strip()
            if not normalized:
                continue
            try:
                score = float(score_str)
            except (ValueError, TypeError):
                score = 0.0
            if score >= 0.18:
                return normalized
        return None

    def _is_placeholder_vision_label(self, value: str | None) -> bool:
        if not value: return True
        norm = str(value).lower().replace("_", " ").strip()
        placeholders = [
            "localized object", "foreground", "midground", "background", 
            "unresolved", "entity", "grounded_entity", "identifying"
        ]
        return any(p in norm for p in placeholders)

    async def _async_apply_zero_shot_labels(
        self,
        tracks: list[EntityTrack],
        patch_tokens: np.ndarray,
        tvlc_engine: Any
    ) -> None:
        """Run Zero-Shot Semantic Consensus in the background without blocking the UI/Camera heartbeat"""
        def _score_all():
            updated = False
            for track in tracks:
                if str(track.label or "").strip() and not self._is_placeholder_vision_label(track.label):
                    continue
                box = track.metadata.get("bbox_pixels") if isinstance(track.metadata, dict) else None
                if box:
                    indices = self._get_patch_indices_for_box(box)
                    if indices:
                        name, conf = tvlc_engine.score_anchor(patch_tokens, indices)
                        # Require 0.35 confidence to override the UI placeholder with a real noun
                        if name and conf >= 0.35:
                            track.label = name
                            track.last_similarity = float(conf)
                            if not isinstance(track.metadata, dict):
                                track.metadata = {}
                            track.metadata["primary_object_label"] = name
                            track.metadata["label_source"] = "latent_manifold_gemma4"
                            track.metadata["label_evidence"] = {"model": "tvlc_v2_consensus", "conf": conf}
                            updated = True
            return updated

        # Yield to event loop, execute CPU-bound matrix dot products in thread
        updated = await asyncio.to_thread(_score_all)
        if updated:
            # Commit the newly resolved open-vocab labels to the system database
            self.store.save_entity_tracks(tracks)

    def _apply_fallback_anchor_labels(self, tick_dict: dict[str, Any], *, patch_tokens: np.ndarray | None = None) -> None:
        """ Enrich anchors with semantic labels using TVLC Consensus. """
        anchors = tick_dict.get("anchor_matches")
        descriptions = tick_dict.get("setu_descriptions")
        
        if not isinstance(anchors, list) or not isinstance(descriptions, list):
            return

        desc_list = descriptions if isinstance(descriptions, list) else []
        tvlc_engine = None
        try:
            from cloud.perception.tvlc_connector import TVLCConnector
            if TVLCConnector.is_available():
                if not hasattr(self, "_mandatory_tvlc_connector"):
                    from cloud.perception.tvlc_connector import TVLCConnector
                    self._mandatory_tvlc_connector = TVLCConnector(TVLCConnector.default_weights_path())
                tvlc_engine = self._mandatory_tvlc_connector
        except Exception:
            pass

        # 1. Label ANCHORS (The Graph Dots)
        for index, anchor in enumerate(anchors):
            if not isinstance(anchor, dict): continue
            if str(anchor.get("open_vocab_label") or "").strip(): continue
            
            label, score = self._perform_semantic_pass(tvlc_engine, anchor, patch_tokens, desc_list, index)
            if label:
                self._enrich_node_with_label(anchor, label, score)
            else:
                self._apply_deterministic_fallback(anchor)

    _PERSON_CLASS_TOKENS: frozenset[str] = frozenset({
        "man", "woman", "boy", "girl", "person", "people",
        "human", "figure", "child", "adult", "male", "female",
    })

    _PERSON_SAG_TEMPLATES: frozenset[str] = frozenset({
        "person_torso", "hand_region",
    })

    _NON_PERSON_SAG_TEMPLATES: frozenset[str] = frozenset({
        "cylindrical_object", "screen_display", "desk_surface",
        "background_plane", "chair_seated",
    })

    @staticmethod
    def _label_is_person_class(label: str) -> bool:
        """True if the TVLC label contains any person-class token."""
        tokens = frozenset(label.lower().split())
        return bool(tokens & RuntimeContainer._PERSON_CLASS_TOKENS)

    @staticmethod
    def _label_is_coherent(label: str) -> bool:
        """False if the label is known incoherent multi-slot word salad."""
        tokens = label.lower().split()
        if len(tokens) == 1:
            return True
        if len(tokens) > 3:
            return False
        token_set = frozenset(tokens)
        _INCOHERENT_PAIRS: frozenset[frozenset[str]] = frozenset({
            frozenset({"man", "boy"}),
            frozenset({"woman", "man"}),
            frozenset({"train", "people"}),
            frozenset({"cat", "dog"}),
            frozenset({"woman", "pulling"}),
            frozenset({"man", "hold"}),
            frozenset({"boy", "girl"}),
        })
        if token_set in _INCOHERENT_PAIRS:
            return False
        _GENDER_TOKENS: frozenset[str] = frozenset({
            "man", "boy", "male", "woman", "girl", "female",
        })
        if len(token_set & _GENDER_TOKENS) > 1:
            return False
        return True

    def _perform_semantic_pass(self, engine, node, tokens, descs, idx):
        label = self._extract_tvlc_prototype_label(descs, idx)
        score = 0.0

        if not label and engine is not None and tokens is not None:
            indices = node.get("patch_indices")
            if indices:
                label, score = engine.score_anchor(tokens, indices)
                if score < 0.42:
                    return None, 0.0

        if not label:
            return None, 0.0

        if not self._label_is_coherent(label):
            get_logger("runtime").info(
                "tvlc_compound_rejected",
                label=label,
                node_template=node.get("template_name"),
            )
            return None, 0.0

        if self._label_is_person_class(label):
            sag_template = str(node.get("template_name") or "")
            if sag_template in self._NON_PERSON_SAG_TEMPLATES:
                get_logger("runtime").info(
                    "tvlc_sag_person_veto",
                    tvlc_label=label,
                    sag_template=sag_template,
                )
                return None, 0.0
            depth = str(node.get("depth_stratum") or "")
            if sag_template == "unknown" and depth not in ("foreground", ""):
                get_logger("runtime").info(
                    "tvlc_depth_person_veto",
                    tvlc_label=label,
                    depth_stratum=depth,
                )
                return None, 0.0

        return label, score

    def _enrich_node_with_label(self, node, label, score):
        node["open_vocab_label"] = label
        node["label_source"] = "tvlc_prototype_match"
        node["label_evidence"] = {
            "policy": "mandatory_semantic_pass_consensus",
            "similarity": float(score),
            "model": "tvlc_v2_consensus_4096"
        }

    def _apply_deterministic_fallback(self, node):
        node["open_vocab_label"] = self._fallback_anchor_label(
            node.get("template_name") or node.get("name") or "object",
            node.get("depth_stratum") or "unknown",
        )
        node["label_source"] = "deterministic_alias"

    def _get_patch_indices_for_box(self, box: dict) -> list[int]:
        """ Map pixel-space bounding box to 14x14 grid patch indices. """
        try:
            x, y = box.get("x", 0), box.get("y", 0)
            w, h = box.get("width", 0), box.get("height", 0)
            grid_size = 14
            # Robust mapping assuming 640x480 resampler input
            start_col = max(0, min(grid_size - 1, int(x * grid_size / 640)))
            end_col = max(0, min(grid_size - 1, int((x + w) * grid_size / 640)))
            start_row = max(0, min(grid_size - 1, int(y * grid_size / 480)))
            end_row = max(0, min(grid_size - 1, int((y + h) * grid_size / 480)))
            
            indices = []
            for r in range(start_row, end_row + 1):
                for c in range(start_col, end_col + 1):
                    indices.append(r * grid_size + c)
            return indices
        except Exception:
            return []

    def _strip_disabled_tvlc_context(self, tick_dict: dict[str, Any]) -> None:
        descriptions = tick_dict.get("setu_descriptions")
        if not isinstance(descriptions, list):
            return
        for item in descriptions:
            if not isinstance(item, dict):
                continue
            payload = item.get("description")
            if isinstance(payload, dict):
                payload.pop("tvlc_context", None)
                payload.pop("connector_type", None)

    def _update_observation_from_tick(
        self,
        observation_id: str,
        *,
        base_metadata: dict[str, Any] | None,
        tick_dict: dict[str, Any],
    ) -> Observation | None:
        preferred_summary, preferred_metadata = self._preferred_anchor_summary(tick_dict)
        if not preferred_summary:
            return None
        return self.store.update_observation(
            observation_id,
            summary=preferred_summary,
            metadata={
                **(base_metadata or {}),
                **preferred_metadata,
            },
        )

    def _replace_session_jepa_tick(self, session_id: str, replacement: JEPATick) -> None:
        ticks = self._jepa_ticks_by_session.get(session_id)
        if not ticks:
            return
        for index in range(len(ticks) - 1, -1, -1):
            if ticks[index].timestamp_ms == replacement.timestamp_ms:
                ticks[index] = replacement
                return
        ticks.append(replacement)

    def confirm_entity_label(
        self,
        session_id: str,
        track_id: str,
        label: str,
    ) -> bool:
        """Persist a user-confirmed label for an entity track."""
        cleaned_label = self._clean_label(label)
        if not cleaned_label:
            return False

        tracks = self.store.list_entity_tracks(session_id=session_id, limit=512)
        track = next((item for item in tracks if item.id == track_id), None)
        if track is None:
            return False

        track.metadata = dict(track.metadata or {})
        track.metadata["confirmed_label"] = cleaned_label
        track.label = cleaned_label
        self.store.save_entity_tracks([track])

        patch_indices: list[int] = [
            int(index)
            for index in (track.metadata.get("patch_indices") or [])
            if isinstance(index, (int, np.integer))
        ]
        sag = None
        for engine in (self._immersive_engines.get(session_id), self._fallback_engines.get(session_id)):
            if engine is None:
                continue
            state = getattr(engine, "_track_states", {}).get(track_id)
            if state is None:
                continue
            state.confirmed_label = cleaned_label
            state.label = cleaned_label
            if state.patch_indices:
                patch_indices = [int(index) for index in state.patch_indices]
            if hasattr(engine, "_sag"):
                sag = engine._sag

        for previous_track in self._previous_tracks:
            if previous_track.id != track_id:
                continue
            previous_track.metadata = dict(previous_track.metadata or {})
            previous_track.metadata["confirmed_label"] = cleaned_label
            previous_track.label = cleaned_label

        get_logger("runtime").info(
            "entity_label_confirmed",
            session_id=session_id,
            track_id=track_id,
            label=cleaned_label,
        )

        try:
            if sag is not None and patch_indices:
                sag.learn_template_from_confirmation(
                    region_patches=patch_indices,
                    confirmed_label=cleaned_label,
                    patch_tokens=np.zeros((196, 384), dtype=np.float32),
                )
                self._save_sag_templates()
        except Exception as _exc:
            get_logger("runtime").debug(
                "sag_learn_from_correction_failed",
                error=str(_exc),
            )
        return True

    async def _finish_living_lens_enrichment(
        self,
        *,
        session_id: str,
        observation_id: str,
        tick_dict: dict[str, Any],
        base_metadata: dict[str, Any] | None,
    ) -> None:
        settings = self.get_settings()
        if not settings.live_features.open_vocab_labels_enabled:
            return
        try:
            observation = self.store.get_observation(observation_id)
            await self._enrich_open_vocab_anchor_matches(
                tick_dict,
                settings=settings,
                max_anchors=3,
                observation=observation,
            )
            if settings.live_features.tvlc_enabled is False:
                self._strip_disabled_tvlc_context(tick_dict)
            self._update_observation_from_tick(
                observation_id,
                base_metadata=base_metadata,
                tick_dict=tick_dict,
            )
            payload = JEPATickPayload.model_validate(tick_dict)
            jepa_tick = _jepa_tick_from_payload(payload)
            if jepa_tick.surprise_score is not None and jepa_tick.surprise_score > 0.55:
                try:
                    from .smriti_gemma4_enricher import SmetiGemma4Enricher as _G4E

                    _mlx_p, _mlx_cfg = self._mlx_reasoning_provider()
                    _g4_alert = await _G4E(_mlx_p, _mlx_cfg).live_tick_alert(
                        tick_dict,
                        None,
                        jepa_tick.entity_tracks[:12],
                    )
                    jepa_tick.gemma4_alert = _g4_alert
                except Exception as exc:
                    get_logger("runtime").debug(
                        "living_lens_alert_skipped",
                        error=str(exc),
                        session_id=session_id,
                        observation_id=observation_id,
                    )
            self._replace_session_jepa_tick(session_id, jepa_tick)
            self.events.publish(
                "jepa_tick.enriched",
                {
                    "session_id": session_id,
                    "observation_id": observation_id,
                    "payload": jepa_tick.to_payload().model_dump(mode="json"),
                },
            )
        except Exception as exc:
            get_logger("runtime").warning(
                "living_lens_enrichment_failed",
                error=str(exc),
                session_id=session_id,
                observation_id=observation_id,
            )

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
                        "depth_stratum": match.depth_stratum or result.depth_stratum,
                        "open_vocab_label": match.open_vocab_label,
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

    async def audio_query(self, request: "AudioQueryRequest") -> "AudioQueryResponse":
        """
        Query Smriti using audio embedding.
        Phase 1 (default): same-modal retrieval in audio embedding space.
        Phase 2 (cross_modal=True): project audio into visual embedding space via CLAP,
        then search visual FAISS index — enables 'hum to find video frame'.
        """
        import time
        t0 = time.perf_counter()

        from .models import AudioQueryRequest, AudioQueryResponse, AudioQueryResult

        try:
            import base64
            import numpy as np
            pcm_bytes = base64.b64decode(request.audio_base64)
        except Exception as e:
            raise ValueError(f"Invalid audio_base64: {e}")

        try:
            from cloud.perception.audio_encoder import AudioEncoder
            ae = AudioEncoder()
            query_emb = ae.encode_bytes(pcm_bytes, sample_rate=request.sample_rate)
            query_energy = float(np.mean(np.abs(query_emb)))
        except ImportError:
            # AudioEncoder unavailable (PyAV not installed) — return empty response gracefully
            return AudioQueryResponse(
                results=[],
                query_audio_energy=0.0,
                index_size=0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                encoder="unavailable"
            )

        # Cross-modal or same-modal search
        if request.cross_modal:
            try:
                from cloud.perception.clap_projector import CLAPProjector
                if CLAPProjector.is_available():
                    projector = CLAPProjector(CLAPProjector.default_weights_path())
                    # CLAPProjector expects 512-dim CLAP embedding; pad 384-dim audio embedding
                    clap_input = np.zeros(512, dtype=np.float32)
                    clap_input[:query_emb.shape[0]] = query_emb
                    projected_emb = projector.project(clap_input)
                    raw_hits = self.smriti_db.cross_modal_audio_to_visual(projected_emb, top_k=request.top_k)
                    encoder_label = "clap-cross-modal"
                else:
                    raw_hits = self.smriti_db.audio_faiss_search(query_emb, top_k=request.top_k)
                    encoder_label = "audio-same-modal-fallback-no-clap"
            except Exception as e:
                get_logger("runtime").warning("cross_modal_query_failed", error=str(e))
                raw_hits = self.smriti_db.audio_faiss_search(query_emb, top_k=request.top_k)
                encoder_label = "audio-same-modal-fallback-exception"
        else:
            raw_hits = self.smriti_db.audio_faiss_search(query_emb, top_k=request.top_k)
            encoder_label = "audio_jepa_phase1"

        # Enrich hits with media metadata
        results = []
        for hit in raw_hits:
            media_id = hit["media_id"]
            # Cross-modal hits use 'score', same-modal use 'audio_score'
            audio_score = hit.get("audio_score", hit.get("score", 0.0))

            # Apply confidence filter
            if audio_score < request.confidence_min:
                continue

            # Fetch media record
            try:
                media_row = self.smriti_db.get_smriti_media(media_id)
            except Exception:
                media_row = None

            setu_descs = []
            thumbnail = None
            audio_energy = None
            audio_dur = None
            gemma4_narration = None

            if media_row:
                thumbnail = getattr(media_row, 'thumbnail_path', None)
                audio_energy = getattr(media_row, 'audio_energy', None)
                audio_dur = getattr(media_row, 'audio_duration_seconds', None)
                raw_setu = getattr(media_row, 'setu_descriptions', None) or []
                if isinstance(raw_setu, str):
                    try:
                        import json
                        raw_setu = json.loads(raw_setu)
                    except Exception:
                        raw_setu = []
                setu_descs = raw_setu if isinstance(raw_setu, list) else []

                # Apply depth_stratum filter
                if request.depth_stratum and setu_descs:
                    top_stratum = setu_descs[0].get('gate', {}).get('depth_stratum', '')
                    if top_stratum and top_stratum != request.depth_stratum:
                        continue

                # Extract Gemma4 narration if present
                for desc in setu_descs:
                    if isinstance(desc, dict) and desc.get('gate', {}).get('narrator') == 'gemma4':
                        gemma4_narration = desc.get('description', {}).get('text')
                        break

            results.append(AudioQueryResult(
                media_id=media_id,
                audio_score=round(audio_score, 4),
                rank=len(results) + 1,
                thumbnail_path=thumbnail,
                setu_descriptions=setu_descs[:3],
                audio_energy=audio_energy,
                audio_duration_seconds=audio_dur,
                gemma4_narration=gemma4_narration,
            ))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        index_size = len(getattr(self.smriti_db, '_audio_media_ids', []))

        return AudioQueryResponse(
            results=results,
            query_audio_energy=round(query_energy, 4),
            index_size=index_size,
            latency_ms=round(elapsed_ms, 1),
            encoder=encoder_label
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
            w_diag = np.ones(128, dtype=np.float32)
            for k, v in getattr(self.smriti_db, '_w', {}).items():
                if k.startswith("dim_"):
                    try:
                        idx = int(k[4:])
                        if 0 <= idx < 128:
                            w_diag[idx] = float(v)
                    except ValueError:
                        pass
            self._setu2_bridge = Setu2Bridge(w_diagonal=w_diag)
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
            w_diag = np.ones(128, dtype=np.float32)
            for k, v in getattr(self.smriti_db, '_w', {}).items():
                if k.startswith("dim_"):
                    try:
                        idx = int(k[4:])
                        if 0 <= idx < 128:
                            w_diag[idx] = float(v)
                    except ValueError:
                        pass
            self._setu2_bridge = Setu2Bridge(w_diagonal=w_diag)
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
            and previous.vjepa2_cache_dir == current.vjepa2_cache_dir
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
        effective_cache_dir = self._effective_vjepa2_cache_dir(settings)

        return WorldModelConfig(
            model_path=settings.vjepa2_model_path or "",
            cache_dir=effective_cache_dir,
            n_frames=settings.vjepa2_n_frames or 0,
            effective_model=effective_model,
            download_url="https://huggingface.co/facebook/vjepa2-vitl-fpc64-256",
        )

    def update_vjepa2_settings(self, model_path: str, cache_dir: str, n_frames: int) -> WorldModelConfig:
        settings = self.get_settings()
        updated = settings.model_copy(
            update={
                "vjepa2_model_path": model_path,
                "vjepa2_cache_dir": cache_dir,
                "vjepa2_n_frames": n_frames,
            }
        )
        self.update_settings(updated)
        self._jepa_native_ready = False
        self._jepa_preflight_status = "not_run"
        self._jepa_preflight_device = "cpu"
        self._jepa_preflight_n_frames = 0
        self._jepa_preflight_model_id = ""
        self._jepa_native_process_state = "idle"
        self._jepa_last_native_exit_code = None
        self._jepa_last_native_signal = None
        self._jepa_safe_fallback_reason = None
        self._jepa_circuit_reason = None
        self._jepa_last_failure_at = None
        self._jepa_crash_fingerprint = None
        if self._jepa_pool is not None:
            pool = self._jepa_pool
            self._jepa_pool = None
            try:
                asyncio.run(pool.shutdown())
            except Exception as exc:
                get_logger("runtime").warning("jepa_pool_reset_failed", error=str(exc))
        self._fallback_engines.clear()
        return self.get_vjepa2_settings()

    async def retry_native_jepa(self) -> WorldModelStatus:
        self._jepa_native_ready = False
        self._jepa_preflight_status = "not_run"
        self._jepa_preflight_device = "cpu"
        self._jepa_preflight_n_frames = 0
        self._jepa_preflight_model_id = ""
        self._jepa_native_process_state = "idle"
        self._jepa_last_native_exit_code = None
        self._jepa_last_native_signal = None
        self._jepa_safe_fallback_reason = None
        self._jepa_circuit_reason = None
        self._jepa_last_failure_at = None
        self._jepa_crash_fingerprint = None
        if self._jepa_pool is not None:
            pool = self._jepa_pool
            self._jepa_pool = None
            try:
                await pool.shutdown()
            except Exception as exc:
                get_logger("runtime").warning("jepa_pool_retry_reset_failed", error=str(exc))
        self._fallback_engines.clear()
        return self.get_world_model_status()

    def _effective_vjepa2_model_id(self, settings: RuntimeSettings) -> str:
        env_override = os.environ.get("TOORI_VJEPA2_MODEL", "").strip()
        if env_override:
            if env_override.startswith(("~", "/")):
                return str(Path(env_override).expanduser().resolve())
            else:
                return env_override

        configured = str(settings.vjepa2_model_path or "").strip()
        if configured:
            if configured.startswith(("~", "/")):
                return str(Path(configured).expanduser().resolve())
            else:
                return configured

        from cloud.perception.vjepa2_encoder import _resolve_model_id

        return _resolve_model_id()

    def _effective_vjepa2_cache_dir(self, settings: RuntimeSettings) -> str:
        env_override = os.environ.get("TOORI_VJEPA2_CACHE_DIR", "").strip()
        if env_override:
            return str(Path(env_override).expanduser().resolve())

        configured = str(settings.vjepa2_cache_dir or "").strip()
        if configured:
            return str(Path(configured).expanduser().resolve())

        from cloud.perception.vjepa2_encoder import _resolve_cache_dir

        return str(_resolve_cache_dir())

    def get_world_model_status(self) -> WorldModelStatus:
        from cloud.perception.vjepa2_encoder import (
            _is_test_environment,
            get_vjepa2_encoder,
            has_local_vjepa2_weights,
        )

        settings = self.get_settings()
        test_mode = _is_test_environment()
        encoder_type = "surrogate"
        model_id = "mobilenetv2-12.onnx"
        model_loaded = True
        device = "cpu"
        n_frames = 0
        configured_encoder = "surrogate" if test_mode else "vjepa2"
        last_tick_encoder_type = "surrogate" if test_mode else "not_loaded"
        active_backend = last_tick_encoder_type
        degraded = False
        degrade_reason = None
        degrade_stage = None
        safe_fallback_mode = bool(self._jepa_safe_fallback_reason)

        if not test_mode:
            try:
                if safe_fallback_mode:
                    raise RuntimeError(self._jepa_safe_fallback_reason or "safe fallback mode enabled")
                if self._jepa_native_ready and self._jepa_preflight_model_id:
                    encoder_type = "vjepa2"
                    model_id = self._jepa_preflight_model_id
                    model_loaded = True
                    device = self._jepa_preflight_device
                    n_frames = self._jepa_preflight_n_frames
                    active_backend = "vjepa2"
                elif has_local_vjepa2_weights():
                    encoder = get_vjepa2_encoder()
                    encoder_type = encoder.encoder_type
                    model_id = encoder.model_id
                    model_loaded = encoder.is_loaded
                    device = str(encoder.device)
                    n_frames = encoder.n_frames
                else:
                    raise RuntimeError("V-JEPA2 weights are not cached locally")
            except Exception as exc:
                runtime_log = get_logger("runtime")
                if safe_fallback_mode:
                    runtime_log.info("world_model_status_safe_fallback", error=str(exc))
                else:
                    runtime_log.warning("world_model_status_fallback", error=str(exc))
                # Check for ViT-S/14 ONNX honest fallback
                try:
                    from cloud.perception.vits14_onnx_encoder import ViTS14OnnxEncoder
                    if ViTS14OnnxEncoder.is_available():
                        encoder_type = "dinov2-vits14-onnx"
                        model_id = ViTS14OnnxEncoder.default_model_path()
                        model_loaded = True
                        device = "cpu"
                        last_tick_encoder_type = "dinov2-vits14-onnx"
                        active_backend = "dinov2-vits14-onnx"
                        degraded = bool(safe_fallback_mode)
                        degrade_reason = self._jepa_safe_fallback_reason or "vjepa2_unavailable_dinov2_vits14_active"
                        degrade_stage = "safe_fallback" if safe_fallback_mode else degrade_stage
                    else:
                        degraded = True
                        active_backend = "unavailable"
                        degrade_reason = self._jepa_safe_fallback_reason or "vjepa2_and_vits14_unavailable"
                        degrade_stage = "status"
                except Exception:
                    degraded = True
                    active_backend = "unavailable"
                    degrade_reason = self._jepa_safe_fallback_reason or str(exc)
                    degrade_stage = "status"
        else:
            model_id = "mobilenetv2-12.onnx"
            active_backend = "surrogate"

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
            active_backend = last_tick_encoder_type
            degraded = bool(latest_tick.degraded)
            degrade_reason = latest_tick.degrade_reason
            degrade_stage = latest_tick.degrade_stage
            if latest_tick.world_model_version == "vjepa2":
                encoder_type = "vjepa2"
            elif latest_tick.world_model_version == "dinov2-vits14-onnx":
                encoder_type = "dinov2-vits14-onnx"
            if latest_tick.world_model_version == "surrogate" and not test_mode:
                encoder_type = "surrogate"
        elif self._jepa_circuit_reason:
            degraded = True
            degrade_reason = self._jepa_circuit_reason
            degrade_stage = "disabled"
            last_tick_encoder_type = "unavailable"
            active_backend = "unavailable"
        elif safe_fallback_mode:
            degraded = True
            degrade_reason = self._jepa_safe_fallback_reason
            degrade_stage = "safe_fallback"
            last_tick_encoder_type = "dinov2-vits14-onnx"
            active_backend = "dinov2-vits14-onnx"

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
            active_backend=active_backend,
            native_ready=bool(self._jepa_native_ready),
            preflight_status=self._jepa_preflight_status,
            last_failure_at=self._jepa_last_failure_at,
            crash_fingerprint=self._jepa_crash_fingerprint,
            native_process_state=self._jepa_native_process_state,
            last_native_exit_code=self._jepa_last_native_exit_code,
            last_native_signal=self._jepa_last_native_signal,
            retryable_native_failure=bool(self._jepa_crash_fingerprint),
        )

    def provider_health(self) -> ProviderHealthResponse:
        settings = self.get_settings()
        return ProviderHealthResponse(
            providers=self.providers.health_snapshot(settings),
            features=self._feature_health_snapshot(settings),
        )

    def list_observations(
        self,
        *,
        session_id: Optional[str],
        limit: int = 50,
        summary_only: bool = False,
    ) -> ObservationsResponse | ObservationSummariesResponse:
        observations = self.store.list_observations(session_id=session_id, limit=limit)
        if summary_only:
            return ObservationSummariesResponse(
                observations=[self._observation_summary(observation) for observation in observations],
            )
        return ObservationsResponse(observations=observations)

    def get_runtime_snapshot(self, session_id: str, *, observation_limit: int = 12) -> RuntimeSnapshotResponse:
        observations = self.store.recent_observations(session_id=session_id, limit=observation_limit)
        tracks = self.store.list_entity_tracks(session_id=session_id, limit=32)
        ticks = self._jepa_ticks_by_session.get(session_id)
        latest_tick = ticks[-1] if ticks else None
        current_scene = self.store.latest_scene_state(session_id=session_id)
        return RuntimeSnapshotResponse(
            session_id=session_id,
            current=current_scene,
            entity_tracks=tracks,
            latest_jepa_tick=latest_tick.to_payload() if latest_tick is not None else None,
            world_model_status=self.get_world_model_status(),
            observations=[self._observation_summary(observation) for observation in observations],
            observation_count=self.store.count_observations(session_id=session_id),
            scene_graph=self._scene_graph_payload(
                session_id=session_id,
                scene_state=current_scene,
                tracks=tracks,
                latest_tick=latest_tick,
            ),
        )

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
        frame = await asyncio.to_thread(
            self._load_frame_array, response.observation.image_path
        )
        # Pre-compute DINOv2+SAM in parent process before dispatching
        # to the worker pool. This eliminates the duplicate perception
        # call inside the worker subprocess (~1.08GB torch savings),
        # preventing the Signal 11 SIGSEGV on 8GB M1. DINOv2+SAM
        # quality is fully preserved — computed once by the parent.
        patch_tokens_bytes, mask_results_json = (
            await asyncio.to_thread(
                self._precompute_perception_for_worker, frame
            )
        )
        # If parent-side perception failed, fall back to inline JEPA
        # immediately. Workers now have TOORI_JEPA_WORKER_NO_PERCEPTION=1
        # and will raise if they receive a bare frame without
        # pre-computed tokens.
        if patch_tokens_bytes is None and self._jepa_pool is not None:
            get_logger("runtime").warning(
                "worker_perception_precompute_unavailable",
                session_id=request.session_id,
                action="falling back to inline JEPA",
            )
            correlation_id = CorrelationContext.get()
            if correlation_id == "no-correlation":
                correlation_id = CorrelationContext.new()
            result = self._inline_jepa_result(
                session_id=request.session_id,
                observation_id=response.observation.id,
                frame=frame,
                correlation_id=correlation_id,
                fallback_reason="parent perception pre-computation failed",
            )
        else:
            result = await self._run_jepa_tick(
                session_id=request.session_id,
                observation_id=response.observation.id,
                frame=frame,
                priority=0 if request.query else 3,
                precomputed_patch_tokens=patch_tokens_bytes,
                precomputed_mask_results_json=mask_results_json,
            )
        settings = self.get_settings()
        patch_tokens_arr = None
        if patch_tokens_bytes is not None:
            try:
                patch_tokens_arr = np.frombuffer(patch_tokens_bytes, dtype=np.float32).reshape(196, 384)
            except Exception:
                pass
        
        self._apply_fallback_anchor_labels(result.jepa_tick_dict, patch_tokens=patch_tokens_arr)
        if settings.live_features.tvlc_enabled is False:
            self._strip_disabled_tvlc_context(result.jepa_tick_dict)
        observation = self._update_observation_from_tick(
            response.observation.id,
            base_metadata=response.observation.metadata if isinstance(response.observation.metadata, dict) else None,
            tick_dict=result.jepa_tick_dict,
        )
        if observation is not None:
            response = response.model_copy(update={"observation": observation})
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

        scene_state.metrics.energy_activation_score = round(float(min(jepa_tick.mean_energy, 1.0)), 4)

        threshold = float(self._jepa_energy_ema[request.session_id] + (2.0 * jepa_tick.energy_std))
        should_talk = bool(jepa_tick.talker_event)
        # ADAPTIVE HEATMAP THROTTLING: 
        # Only broadcast at high frequency if energy is actually rising,
        # otherwise throttle to 10Hz to save CPU/Network for SAM2 tracking stability.
        now_ms = int(time.time() * 1000)
        last_heatmap_ms = getattr(self, "_last_heatmap_broadcast_ms", 0)
        is_high_energy = jepa_tick.mean_energy > 0.4
        
        if is_high_energy or (now_ms - last_heatmap_ms > 100):
            self._last_heatmap_broadcast_ms = now_ms
            self.events.publish(
                "jepa.energy_map",
                {
                    "grid": [14, 14],
                    # OPTIMIZATION: Ravel and send as float32 list with minimal overhead
                    "values": jepa_tick.energy_map.ravel().tolist(),
                    "mean_energy": float(jepa_tick.mean_energy),
                    "threshold": threshold,
                    "should_talk": should_talk,
                    "sigreg_loss": float(jepa_tick.sigreg_loss or 0),
                },
            )
        self.events.publish(
            "jepa_tick",
            {
                "session_id": request.session_id,
                "observation_id": response.observation.id,
                "payload": jepa_tick.to_payload().model_dump(mode="json"),
            },
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
        if settings.live_features.open_vocab_labels_enabled:
            # Cancel any pending enrichment to prevent GPU crash accumulation
            if self._pending_enrichment_task is not None and not self._pending_enrichment_task.done():
                self._pending_enrichment_task.cancel()
            self._pending_enrichment_task = asyncio.create_task(
                self._finish_living_lens_enrichment(
                    session_id=request.session_id,
                    observation_id=response.observation.id,
                    tick_dict=result.jepa_tick_dict,
                    base_metadata=dict(response.observation.metadata or {}) if isinstance(response.observation.metadata, dict) else None,
                )
            )

        # 6. ENRICHMENT: Universal Semantic Labeling for all Tracked Entities
        # We spawn this as an asyncio background task that mutates the in-memory Pydantic
        # tracks exactly so we do not block the 30 FPS camera heartbeat tick response.
        try:
            if hasattr(self, "tvlc") and self.tvlc is not None and jepa_tick.patch_tokens is not None:
                # 6(a). Spawn UI Track labeling directly on the persistent Pydantic models
                asyncio.create_task(
                    self._async_apply_zero_shot_labels(
                        tracks=immersive_tracks,
                        patch_tokens=jepa_tick.patch_tokens,
                        tvlc_engine=self.tvlc
                    )
                )
                
                # 6(b). Still enrich the anchors array (which the MLX agent loops over later)
                tick_payload = jepa_tick.to_payload().model_dump(mode="json")
                self._apply_fallback_anchor_labels(tick_payload, patch_tokens=jepa_tick.patch_tokens)
        except Exception as e:
            # Defensive: Log and continue so a labeling failure doesn't crash the heartbeat
            print(f"[warning] Semantic labeling failed for tick: {e}")

        return LivingLensTickResponse(
            **response.model_dump(),
            scene_state=scene_state,
            entity_tracks=immersive_tracks,
            baseline_comparison=baseline,
            talker_event=talker_event,
            jepa_tick=jepa_tick.to_payload(),
        )

    def _mlx_reasoning_provider(self):
        settings = self.get_settings()
        mlx_config = settings.providers.get("mlx")
        if mlx_config is None or not mlx_config.enabled:
            return None, mlx_config
        return self.providers.mlx, mlx_config

    def _mlx_bridge_call(self):
        provider, config = self._mlx_reasoning_provider()
        if provider is None or config is None:
            return None
        return make_mlx_bridge_call(provider, config)

    async def _enrich_open_vocab_anchor_matches(
        self,
        tick_dict: dict[str, Any],
        *,
        settings: RuntimeSettings | None = None,
        max_anchors: int | None = None,
        observation: Observation | None = None,
    ) -> None:
        anchors = tick_dict.get("anchor_matches")
        if not isinstance(anchors, list) or not anchors:
            return
        settings = settings or self.get_settings()
        self._apply_fallback_anchor_labels(tick_dict)
        if not settings.live_features.open_vocab_labels_enabled:
            return
        descriptions = tick_dict.get("setu_descriptions")
        description_records = descriptions if isinstance(descriptions, list) else []

        from .smriti_gemma4_enricher import SmetiGemma4Enricher as _G4E

        mlx_provider, mlx_config = self._mlx_reasoning_provider()
        enricher = _G4E(mlx_provider, mlx_config)
        candidates: list[tuple[float, int, int, dict[str, Any], dict[str, Any], str | None, str | None]] = []
        for index, anchor in enumerate(anchors):
            if not isinstance(anchor, dict):
                continue
            # Skip if TVLC already confidently matched a prototype!
            if anchor.get("label_source") == "tvlc_prototype_match" and str(anchor.get("open_vocab_label") or "").strip():
                continue
                
            gate = description_records[index].get("gate", {}) if index < len(description_records) and isinstance(description_records[index], dict) else {}
            if gate and not bool(gate.get("passes", False)):
                continue
            confidence = float(anchor.get("confidence", 0.0) or 0.0)
            if confidence < 0.55:
                continue
            tvlc_context, connector_type = self._anchor_tvlc_context(description_records, index)
            candidates.append(
                (
                    confidence,
                    len(anchor.get("patch_indices") or []),
                    index,
                    anchor,
                    gate,
                    tvlc_context,
                    connector_type,
                )
            )
        if not candidates:
            return
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        
        # Hard limit MLX bounding-box fallback reasoning to max 1 sequential call.
        # This prevents 55-second blocks and Metal CompletionQueue errors during high motion!
        allowed = min(max(int(max_anchors), 0) if max_anchors is not None else 1, 1)
        candidates = candidates[:allowed]
        
        for candidate in candidates:
            _confidence, _patch_count, index, anchor, gate, tvlc_context, connector_type = candidate
            try:
                selected_label, selected_evidence = await enricher.get_open_vocab_label_with_evidence(
                    anchor_name=str(anchor.get("template_name") or "object"),
                    depth_stratum=str(anchor.get("depth_stratum") or gate.get("depth_stratum") or "unknown"),
                    confidence=_confidence,
                    patch_count=_patch_count,
                    tvlc_context=tvlc_context,
                    connector_type=connector_type,
                    image_base64=self._anchor_crop_base64(observation=observation, anchor=anchor),
                )
                if isinstance(anchor, dict) and str(selected_label or "").strip():
                    anchor["open_vocab_label"] = str(selected_label).strip()
                    anchor["label_source"] = "open_vocab_tvlc_mlx" if connector_type == "tvlc_trained" else "deterministic_alias"
                    anchor["label_evidence"] = {
                        "policy": "strict_evidence_gate",
                        "connector_type": connector_type,
                        "label_gate_passed": bool((selected_evidence or {}).get("semantic_gate_passed", True)),
                        "anchor_index": index,
                        **(selected_evidence or {}),
                    }
            except Exception:
                pass

    async def shutdown(self) -> None:
        if self._jepa_pool is not None:
            await self._jepa_pool.shutdown()
            self._jepa_pool = None
        self._fallback_engines.clear()
        self._immersive_engines.clear()
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
            *[label for label in (self._preferred_object_label(entity.label) for entity in request.visible_entities) if label],
            *[label for label in (self._preferred_object_label(affordance.label) for affordance in request.affordances) if label],
            *[banner for banner in request.error_banners if banner],
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
            "summary_candidates": [
                label
                for label in (self._preferred_object_label(entity.get("label")) for entity in grounded_entities)
                if label and label != "object"
            ][:6],
            "primary_object_label": (
                self._preferred_object_label(request.visible_entities[0].label)
                if request.visible_entities and self._preferred_object_label(request.visible_entities[0].label) != "object"
                else None
            )
            or (
                self._preferred_object_label(request.view_id)
                if self._preferred_object_label(request.view_id) != "object"
                else None
            ),
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

    async def generate_proof_report(self, session_id: str, chart_b64: str | None = None) -> ProofReportResponse:
        from .proof_report import generate_proof_report, aggregate_stats

        ticks = list(self._jepa_ticks_by_session.get(session_id, deque()))
        stats = aggregate_stats(ticks)

        narration_text = "Analysis skipped due to provider unavailability."
        analysis_source = "skipped"
        try:
            settings = self.get_settings()
            mlx_config = settings.providers.get("mlx")
            mlx_p = self.providers.get("mlx")
            mlx_health = mlx_p.health(mlx_config) if mlx_p is not None and mlx_config is not None else None
            if mlx_p is not None and mlx_config is not None and mlx_health is not None and mlx_health.healthy:
                from .gemma4_bridge import Gemma4Bridge
                bridge_call = self._mlx_bridge_call()
                if bridge_call is None:
                    raise RuntimeError("MLX reasoning provider unavailable")
                bridge = Gemma4Bridge(bridge_call)
                narration_text = await bridge.narrate_proof_report(stats)
                analysis_source = "gemma4"
        except Exception as e:
            get_logger("runtime").warning("proof_report_narration_failed", error=str(e))
            narration_text = "Analysis unavailable for this export. Core JEPA metrics are still included below."
            analysis_source = "error"

        path = generate_proof_report(
            ticks=ticks,
            session_id=session_id,
            narration_text=narration_text,
            chart_b64=chart_b64,
            analysis_source=analysis_source,
        )
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

    def _fallback_engine_for_session(self, session_id: str):
        if session_id not in self._fallback_engines:
            previous_disable = os.environ.get("TOORI_VJEPA2_DISABLE")
            previous_device = os.environ.get("TOORI_VJEPA2_DEVICE")
            previous_frames = os.environ.get("TOORI_VJEPA2_FRAMES")
            os.environ["TOORI_VJEPA2_DISABLE"] = "1"
            os.environ.setdefault("TOORI_VJEPA2_DEVICE", "cpu")
            os.environ.setdefault("TOORI_VJEPA2_FRAMES", "4")
            try:
                engine = self._immersive_engine_factory(device="cpu")
            finally:
                if previous_disable is None:
                    os.environ.pop("TOORI_VJEPA2_DISABLE", None)
                else:
                    os.environ["TOORI_VJEPA2_DISABLE"] = previous_disable
                if previous_device is None:
                    os.environ.pop("TOORI_VJEPA2_DEVICE", None)
                else:
                    os.environ["TOORI_VJEPA2_DEVICE"] = previous_device
                if previous_frames is None:
                    os.environ.pop("TOORI_VJEPA2_FRAMES", None)
                else:
                    os.environ["TOORI_VJEPA2_FRAMES"] = previous_frames
            self._load_sag_templates_into_engine(engine)
            self._fallback_engines[session_id] = engine
        return self._fallback_engines[session_id]

    def _atlas_for_session(self, session_id: str) -> EpistemicAtlas:
        atlas = self._atlases.get(session_id)
        if atlas is None:
            atlas = EpistemicAtlas()
            self._atlases[session_id] = atlas
        return atlas

    def _open_jepa_circuit(self, reason: str) -> None:
        normalized = " ".join(str(reason or "").split()).strip() or "JEPA worker unavailable"
        if self._jepa_circuit_reason == normalized:
            return
        self._jepa_circuit_reason = normalized
        if self._jepa_pool is not None:
            try:
                worker_stats = self._jepa_pool.get_worker_stats()
            except Exception:
                worker_stats = []
            failed = next((item for item in worker_stats if not item.get("alive", True)), None)
            if failed is not None:
                self._jepa_last_native_exit_code = failed.get("last_exit_code")
                self._jepa_last_native_signal = failed.get("last_signal")
        self._record_jepa_native_failure(normalized, stage="runtime")
        get_logger("runtime").warning("jepa_worker_circuit_open", error=normalized)

    def _set_jepa_safe_fallback(self, reason: str, *, stage: str) -> str:
        normalized = " ".join(str(reason or "").split()).strip() or "V-JEPA2 worker crashed"
        self._jepa_safe_fallback_reason = normalized
        self._jepa_circuit_reason = None
        self._record_jepa_native_failure(normalized, stage=stage)
        return normalized

    def _ensure_native_jepa_ready(self) -> bool:
        if self._jepa_safe_fallback_reason:
            return False
        if self._jepa_native_ready:
            return True
        self._jepa_preflight_status = "running"
        self._jepa_native_process_state = "starting"
        result = run_jepa_worker_preflight(timeout_s=45.0)
        self._jepa_last_native_exit_code = result.exit_code
        self._jepa_last_native_signal = result.signal
        if result.ok:
            self._mark_jepa_native_ready(
                device=result.device,
                n_frames=result.n_frames,
                model_id=result.model_id,
            )
            get_logger("runtime").info(
                "jepa_native_preflight_ready",
                device=result.device,
                n_frames=result.n_frames,
                model_id=result.model_id,
            )
            return True
        normalized = self._set_jepa_safe_fallback(result.error or "native preflight failed", stage="preflight")
        if result.crash_fingerprint is not None:
            self._jepa_crash_fingerprint = result.crash_fingerprint
        get_logger("runtime").warning(
            "jepa_native_preflight_failed",
            error=normalized,
            crash_fingerprint=self._jepa_crash_fingerprint,
            exit_code=result.exit_code,
            signal=result.signal,
        )
        return False

    async def _activate_jepa_safe_fallback(self, reason: str) -> None:
        previous_reason = self._jepa_safe_fallback_reason
        normalized = self._set_jepa_safe_fallback(reason, stage="runtime")
        if previous_reason == normalized and self._jepa_pool is None:
            return
        get_logger("runtime").warning("jepa_safe_fallback_enabled", error=normalized)
        if self._jepa_pool is None:
            return
        pool = self._jepa_pool
        self._jepa_pool = None
        try:
            await pool.shutdown()
        except Exception as exc:
            get_logger("runtime").warning("jepa_safe_fallback_shutdown_failed", error=str(exc))

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
        if self._jepa_safe_fallback_reason or self._jepa_circuit_reason:
            raise RuntimeError(self._jepa_safe_fallback_reason or self._jepa_circuit_reason or "native JEPA quarantined")
        if self._jepa_pool is None:
            if not self._ensure_native_jepa_ready():
                raise RuntimeError(self._jepa_safe_fallback_reason or "native JEPA unavailable")
            self._jepa_pool = JEPAWorkerPool(disable_vjepa2=True)
        return self._jepa_pool

    def _degraded_jepa_result(
        self,
        *,
        session_id: str,
        observation_id: str | None,
        correlation_id: str,
        fallback_reason: str,
        degrade_stage: str,
        worker_id: int | None = None,
    ) -> JEPAWorkResult:
        trace = PipelineTrace(correlation_id=correlation_id)
        trace.record_error("jepa_worker", fallback_reason)
        trace.record_error("jepa_worker_pool", degrade_stage)
        tick = JEPATick(
            energy_map=np.zeros((14, 14), dtype=np.float32),
            entity_tracks=[],
            talker_event=None,
            sigreg_loss=0.0,
            forecast_errors={1: 0.0, 2: 0.0, 5: 0.0},
            session_fingerprint=np.zeros(0, dtype=np.float32),
            planning_time_ms=0.0,
            caption_score=0.0,
            retrieval_score=0.0,
            timestamp_ms=int(time.time() * 1000),
            warmup=False,
            mask_results=[],
            mean_energy=0.0,
            energy_std=0.0,
            world_model_version="unavailable",
            configured_encoder="vjepa2",
            last_tick_encoder_type="unavailable",
            degraded=True,
            degrade_reason=fallback_reason,
            degrade_stage=degrade_stage,
        )
        return JEPAWorkResult(
            correlation_id=correlation_id,
            session_id=session_id,
            observation_id=observation_id,
            jepa_tick_dict=tick.to_payload().model_dump(mode="json"),
            pipeline_trace=trace.to_dict(),
            error=None,
            state_vector=None,
            worker_id=worker_id,
        )

    def _inline_jepa_result(
        self,
        *,
        session_id: str,
        observation_id: str | None,
        frame: np.ndarray,
        correlation_id: str,
        fallback_reason: str,
    ) -> JEPAWorkResult:
        trace = PipelineTrace(correlation_id=correlation_id)
        trace.record_error("jepa_worker", fallback_reason)
        with trace_stage(trace, "tick.inline"):
            engine = self._fallback_engine_for_session(session_id)
            tick = engine.tick(
                frame,
                session_id=session_id,
                observation_id=observation_id,
            )
        state_vector = (
            np.asarray(engine._last_state, dtype=np.float32).reshape(-1).tolist()
            if getattr(engine, "_last_state", None) is not None
            else None
        )
        return JEPAWorkResult(
            correlation_id=correlation_id,
            session_id=session_id,
            observation_id=observation_id,
            jepa_tick_dict=tick.to_payload().model_dump(mode="json"),
            pipeline_trace=trace.to_dict(),
            error=None,
            state_vector=state_vector,
            worker_id=None,
        )

    def _safe_fallback_jepa_result(
        self,
        *,
        session_id: str,
        observation_id: str | None,
        frame: np.ndarray,
        correlation_id: str,
        fallback_reason: str,
        degrade_stage: str,
    ) -> JEPAWorkResult:
        trace = PipelineTrace(correlation_id=correlation_id)
        trace.record_error("jepa_worker", fallback_reason)
        trace.record_error("jepa_worker_pool", degrade_stage)
        with trace_stage(trace, "tick.safe_fallback"):
            engine = self._fallback_engine_for_session(session_id)
            tick = engine.tick(
                frame,
                session_id=session_id,
                observation_id=observation_id,
            )
            tick.degraded = True
            tick.degrade_reason = fallback_reason
            tick.degrade_stage = degrade_stage
        state_vector = (
            np.asarray(engine._last_state, dtype=np.float32).reshape(-1).tolist()
            if getattr(engine, "_last_state", None) is not None
            else None
        )
        return JEPAWorkResult(
            correlation_id=correlation_id,
            session_id=session_id,
            observation_id=observation_id,
            jepa_tick_dict=tick.to_payload().model_dump(mode="json"),
            pipeline_trace=trace.to_dict(),
            error=None,
            state_vector=state_vector,
            worker_id=None,
        )

    async def _run_jepa_tick(
        self,
        *,
        session_id: str,
        observation_id: str | None,
        frame: np.ndarray,
        priority: int,
        precomputed_patch_tokens: bytes | None = None,
        precomputed_mask_results_json: str | None = None,
    ) -> JEPAWorkResult:
        correlation_id = CorrelationContext.get()
        if correlation_id == "no-correlation":
            correlation_id = CorrelationContext.new()
        if self._jepa_circuit_reason:
            return self._degraded_jepa_result(
                session_id=session_id,
                observation_id=observation_id,
                correlation_id=correlation_id,
                fallback_reason=self._jepa_circuit_reason,
                degrade_stage="disabled",
            )
        if self._jepa_safe_fallback_reason:
            return self._safe_fallback_jepa_result(
                session_id=session_id,
                observation_id=observation_id,
                frame=frame,
                correlation_id=correlation_id,
                fallback_reason=self._jepa_safe_fallback_reason,
                degrade_stage="safe_fallback",
            )
        try:
            pool = await asyncio.to_thread(self._ensure_jepa_pool)
        except RuntimeError as exc:
            fallback_reason = str(exc)
            degrade_stage = "pool_unavailable"
            if not self._jepa_safe_fallback_reason:
                await self._activate_jepa_safe_fallback(fallback_reason)
                self._open_jepa_circuit(fallback_reason)
            if self._jepa_safe_fallback_reason and not self._jepa_circuit_reason:
                return self._safe_fallback_jepa_result(
                    session_id=session_id,
                    observation_id=observation_id,
                    frame=frame,
                    correlation_id=correlation_id,
                    fallback_reason=self._jepa_safe_fallback_reason or fallback_reason,
                    degrade_stage="safe_fallback",
                )
            return self._degraded_jepa_result(
                session_id=session_id,
                observation_id=observation_id,
                correlation_id=correlation_id,
                fallback_reason=self._jepa_safe_fallback_reason or fallback_reason,
                degrade_stage=degrade_stage,
            )
        work_item = JEPAWorkItem(
            correlation_id=correlation_id,
            session_id=session_id,
            frame_array=frame.tobytes(),
            frame_shape=frame.shape,
            frame_dtype=str(frame.dtype),
            priority=priority,
            observation_id=observation_id,
            precomputed_patch_tokens=precomputed_patch_tokens,
            precomputed_mask_results_json=precomputed_mask_results_json,
        )
        fallback_reason = ""
        degrade_stage = "submit"
        open_circuit = False
        try:
            result = await asyncio.wait_for(pool.submit(work_item), 20.0)
            if result.error is None and result.jepa_tick_dict:
                return result
            fallback_reason = result.error or "worker returned empty JEPA tick"
            degrade_stage = "worker_result"
            open_circuit = True
        except SmritiRateLimitError as exc:
            fallback_reason = str(exc)
            degrade_stage = "rate_limit"
        except asyncio.TimeoutError:
            fallback_reason = "JEPA worker timed out"
            degrade_stage = "timeout"
            open_circuit = True
        except Exception as exc:
            fallback_reason = str(exc)
            degrade_stage = "submit"
            open_circuit = True

        if open_circuit and not self._jepa_safe_fallback_reason:
            await self._activate_jepa_safe_fallback(fallback_reason or "V-JEPA2 worker unavailable")
            self._open_jepa_circuit(fallback_reason or "JEPA worker unavailable")
            return self._degraded_jepa_result(
                session_id=session_id,
                observation_id=observation_id,
                correlation_id=correlation_id,
                fallback_reason=self._jepa_safe_fallback_reason or fallback_reason or "V-JEPA2 worker unavailable",
                degrade_stage=degrade_stage,
            )

        get_logger("runtime").warning(
            "jepa_worker_fallback_degraded",
            session_id=session_id,
            observation_id=observation_id,
            error=fallback_reason,
            degrade_stage=degrade_stage,
            safe_fallback_retry=False,
        )
        return self._degraded_jepa_result(
            session_id=session_id,
            observation_id=observation_id,
            correlation_id=correlation_id,
            fallback_reason=fallback_reason,
            degrade_stage=degrade_stage,
        )

    def _load_frame_array(self, image_path: str) -> np.ndarray:
        with Image.open(image_path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _precompute_perception_for_worker(
        self, frame: np.ndarray
    ) -> tuple[bytes | None, str | None]:
        """
        Run PerceptionPipeline.encode() in the parent process and
        serialize the output for worker subprocess consumption.

        This eliminates DINOv2+SAM torch loading inside the worker,
        reducing worker memory from ~2.7GB to ~2.4GB on 8GB M1 —
        enough headroom to prevent the SIGSEGV that caused the
        persistent Signal 11 crash.

        Returns:
            patch_tokens_bytes: raw float32 bytes of ndarray[196, 384]
                                 or None on failure
            mask_results_json:  JSON string of list[MaskResult.to_dict()]
                                 or None on failure
        """
        try:
            from cloud.perception import PerceptionPipeline
            # Use a lazily-initialized parent-side PerceptionPipeline.
            # This is separate from self.providers to avoid coupling.
            if not hasattr(self, "_worker_perception_pipeline"):
                settings = self.get_settings()
                dinov2_config = settings.providers.get("dinov2")
                device = (
                    dinov2_config.metadata.get("device", "mps")
                    if dinov2_config is not None
                    else "mps"
                )
                self._worker_perception_pipeline = PerceptionPipeline(
                    device=device
                )
            pipeline: PerceptionPipeline = (
                self._worker_perception_pipeline
            )
            patch_tokens, mask_results = pipeline.encode(frame)
            patch_tokens_bytes = np.asarray(
                patch_tokens, dtype=np.float32
            ).tobytes()
            mask_results_json = json.dumps(
                [mask.to_dict() for mask in mask_results]
            )
            return patch_tokens_bytes, mask_results_json
        except Exception as exc:
            get_logger("runtime").warning(
                "worker_perception_precompute_failed",
                error=str(exc),
            )
            return None, None

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
        observations: list[Observation] = []
        scene_states: list[SceneState] = []
        window_label = "latest session window"
        if request.observation_ids:
            observations = self.store.get_observations_by_ids(request.observation_ids)
            window_label = "selected observation window"
        elif request.challenge_set == "curated":
            previous_runs = self.store.recent_challenge_runs(session_id=request.session_id, limit=1)
            if previous_runs:
                latest_run = previous_runs[0]
                observations = [
                    item
                    for item in self.store.get_observations_by_ids(latest_run.observation_ids)
                    if item.session_id == request.session_id
                ]
                scene_states = [
                    state
                    for state in (
                        self.store.get_scene_state(world_state_id)
                        for world_state_id in latest_run.world_state_ids
                    )
                    if state is not None and state.session_id == request.session_id
                ]
                if observations or scene_states:
                    window_label = "stored challenge window"
            if not scene_states:
                scene_states = list(
                    reversed(self.store.recent_scene_states(session_id=request.session_id, limit=request.limit))
                )
                observations = [
                    observation
                    for observation in (
                        self.store.get_observation(state.observation_id)
                        for state in scene_states
                    )
                    if observation is not None
                ]
                window_label = "stored scene window"
        else:
            observations = list(
                reversed(self.store.recent_observations(session_id=request.session_id, limit=request.limit))
            )
            window_label = "live session window" if request.challenge_set == "live" else "latest session window"
        if not scene_states:
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
            if not observations:
                observations = [
                    observation
                    for observation in (
                        self.store.get_observation(state.observation_id)
                        for state in scene_states
                    )
                    if observation is not None
                ]
        challenge = build_challenge_run(
            session_id=request.session_id,
            challenge_set=request.challenge_set,
            proof_mode=request.proof_mode,
            observations=observations,
            scene_states=scene_states,
            window_label=window_label,
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
            tags=[
                *[label for label in (self._preferred_object_label(tag) for tag in request.tags) if label],
                *self._fallback_tags(provider_metadata),
            ],
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
        top_label = self._preferred_object_label(
            metadata.get("top_label"),
            metadata.get("open_vocab_label"),
            metadata.get("primary_object_label"),
        )
        summary = "Local observation"
        if top_label and top_label != "object":
            summary = top_label
        if query:
            summary = f"{summary}; prompt: {query}"
        return summary

    def _fallback_tags(self, metadata: dict) -> list[str]:
        tags = []
        top_label = self._preferred_object_label(
            metadata.get("top_label"),
            metadata.get("open_vocab_label"),
            metadata.get("primary_object_label"),
        )
        if top_label and top_label != "object":
            tags.append(top_label)
        dominant_color = self._clean_label(metadata.get("dominant_color"))
        if dominant_color:
            if dominant_color not in tags:
                tags.append(dominant_color)
            scene_tag = f"{dominant_color} scene"
            if scene_tag not in tags:
                tags.append(scene_tag)
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
            w_diag = np.ones(128, dtype=np.float32)
            for k, v in getattr(self.smriti_db, '_w', {}).items():
                if k.startswith("dim_"):
                    try:
                        idx = int(k[4:])
                        if 0 <= idx < 128:
                            w_diag[idx] = float(v)
                    except ValueError:
                        pass
            self._setu2_bridge = Setu2Bridge(w_diagonal=w_diag)

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
