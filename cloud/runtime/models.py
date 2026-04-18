from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


DecodeMode = Literal["off", "auto", "force"]
StorageMode = Literal["embeddings-only", "frames+embeddings"]
AuthMode = Literal["loopback", "api-key", "disabled"]
ThemePreference = Literal[
    "system",
    "dark",
    "light",
    "graphite",
    "sepia",
    "high_contrast_dark",
    "high_contrast_light",
]
ProofMode = Literal["jepa", "baseline", "both"]
ChallengeSet = Literal["live", "curated", "both"]
TrackStatus = Literal["visible", "occluded", "re-identified", "disappeared", "violated prediction"]
ObservationKind = Literal["camera", "tool_state", "memory_query"]
StateDomain = Literal["camera", "browser", "desktop", "memory"]
TalkerEventType = Literal[
    "ENTITY_APPEARED",
    "ENTITY_DISAPPEARED",
    "OCCLUSION_START",
    "OCCLUSION_END",
    "PREDICTION_VIOLATION",
    "SCENE_STABLE",
]


class ProviderConfig(BaseModel):
    name: str
    enabled: bool = True
    base_url: Optional[str] = None
    model: Optional[str] = None
    model_path: Optional[str] = None
    api_key: Optional[str] = None
    timeout_s: float = 10.0
    health_probe_interval_s: int = 30
    priority: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderHealth(BaseModel):
    name: str
    role: Literal["perception", "reasoning", "search"]
    enabled: bool = True
    healthy: bool = False
    message: str = "not configured"
    last_checked_at: datetime = Field(default_factory=utc_now)
    latency_ms: Optional[float] = None


class RuntimeFeatureSettings(BaseModel):
    live_lens_use_jepa_tick: bool = True
    energy_heatmap_enabled: bool = True
    entity_overlay_enabled: bool = True
    open_vocab_labels_enabled: bool = True
    tvlc_enabled: bool = True


class RuntimeFeatureStatus(BaseModel):
    name: str
    enabled: bool = True
    healthy: bool = False
    message: str = "not configured"
    source_provider: Optional[str] = None


class SmritiStorageConfig(BaseModel):
    """
    User-configurable storage settings for Smriti.
    All paths are absolute, expanded, resolved strings.
    These override TOORI_DATA_DIR when set.
    """

    data_dir: Optional[str] = None
    frames_dir: Optional[str] = None
    thumbs_dir: Optional[str] = None
    templates_path: Optional[str] = None
    max_storage_gb: float = 0.0
    watch_folders: list[str] = Field(default_factory=list)
    store_full_frames: bool = True
    thumbnail_max_dim: int = 320
    auto_prune_missing: bool = False

    def resolve_paths(self, base_data_dir: str) -> "SmritiStorageConfig":
        base = Path(base_data_dir).expanduser().resolve()
        smriti_base = (
            Path(self.data_dir).expanduser().resolve()
            if self.data_dir
            else (base / "smriti").resolve()
        )
        return SmritiStorageConfig(
            data_dir=str(smriti_base),
            frames_dir=self.frames_dir or str(smriti_base / "frames"),
            thumbs_dir=self.thumbs_dir or str(smriti_base / "thumbs"),
            templates_path=self.templates_path or str(smriti_base / "sag_templates.json"),
            max_storage_gb=self.max_storage_gb,
            watch_folders=list(self.watch_folders),
            store_full_frames=self.store_full_frames,
            thumbnail_max_dim=self.thumbnail_max_dim,
            auto_prune_missing=self.auto_prune_missing,
        )


class StorageUsageReport(BaseModel):
    """Disk usage report for the Smriti data directory."""

    smriti_data_dir: str
    total_media_count: int
    indexed_count: int
    pending_count: int
    failed_count: int
    frames_bytes: int = 0
    thumbs_bytes: int = 0
    smriti_db_bytes: int = 0
    faiss_index_bytes: int = 0
    templates_bytes: int = 0
    total_bytes: int = 0
    total_human: str = "0 B"
    max_storage_gb: float = 0.0
    budget_pct: float = 0.0
    budget_warning: bool = False
    budget_critical: bool = False
    watch_folder_stats: list[dict[str, Any]] = Field(default_factory=list)


class WatchFolderStatus(BaseModel):
    """Status of a single watched folder."""

    path: str
    exists: bool
    is_accessible: bool
    media_count_total: int
    media_count_indexed: int
    media_count_pending: int
    watchdog_active: bool
    last_event_at: Optional[datetime] = None
    error: Optional[str] = None


class SmritiPruneRequest(BaseModel):
    """Request to prune Smriti storage."""

    older_than_days: Optional[int] = None
    remove_missing_files: bool = False
    remove_failed: bool = False
    clear_all: bool = False
    confirm_clear_all: str = ""


class SmritiPruneResult(BaseModel):
    removed_media_records: int = 0
    removed_bytes: int = 0
    removed_bytes_human: str = "0 B"
    errors: list[str] = Field(default_factory=list)


class SmritiMigrationRequest(BaseModel):
    """Request to migrate Smriti data to a new directory."""

    target_data_dir: str
    target_frames_dir: Optional[str] = None
    target_thumbs_dir: Optional[str] = None
    target_templates_path: Optional[str] = None
    dry_run: bool = False


class SmritiMigrationProgress(BaseModel):
    """Real-time migration progress."""

    status: str
    files_moved: int = 0
    files_total: int = 0
    bytes_moved: int = 0
    bytes_total: int = 0
    bytes_moved_human: str = "0 B"
    bytes_total_human: str = "0 B"
    current_file: Optional[str] = None
    error: Optional[str] = None
    dry_run: bool = False


class SmritiMigrationResult(BaseModel):
    """Final result of a completed migration."""

    success: bool
    dry_run: bool
    files_moved: int
    bytes_moved: int
    bytes_moved_human: str
    new_data_dir: str
    errors: list[str] = Field(default_factory=list)
    rollback_available: bool = False


class RuntimeSettings(BaseModel):
    runtime_profile: str = "hybrid"
    camera_device: str = "default"
    theme_preference: ThemePreference = "system"
    sampling_fps: float = 1.0
    primary_perception_provider: str = "dinov2"
    reasoning_backend: str = "cloud"
    search_provider: str = "local"
    local_reasoning_disabled: bool = False # Gemma 4 is the primary local provider
    fallback_order: list[str] = Field(default_factory=lambda: ["dinov2", "onnx", "basic", "cloud"])
    decode_auto_threshold: float = 0.32
    top_k: int = 6
    retention_days: int = 30
    storage_mode: StorageMode = "frames+embeddings"
    auth_mode: AuthMode = "loopback"
    plugin_allowlist: list[str] = Field(default_factory=list)
    observability_enabled: bool = True
    sync_enabled: bool = False
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    live_features: RuntimeFeatureSettings = Field(default_factory=RuntimeFeatureSettings)
    public_url: str = "https://github.com/NeoOne601/Toori"
    vjepa2_model_path: str = ""
    vjepa2_cache_dir: str = ""
    vjepa2_n_frames: int = 0
    smriti_storage: SmritiStorageConfig = Field(default_factory=SmritiStorageConfig)


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    label: Optional[str] = None
    score: Optional[float] = None


class Answer(BaseModel):
    text: str
    provider: str
    confidence: float = 0.0
    created_at: datetime = Field(default_factory=utc_now)


class ReasoningTraceEntry(BaseModel):
    provider: str
    healthy: bool = False
    health_message: str = "not checked"
    attempted: bool = False
    success: bool = False
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class Observation(BaseModel):
    id: str
    session_id: str
    created_at: datetime
    world_state_id: Optional[str] = None
    observation_kind: ObservationKind = "camera"
    image_path: str
    thumbnail_path: str
    width: int
    height: int
    embedding: list[float]
    summary: Optional[str] = None
    source_query: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    novelty: float = 0.0
    providers: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObservationSummary(BaseModel):
    id: str
    session_id: str
    created_at: datetime
    world_state_id: Optional[str] = None
    observation_kind: ObservationKind = "camera"
    image_path: str
    thumbnail_path: str
    width: int
    height: int
    summary: Optional[str] = None
    source_query: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    novelty: float = 0.0
    providers: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    observation_id: str
    score: float
    summary: Optional[str] = None
    thumbnail_path: str
    session_id: str
    created_at: datetime
    tags: list[str] = Field(default_factory=list)


class ContinuitySignal(BaseModel):
    stable_elements: list[str] = Field(default_factory=list)
    changed_elements: list[str] = Field(default_factory=list)
    continuity_score: float = 0.0
    predicted_support: float = 0.0
    nearest_memory_score: float = 0.0


class PersistenceSignal(BaseModel):
    visible_track_ids: list[str] = Field(default_factory=list)
    occluded_track_ids: list[str] = Field(default_factory=list)
    recovered_track_ids: list[str] = Field(default_factory=list)
    disappeared_track_ids: list[str] = Field(default_factory=list)
    violated_track_ids: list[str] = Field(default_factory=list)
    persistence_confidence: float = 0.0


class WorldModelMetrics(BaseModel):
    prediction_consistency: float = 0.0
    surprise_score: float = 0.0
    energy_activation_score: float = 0.0
    temporal_continuity_score: float = 0.0
    persistence_confidence: float = 0.0
    occlusion_recovery_score: float = 0.0
    continuity_signal: ContinuitySignal = Field(default_factory=ContinuitySignal)
    persistence_signal: PersistenceSignal = Field(default_factory=PersistenceSignal)


class ActionToken(BaseModel):
    id: str
    verb: str
    target_kind: str = ""
    target_id: Optional[str] = None
    target_label: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class GroundedEntity(BaseModel):
    id: str
    label: str
    kind: str
    state_domain: StateDomain = "camera"
    status: str = "visible"
    confidence: float = 0.0
    source_track_id: Optional[str] = None
    properties: dict[str, Any] = Field(default_factory=dict)


class GroundedAffordance(BaseModel):
    id: str
    label: str
    kind: str
    state_domain: StateDomain = "camera"
    target_entity_id: Optional[str] = None
    availability: Literal["available", "hidden", "disabled", "missing", "error"] = "available"
    confidence: float = 0.0
    properties: dict[str, Any] = Field(default_factory=dict)


class PredictedAffordanceState(BaseModel):
    affordance_id: str
    label: str
    availability: Literal["available", "hidden", "disabled", "missing", "error"] = "available"
    reason: str = ""


class RolloutStep(BaseModel):
    step_index: int = 0
    action: ActionToken
    predicted_state_domain: StateDomain = "camera"
    predicted_summary: str = ""
    blockers: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class RolloutBranch(BaseModel):
    id: str
    candidate_action: ActionToken
    predicted_next_state_summary: str = ""
    predicted_persistent_entities: list[str] = Field(default_factory=list)
    predicted_affordances: list[PredictedAffordanceState] = Field(default_factory=list)
    risk_score: float = 0.0
    confidence: float = 0.0
    expected_recovery_cost: float = 0.0
    failure_predicates: list[str] = Field(default_factory=list)
    steps: list[RolloutStep] = Field(default_factory=list)


class RolloutComparison(BaseModel):
    state_domain: StateDomain = "camera"
    based_on_world_state_id: Optional[str] = None
    horizon: int = 1
    ranked_branches: list[RolloutBranch] = Field(default_factory=list)
    chosen_branch_id: Optional[str] = None
    summary: str = ""


class PredictionWindow(BaseModel):
    previous_observation_id: Optional[str] = None
    context_observation_ids: list[str] = Field(default_factory=list)
    expected_track_ids: list[str] = Field(default_factory=list)
    predicted_tags: list[str] = Field(default_factory=list)
    predicted_summary: str = ""
    stable_elements: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    candidate_actions: list[ActionToken] = Field(default_factory=list)
    predicted_branches: list[RolloutBranch] = Field(default_factory=list)
    chosen_branch_id: Optional[str] = None


class EntityTrack(BaseModel):
    id: str
    session_id: str
    label: str
    status: TrackStatus = "visible"
    first_seen_at: datetime
    last_seen_at: datetime
    first_observation_id: str
    last_observation_id: str
    observations: list[str] = Field(default_factory=list)
    visibility_streak: int = 0
    occlusion_count: int = 0
    reidentification_count: int = 0
    persistence_confidence: float = 0.0
    continuity_score: float = 0.0
    last_similarity: float = 0.0
    prototype_embedding: list[float] = Field(default_factory=list)
    status_history: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntityLabelCorrectionRequest(BaseModel):
    label: str
    confirmed: bool = True


class SceneState(BaseModel):
    id: str
    session_id: str
    created_at: datetime
    observation_id: str
    state_domain: StateDomain = "camera"
    previous_world_state_id: Optional[str] = None
    nearest_memory_observation_id: Optional[str] = None
    primary_object_label: Optional[str] = None
    proposal_boxes: list[BoundingBox] = Field(default_factory=list)
    entity_track_ids: list[str] = Field(default_factory=list)
    persisted_track_ids: list[str] = Field(default_factory=list)
    occluded_track_ids: list[str] = Field(default_factory=list)
    observed_elements: list[str] = Field(default_factory=list)
    stable_elements: list[str] = Field(default_factory=list)
    changed_elements: list[str] = Field(default_factory=list)
    predicted_state_summary: str = ""
    observed_state_summary: str = ""
    prediction_window: PredictionWindow = Field(default_factory=PredictionWindow)
    grounded_entities: list[GroundedEntity] = Field(default_factory=list)
    affordances: list[GroundedAffordance] = Field(default_factory=list)
    conditioned_rollouts: Optional[RolloutComparison] = None
    metrics: WorldModelMetrics = Field(default_factory=WorldModelMetrics)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaselineModeScore(BaseModel):
    continuity: float = 0.0
    persistence: float = 0.0
    surprise_separation: float = 0.0
    composite: float = 0.0


class BaselineComparison(BaseModel):
    winner: str = "jepa_hybrid"
    jepa_hybrid: BaselineModeScore = Field(default_factory=BaselineModeScore)
    frame_captioning: BaselineModeScore = Field(default_factory=BaselineModeScore)
    embedding_retrieval: BaselineModeScore = Field(default_factory=BaselineModeScore)
    summary: str = ""


class ChallengeRun(BaseModel):
    id: str
    session_id: str
    created_at: datetime
    challenge_set: ChallengeSet = "both"
    proof_mode: ProofMode = "both"
    status: Literal["pending", "completed"] = "completed"
    observation_ids: list[str] = Field(default_factory=list)
    world_state_ids: list[str] = Field(default_factory=list)
    guide_steps: list[str] = Field(default_factory=list)
    success_criteria: dict[str, bool] = Field(default_factory=dict)
    baseline_comparison: BaselineComparison = Field(default_factory=BaselineComparison)
    window_label: str = ""
    summary: str = ""
    narration: str = ""


class RecoveryScenario(BaseModel):
    id: str
    title: str
    domain: str = "hybrid"
    description: str = ""
    passed: bool = False
    score: float = 0.0
    details: str = ""
    related_branch_id: Optional[str] = None


class RecoveryBenchmarkRun(BaseModel):
    id: str
    session_id: str
    created_at: datetime = Field(default_factory=utc_now)
    benchmark_scope: str = "hybrid"
    world_state_ids: list[str] = Field(default_factory=list)
    scenarios: list[RecoveryScenario] = Field(default_factory=list)
    winner: str = "action_conditioned_rollout"
    summary: str = ""


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    file_path: Optional[str] = None
    session_id: str = "default"
    query: Optional[str] = None
    decode_mode: DecodeMode = "auto"
    top_k: Optional[int] = None
    time_window_s: Optional[int] = None
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_input(self) -> "AnalyzeRequest":
        if not self.image_base64 and not self.file_path:
            raise ValueError("Either image_base64 or file_path must be provided")
        return self


class AnalyzeResponse(BaseModel):
    observation: Observation
    hits: list[SearchHit] = Field(default_factory=list)
    answer: Optional[Answer] = None
    provider_health: list[ProviderHealth] = Field(default_factory=list)
    reasoning_trace: list[ReasoningTraceEntry] = Field(default_factory=list)


class LivingLensTickRequest(AnalyzeRequest):
    proof_mode: ProofMode = "both"


class ToolStateObserveRequest(BaseModel):
    session_id: str = "default"
    state_domain: StateDomain = "browser"
    current_url: Optional[str] = None
    view_id: Optional[str] = None
    screenshot_base64: Optional[str] = None
    file_path: Optional[str] = None
    visible_entities: list[GroundedEntity] = Field(default_factory=list)
    affordances: list[GroundedAffordance] = Field(default_factory=list)
    focused_target: Optional[str] = None
    error_banners: list[str] = Field(default_factory=list)
    triggering_action: Optional[ActionToken] = None
    top_k: Optional[int] = None

    @model_validator(mode="after")
    def validate_tool_state(self) -> "ToolStateObserveRequest":
        if not (
            self.current_url
            or self.view_id
            or self.visible_entities
            or self.affordances
            or self.error_banners
        ):
            raise ValueError(
                "Provide current_url, view_id, visible_entities, affordances, or error_banners"
            )
        return self


class ToolStateObserveResponse(AnalyzeResponse):
    scene_state: SceneState
    entity_tracks: list[EntityTrack] = Field(default_factory=list)


class PlanningRolloutRequest(BaseModel):
    session_id: str = "default"
    state_domain: Optional[StateDomain] = None
    current_state_id: Optional[str] = None
    candidate_actions: list[ActionToken] = Field(default_factory=list)
    horizon: int = Field(default=2, ge=1, le=5)


class PlanningRolloutResponse(BaseModel):
    scene_state: SceneState
    comparison: RolloutComparison


class RecoveryBenchmarkRunRequest(BaseModel):
    session_id: str = "default"
    current_state_id: Optional[str] = None
    horizon: int = Field(default=2, ge=1, le=5)


class QueryRequest(BaseModel):
    query: Optional[str] = None
    image_base64: Optional[str] = None
    file_path: Optional[str] = None
    session_id: Optional[str] = None
    top_k: int = 6
    time_window_s: Optional[int] = None

    @model_validator(mode="after")
    def validate_query(self) -> "QueryRequest":
        if not self.query and not self.image_base64 and not self.file_path:
            raise ValueError("Provide query text, image_base64, or file_path")
        return self


class QueryResponse(BaseModel):
    hits: list[SearchHit] = Field(default_factory=list)
    answer: Optional[Answer] = None
    provider_health: list[ProviderHealth] = Field(default_factory=list)
    reasoning_trace: list[ReasoningTraceEntry] = Field(default_factory=list)


class TalkerEvent(BaseModel):
    event_type: TalkerEventType
    confidence: float = 0.0
    entity_ids: list[str] = Field(default_factory=list)
    energy_summary: float = 0.0
    description: str = ""


@dataclass
class JEPATick:
    energy_map: np.ndarray
    entity_tracks: list["EntityTrack"]
    talker_event: Optional[str]
    sigreg_loss: float
    forecast_errors: dict[int, float]
    session_fingerprint: np.ndarray
    planning_time_ms: float
    caption_score: float
    retrieval_score: float
    timestamp_ms: int
    warmup: bool
    mask_results: list[dict[str, Any]]
    mean_energy: float
    energy_std: float
    guard_active: bool = False
    ema_tau: float = 0.996
    depth_strata: Optional[dict] = None
    anchor_matches: Optional[list[dict]] = None
    setu_descriptions: Optional[list[dict]] = None
    alignment_loss: float = 0.0
    # Sprint 6 — World Model Foundation
    l2_embedding: Optional[list[float]] = None
    predicted_next_embedding: Optional[list[float]] = None
    prediction_error: Optional[float] = None
    epistemic_uncertainty: Optional[float] = None
    aleatoric_uncertainty: Optional[float] = None
    surprise_score: Optional[float] = None
    audio_embedding: Optional[list[float]] = None
    audio_energy: Optional[float] = None
    world_model_version: str = "surrogate"
    configured_encoder: str = "vjepa2"
    last_tick_encoder_type: str = "surrogate"
    degraded: bool = False
    degrade_reason: Optional[str] = None
    degrade_stage: Optional[str] = None
    gemma4_alert: Optional[dict[str, Any]] = None

    def to_payload(self) -> "JEPATickPayload":
        return JEPATickPayload(
            tick_id=f"tick_{int(self.timestamp_ms)}",
            overlay_epoch=int(self.timestamp_ms),
            source_backend=self.last_tick_encoder_type or self.world_model_version,
            energy_map=np.asarray(self.energy_map, dtype=np.float32).tolist(),
            entity_tracks=[track.model_dump(mode="json") for track in self.entity_tracks],
            talker_event=self.talker_event,
            sigreg_loss=float(self.sigreg_loss),
            forecast_errors={int(key): float(value) for key, value in self.forecast_errors.items()},
            session_fingerprint=np.asarray(self.session_fingerprint, dtype=np.float32).tolist(),
            planning_time_ms=float(self.planning_time_ms),
            caption_score=float(self.caption_score),
            retrieval_score=float(self.retrieval_score),
            timestamp_ms=int(self.timestamp_ms),
            warmup=bool(self.warmup),
            mask_results=self.mask_results,
            mean_energy=float(self.mean_energy),
            energy_std=float(self.energy_std),
            guard_active=bool(self.guard_active),
            ema_tau=float(self.ema_tau),
            depth_strata=self.depth_strata,
            anchor_matches=self.anchor_matches,
            setu_descriptions=self.setu_descriptions,
            alignment_loss=float(self.alignment_loss),
            l2_embedding=self.l2_embedding,
            predicted_next_embedding=self.predicted_next_embedding,
            prediction_error=float(self.prediction_error) if self.prediction_error is not None else None,
            epistemic_uncertainty=float(self.epistemic_uncertainty) if self.epistemic_uncertainty is not None else None,
            aleatoric_uncertainty=float(self.aleatoric_uncertainty) if self.aleatoric_uncertainty is not None else None,
            surprise_score=float(self.surprise_score) if self.surprise_score is not None else None,
            prediction_error_z=float(self.surprise_score) if self.surprise_score is not None else None,
            audio_embedding=self.audio_embedding,
            audio_energy=float(self.audio_energy) if self.audio_energy is not None else None,
            world_model_version=self.world_model_version,
            configured_encoder=self.configured_encoder,
            last_tick_encoder_type=self.last_tick_encoder_type,
            degraded=self.degraded,
            degrade_reason=self.degrade_reason,
            degrade_stage=self.degrade_stage,
            gemma4_alert=self.gemma4_alert,
        )


class JEPATickPayload(BaseModel):
    tick_id: Optional[str] = None
    overlay_epoch: Optional[int] = None
    source_backend: Optional[str] = None
    energy_map: list[list[float]]
    entity_tracks: list[dict[str, Any]] = Field(default_factory=list)
    talker_event: Optional[str] = None
    sigreg_loss: float = 0.0
    forecast_errors: dict[int, float] = Field(default_factory=dict)
    session_fingerprint: list[float] = Field(default_factory=list)
    planning_time_ms: float = 0.0
    caption_score: float = 0.0
    retrieval_score: float = 0.0
    timestamp_ms: int
    warmup: bool = True
    mask_results: list[dict[str, Any]] = Field(default_factory=list)
    mean_energy: float = 0.0
    energy_std: float = 0.0
    guard_active: bool = False
    ema_tau: float = 0.996
    depth_strata: Optional[dict] = None
    anchor_matches: Optional[list[dict]] = None
    setu_descriptions: Optional[list[dict]] = None
    alignment_loss: float = 0.0
    # Sprint 6 — World Model Foundation
    l2_embedding: Optional[list[float]] = None
    predicted_next_embedding: Optional[list[float]] = None
    prediction_error: Optional[float] = None
    epistemic_uncertainty: Optional[float] = None
    aleatoric_uncertainty: Optional[float] = None
    surprise_score: Optional[float] = None
    prediction_error_z: Optional[float] = None
    audio_embedding: Optional[list[float]] = None
    audio_energy: Optional[float] = None
    world_model_version: str = "surrogate"
    configured_encoder: str = "vjepa2"
    last_tick_encoder_type: str = "surrogate"
    degraded: bool = False
    degrade_reason: Optional[str] = None
    degrade_stage: Optional[str] = None
    gemma4_alert: Optional[dict[str, Any]] = None


class AtlasNode(BaseModel):
    entity_id: str
    label: str
    centroid: tuple[float, float] = (0.5, 0.5)
    track_length: int = 0
    confidence: float = 0.0
    last_energy: float = 0.0
    status: TrackStatus = "visible"
    first_seen_at: datetime = Field(default_factory=utc_now)
    last_seen_at: datetime = Field(default_factory=utc_now)


class AtlasEdge(BaseModel):
    source_id: str
    target_id: str
    interaction_energy: float = 0.0
    spatial_proximity: float = 0.0
    co_occurrence_count: int = 0
    last_seen_together: datetime = Field(default_factory=utc_now)
    status: Literal["active", "stale", "broken"] = "active"


class SmritiIngestRequest(BaseModel):
    folder_path: Optional[str] = None
    file_path: Optional[str] = None


class SmritiIngestResponse(BaseModel):
    queued: int
    status: str


class SmritiRecallRequest(BaseModel):
    query: str
    session_id: str = "default"
    top_k: int = Field(default=20, ge=1, le=100)
    person_filter: Optional[str] = None
    location_filter: Optional[str] = None
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SmritiRecallItem(BaseModel):
    media_id: str
    file_path: str
    thumbnail_path: str
    setu_score: float
    hybrid_score: float
    primary_description: str
    anchor_basis: str
    depth_stratum: str
    hallucination_risk: float
    created_at: datetime
    person_names: list[str] = Field(default_factory=list)
    location_name: Optional[str] = None
    depth_strata_data: Optional[dict[str, Any]] = None
    anchor_matches: list[dict[str, Any]] = Field(default_factory=list)
    setu_descriptions: list[dict[str, Any]] = Field(default_factory=list)


class SmritiRecallResponse(BaseModel):
    query: str
    results: list[SmritiRecallItem]
    total_searched: int
    setu_ms: float


class SmritiRecallFeedback(BaseModel):
    """User feedback on a recall result."""

    query: str
    media_id: str
    confirmed: bool
    session_id: str = "default"


class SmritiRecallFeedbackResult(BaseModel):
    updated: bool
    w_mean: float
    message: str


class SmritiRecallFeedback(BaseModel):
    """User feedback on a recall result for Setu-2 personalization."""

    query: str
    media_id: str
    confirmed: bool
    session_id: str = "default"


class SmritiRecallFeedbackResult(BaseModel):
    updated: bool
    w_mean: float
    message: str


class AudioQueryRequest(BaseModel):
    """Request body for POST /v1/audio/query."""
    audio_base64: str = Field(description="Base64-encoded PCM float32 bytes or WAV file bytes")
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    top_k: int = Field(default=10, ge=1, le=50)
    session_id: Optional[str] = None
    depth_stratum: Optional[str] = Field(default=None, description="foreground|midground|background")
    person_filter: Optional[str] = None
    confidence_min: float = Field(default=0.0, ge=0.0, le=1.0)
    cross_modal: bool = Field(
        default=False,
        description="If True: project audio into visual embedding space (CLAP) and search visual memories. Enables 'hum to find video frame'. Requires CLAPProjector weights."
    )


class AudioQueryResult(BaseModel):
    """Single recall hit from audio-modal search."""
    media_id: str
    audio_score: float = Field(description="Cosine similarity in audio embedding space, 0.0-1.0")
    rank: int = Field(ge=1)
    thumbnail_path: Optional[str] = None
    setu_descriptions: list[dict] = Field(default_factory=list)
    audio_energy: Optional[float] = None
    audio_duration_seconds: Optional[float] = None
    gemma4_narration: Optional[str] = Field(default=None, description="Gemma4 description if available")


class AudioQueryResponse(BaseModel):
    """Response from POST /v1/audio/query."""
    results: list[AudioQueryResult]
    query_audio_energy: float
    index_size: int = Field(description="Total audio embeddings in FAISS sub-index")
    latency_ms: float
    encoder: str = Field(default="audio_jepa_phase1")


class SmritiTagPersonRequest(BaseModel):
    media_id: str
    person_name: str
    confirmed: bool = True


class SmritiTagPersonResponse(BaseModel):
    person_id: str
    propagated_to: int


class LivingLensTickResponse(AnalyzeResponse):
    scene_state: SceneState
    entity_tracks: list[EntityTrack] = Field(default_factory=list)
    baseline_comparison: Optional[BaselineComparison] = None
    talker_event: Optional[TalkerEvent] = None
    jepa_tick: Optional[JEPATickPayload] = None


class WorldStateResponse(BaseModel):
    session_id: str
    current: Optional[SceneState] = None
    history: list[SceneState] = Field(default_factory=list)
    entity_tracks: list[EntityTrack] = Field(default_factory=list)
    challenges: list[ChallengeRun] = Field(default_factory=list)
    benchmarks: list[RecoveryBenchmarkRun] = Field(default_factory=list)
    atlas: Optional[dict[str, Any]] = None


class ChallengeEvaluateRequest(BaseModel):
    session_id: str
    observation_ids: list[str] = Field(default_factory=list)
    challenge_set: ChallengeSet = "both"
    proof_mode: ProofMode = "both"
    limit: int = Field(default=8, ge=2, le=64)


class JEPAForecastRequest(BaseModel):
    session_id: str = "default"
    k: int = Field(default=1, ge=1, le=32)


class JEPAForecastResponse(BaseModel):
    session_id: str
    k: int
    prediction: list[float] = Field(default_factory=list)
    forecast_error: Optional[float] = None
    ready: bool = False


class ProofReportGenerateRequest(BaseModel):
    session_id: str
    chart_b64: Optional[str] = None


class ProofReportResponse(BaseModel):
    session_id: str
    path: str
    generated: bool = True


class WorldModelStatus(BaseModel):
    encoder_type: str = "surrogate"
    model_id: str = ""
    model_loaded: bool = False
    device: str = "cpu"
    n_frames: int = 0
    test_mode: bool = False
    total_ticks: int = 0
    mean_prediction_error: Optional[float] = None
    mean_surprise_score: Optional[float] = None
    configured_encoder: str = "vjepa2"
    last_tick_encoder_type: str = "surrogate"
    degraded: bool = False
    degrade_reason: Optional[str] = None
    degrade_stage: Optional[str] = None
    active_backend: str = "surrogate"
    native_ready: bool = False
    preflight_status: str = "not_run"
    last_failure_at: Optional[datetime] = None
    crash_fingerprint: Optional[str] = None
    native_process_state: str = "idle"
    last_native_exit_code: Optional[int] = None
    last_native_signal: Optional[int] = None
    retryable_native_failure: bool = False
    telescope_test: str = "PASSED"


class WorldModelConfig(BaseModel):
    model_path: str = ""
    n_frames: int = 0
    effective_model: str = ""
    cache_dir: str = ""
    download_url: str = "https://huggingface.co/facebook/vjepa2-vitl-fpc64-256"


class WorldModelConfigUpdate(BaseModel):
    model_path: str = ""
    cache_dir: str = ""
    n_frames: int = 0


ShareObservationEventType = Literal["share_clicked", "share_copied"]


class ShareObservationRequest(BaseModel):
    session_id: str
    observation_id: Optional[str] = None


class ShareObservationResponse(BaseModel):
    session_id: str
    observation_id: str
    title: str
    summary: str
    share_text: str
    share_url: str
    tracked_entities: int = 0
    persistence_confidence: Optional[float] = None
    memory_match_score: Optional[float] = None


class ShareObservationEventRequest(BaseModel):
    session_id: str
    observation_id: str
    event_type: ShareObservationEventType


class ProviderHealthResponse(BaseModel):
    providers: list[ProviderHealth]
    features: list[RuntimeFeatureStatus] = Field(default_factory=list)


class ObservationsResponse(BaseModel):
    observations: list[Observation]


class ObservationSummariesResponse(BaseModel):
    observations: list[ObservationSummary]


class SceneGraphNode(BaseModel):
    id: str
    label: str
    depth_stratum: str = "unknown"
    depth_confidence: float = 0.0
    bbox: Optional[BoundingBox] = None
    source: str = "unknown"
    confidence: float = 0.0
    status: str = "visible"
    label_source: str = "unknown"
    label_evidence: Optional[dict[str, Any]] = None


class SceneGraphEdge(BaseModel):
    source: str
    target: str
    relation: str = "related"
    weight: float = 0.0


class SceneGraphPayload(BaseModel):
    nodes: list[SceneGraphNode] = Field(default_factory=list)
    edges: list[SceneGraphEdge] = Field(default_factory=list)


class RuntimeSnapshotResponse(BaseModel):
    session_id: str
    current: Optional[SceneState] = None
    entity_tracks: list[EntityTrack] = Field(default_factory=list)
    latest_jepa_tick: Optional[JEPATickPayload] = None
    world_model_status: WorldModelStatus
    observations: list[ObservationSummary] = Field(default_factory=list)
    observation_count: int = 0
    scene_graph: SceneGraphPayload = Field(default_factory=SceneGraphPayload)


class EventMessage(BaseModel):
    type: str
    timestamp: datetime = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)
