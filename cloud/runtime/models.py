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
ThemePreference = Literal["system", "dark", "light"]
ProofMode = Literal["jepa", "baseline", "both"]
ChallengeSet = Literal["live", "curated", "both"]
TrackStatus = Literal["visible", "occluded", "re-identified", "disappeared", "violated prediction"]
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
    local_reasoning_disabled: bool = True
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
    temporal_continuity_score: float = 0.0
    persistence_confidence: float = 0.0
    occlusion_recovery_score: float = 0.0
    continuity_signal: ContinuitySignal = Field(default_factory=ContinuitySignal)
    persistence_signal: PersistenceSignal = Field(default_factory=PersistenceSignal)


class PredictionWindow(BaseModel):
    previous_observation_id: Optional[str] = None
    context_observation_ids: list[str] = Field(default_factory=list)
    expected_track_ids: list[str] = Field(default_factory=list)
    predicted_tags: list[str] = Field(default_factory=list)
    predicted_summary: str = ""
    stable_elements: list[str] = Field(default_factory=list)
    confidence: float = 0.0


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


class SceneState(BaseModel):
    id: str
    session_id: str
    created_at: datetime
    observation_id: str
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

    def to_payload(self) -> "JEPATickPayload":
        return JEPATickPayload(
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
        )


class JEPATickPayload(BaseModel):
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


class ObservationsResponse(BaseModel):
    observations: list[Observation]


class EventMessage(BaseModel):
    type: str
    timestamp: datetime = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)
