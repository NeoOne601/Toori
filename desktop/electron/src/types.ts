export type ProviderHealth = {
  name: string;
  role: string;
  enabled: boolean;
  healthy: boolean;
  message: string;
  latency_ms?: number | null;
};

export type ReasoningTraceEntry = {
  provider: string;
  healthy: boolean;
  health_message: string;
  attempted: boolean;
  success: boolean;
  latency_ms?: number | null;
  error?: string | null;
};

export type Observation = {
  id: string;
  session_id: string;
  created_at: string;
  world_state_id?: string | null;
  image_path: string;
  thumbnail_path: string;
  width: number;
  height: number;
  summary?: string | null;
  source_query?: string | null;
  tags: string[];
  novelty: number;
  confidence: number;
  providers: string[];
  metadata?: Record<string, unknown>;
};

export type SearchHit = {
  observation_id: string;
  score: number;
  summary?: string | null;
  thumbnail_path: string;
  session_id: string;
  created_at: string;
  tags: string[];
};

export type Answer = {
  text: string;
  provider: string;
  confidence: number;
};

export type BoundingBox = {
  x: number;
  y: number;
  width: number;
  height: number;
  label?: string | null;
  score?: number | null;
  metadata?: Record<string, unknown> | null;
};

export type ContinuitySignal = {
  stable_elements: string[];
  changed_elements: string[];
  continuity_score: number;
  predicted_support: number;
  nearest_memory_score: number;
};

export type PersistenceSignal = {
  visible_track_ids: string[];
  occluded_track_ids: string[];
  recovered_track_ids: string[];
  disappeared_track_ids: string[];
  violated_track_ids: string[];
  persistence_confidence: number;
};

export type WorldModelMetrics = {
  prediction_consistency: number;
  surprise_score: number;
  temporal_continuity_score: number;
  persistence_confidence: number;
  occlusion_recovery_score: number;
  continuity_signal: ContinuitySignal;
  persistence_signal: PersistenceSignal;
};

export type PredictionWindow = {
  previous_observation_id?: string | null;
  context_observation_ids: string[];
  expected_track_ids: string[];
  predicted_tags: string[];
  predicted_summary: string;
  stable_elements: string[];
  confidence: number;
};

export type EntityTrack = {
  id: string;
  session_id: string;
  label: string;
  status: string;
  first_seen_at: string;
  last_seen_at: string;
  first_observation_id: string;
  last_observation_id: string;
  observations: string[];
  visibility_streak: number;
  occlusion_count: number;
  reidentification_count: number;
  persistence_confidence: number;
  continuity_score: number;
  last_similarity: number;
  status_history: string[];
  metadata?: Record<string, unknown>;
};

export type SceneState = {
  id: string;
  session_id: string;
  created_at: string;
  observation_id: string;
  previous_world_state_id?: string | null;
  nearest_memory_observation_id?: string | null;
  entity_track_ids: string[];
  persisted_track_ids: string[];
  occluded_track_ids: string[];
  observed_elements: string[];
  stable_elements: string[];
  changed_elements: string[];
  predicted_state_summary: string;
  observed_state_summary: string;
  prediction_window: PredictionWindow;
  metrics: WorldModelMetrics;
  metadata?: Record<string, unknown>;
};

export type BaselineModeScore = {
  continuity: number;
  persistence: number;
  surprise_separation: number;
  composite: number;
};

export type BaselineComparison = {
  winner: string;
  jepa_hybrid: BaselineModeScore;
  frame_captioning: BaselineModeScore;
  embedding_retrieval: BaselineModeScore;
  summary: string;
};

export type ChallengeRun = {
  id: string;
  session_id: string;
  created_at: string;
  challenge_set: string;
  proof_mode: string;
  status: string;
  observation_ids: string[];
  world_state_ids: string[];
  guide_steps: string[];
  success_criteria: Record<string, boolean>;
  baseline_comparison: BaselineComparison;
  summary: string;
};

export type AnalyzeResponse = {
  observation: Observation;
  hits: SearchHit[];
  answer?: Answer | null;
  provider_health: ProviderHealth[];
  reasoning_trace: ReasoningTraceEntry[];
};

export type JEPATickPayload = {
  energy_map: number[][];
  entity_tracks: Array<Record<string, unknown>>;
  talker_event?: string | null;
  sigreg_loss: number;
  forecast_errors: Record<string, number>;
  session_fingerprint: number[];
  planning_time_ms: number;
  caption_score: number;
  retrieval_score: number;
  timestamp_ms: number;
  warmup: boolean;
  mask_results: Array<Record<string, unknown>>;
  mean_energy: number;
  energy_std: number;
  guard_active?: boolean;
  ema_tau?: number;
  depth_strata?: Record<string, unknown> | null;
  anchor_matches?: Array<Record<string, unknown>> | null;
  setu_descriptions?: Array<Record<string, unknown>> | null;
  alignment_loss?: number;
};

export type LivingLensTickResponse = AnalyzeResponse & {
  scene_state: SceneState;
  entity_tracks: EntityTrack[];
  baseline_comparison?: BaselineComparison | null;
  jepa_tick?: JEPATickPayload | null;
};

export type QueryResponse = {
  hits: SearchHit[];
  answer?: Answer | null;
  provider_health: ProviderHealth[];
  reasoning_trace: ReasoningTraceEntry[];
};

export type WorldStateResponse = {
  session_id: string;
  current?: SceneState | null;
  history: SceneState[];
  entity_tracks: EntityTrack[];
  challenges: ChallengeRun[];
  atlas?: Record<string, unknown> | null;
};

export type Settings = Record<string, any>;

export type CameraDeviceOption = {
  deviceId: string;
  label: string;
};

export type CameraDiagnostics = {
  phase: string;
  selectedDeviceId: string;
  selectedLabel: string;
  permissionStatus: string;
  resolution: string;
  readyState: string;
  trackState: string;
  trackMuted: boolean;
  trackEnabled: boolean;
  lastFrameAt: string | null;
  frameLuma: number | null;
  blackFrameDetected: boolean;
  message: string;
  error: string | null;
};

export type CameraAccessState = {
  status: string;
  granted: boolean;
  canPrompt: boolean;
};

export type SpatialCanvasGhostBox = {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  label?: string;
  score?: number | null;
  occluded?: boolean;
  depth?: number | null;
};

export type SpatialCanvasAnchor = {
  id: string;
  x: number;
  y: number;
  z?: number | null;
  label?: string;
  tone?: "live" | "stable" | "memory" | "accent";
};

export type BaselineHistoryPoint = {
  id: string;
  label: string;
  winner: string;
  composite: number;
  continuity: number;
  persistence: number;
  surpriseSeparation: number;
  summary: string;
};

export type ConsumerGraphNode = {
  id: string;
  label: string;
  x: number;
  y: number;
  radius?: number;
  tone?: "accent" | "memory" | "live" | "stable";
};

export type ConsumerGraphLink = {
  source: string;
  target: string;
  strength?: number;
};

export type OcclusionTrackView = {
  id: string;
  label: string;
  status: string;
  confidence: number;
  note: string;
};

export type SmritiDepthStrata = {
  depth_proxy?: number[][];
  foreground_mask?: boolean[][];
  midground_mask?: boolean[][];
  background_mask?: boolean[][];
  confidence?: number;
  strata_entropy?: number;
};

export type SmritiAnchorMatch = {
  template_name: string;
  confidence: number;
  patch_indices: number[];
  depth_stratum: string;
  centroid_patch: number;
  embedding_centroid?: number[];
  is_novel?: boolean;
  bbox_normalized?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
};

export type SmritiGateResult = {
  passes: boolean;
  consistency_score: number;
  failure_reasons: string[];
  anchor_name: string;
  depth_stratum: string;
  estimated_hallucination_risk: number;
  uncertainty_map: number[][];
};

export type SmritiSetuDescription = {
  text: string;
  confidence: number;
  anchor_basis: string;
  depth_stratum: string;
  is_uncertain?: boolean;
  hallucination_risk?: number;
  uncertainty_map?: number[][] | null;
};

export type SmritiSetuRecord = {
  gate?: SmritiGateResult;
  description: SmritiSetuDescription;
};

export type SmritiMedia = {
  id: string;
  observation_id: string | null;
  file_path: string;
  file_hash: string;
  media_type: "image" | "video" | "screenshot" | "burst" | "live_photo";
  depth_strata: SmritiDepthStrata | null;
  anchor_matches: SmritiAnchorMatch[];
  setu_descriptions: SmritiSetuRecord[];
  hallucination_risk: number;
  ingestion_status: "pending" | "processing" | "complete" | "failed";
  visual_cluster_id: number | null;
  location_id: string | null;
  original_created_at: string | null;
  ingested_at: string | null;
  alignment_loss: number | null;
  error_message: string | null;
  embedding: number[] | null;
};

export type SmritiRecallResult = {
  media_id: string;
  file_path: string;
  thumbnail_path: string;
  setu_score: number;
  vector_score?: number;
  fts_score?: number;
  hybrid_score: number;
  primary_description: string;
  anchor_basis: string;
  depth_stratum: string;
  hallucination_risk: number;
  created_at: string;
  person_names: string[];
  location_name: string | null;
};

export type SmritiRecallResponse = {
  query: string;
  results: SmritiRecallResult[];
  total_searched: number;
  setu_ms: number;
};

export type SmritiClusterNode = {
  id: number;
  label: string;
  media_count: number;
  centroid: number[];
  dominant_depth_stratum: string | null;
  temporal_span_days: number | null;
};

export type SmritiClusterEdge = {
  source: number;
  target: number;
  similarity: number;
};

export type SmritiMandalaData = {
  nodes: SmritiClusterNode[];
  edges: SmritiClusterEdge[];
  generated_at: string;
};

export type SmritiPersonJournalEntry = {
  media_id: string;
  file_path: string;
  ingested_at: string;
};

export type SmritiPersonJournal = {
  person_name: string;
  entries: SmritiPersonJournalEntry[];
  count: number;
};

export type SmritiWorkerStats = {
  worker_id: number;
  pid: number;
  alive: boolean;
  queue_size: number;
  pending: number;
  submitted: number;
  completed: number;
  sessions: string[];
};

export type SmritiMetrics = {
  workers: SmritiWorkerStats[];
  pending_media: number;
  recent_sessions: string[];
  energy_ema: Record<string, number>;
};

export type SmritiStatus = {
  ingestion: {
    queued: number;
    processed: number;
    failed: number;
    skipped_duplicate: number;
    queue_depth: number;
    queue_utilization: number;
    watched_folders: string[];
  };
  status: string;
};

export type SmritiIngestRequest = {
  folder_path?: string;
  file_path?: string;
};

export type SmritiTagPersonRequest = {
  media_id: string;
  person_name: string;
  confirmed: boolean;
};

export type SmritiRecallRequest = {
  query: string;
  session_id?: string;
  top_k?: number;
  person_filter?: string | null;
  location_filter?: string | null;
  time_start?: string | null;
  time_end?: string | null;
  min_confidence?: number;
};

export type SmritiSection = "mandala" | "recall" | "deepdive" | "journals" | "hud";

export type SmritiState = {
  section: SmritiSection;
  mandalaData: SmritiMandalaData | null;
  recallResults: SmritiRecallResult[];
  recallQuery: string;
  recallBusy: boolean;
  selectedMedia: SmritiRecallResult | null;
  personFilter: string;
  locationFilter: string;
  minConfidence: number;
  timeRangeDays: number;
  personName: string;
  personJournal: SmritiPersonJournal | null;
  metrics: SmritiMetrics | null;
  status: SmritiStatus | null;
  ingestionFolder: string;
  ingestionBusy: boolean;
  ingestionStatus: string;
  totalIndexed: number;
};

export type SmritiStorageConfig = {
  data_dir: string | null;
  frames_dir: string | null;
  thumbs_dir: string | null;
  templates_path: string | null;
  max_storage_gb: number;
  watch_folders: string[];
  store_full_frames: boolean;
  thumbnail_max_dim: number;
  auto_prune_missing: boolean;
};

export type WatchFolderStatus = {
  path: string;
  exists: boolean;
  is_accessible: boolean;
  media_count_total: number;
  media_count_indexed: number;
  media_count_pending: number;
  watchdog_active: boolean;
  last_event_at: string | null;
  error: string | null;
};

export type StorageUsageReport = {
  smriti_data_dir: string;
  total_media_count: number;
  indexed_count: number;
  pending_count: number;
  failed_count: number;
  frames_bytes: number;
  thumbs_bytes: number;
  smriti_db_bytes: number;
  faiss_index_bytes: number;
  templates_bytes: number;
  total_bytes: number;
  total_human: string;
  max_storage_gb: number;
  budget_pct: number;
  budget_warning: boolean;
  budget_critical: boolean;
  watch_folder_stats: WatchFolderStatus[];
};

export type SmritiPruneRequest = {
  older_than_days?: number | null;
  remove_missing_files: boolean;
  remove_failed: boolean;
  clear_all: boolean;
  confirm_clear_all: string;
};

export type SmritiPruneResult = {
  removed_media_records: number;
  removed_bytes: number;
  removed_bytes_human: string;
  errors: string[];
};
