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
