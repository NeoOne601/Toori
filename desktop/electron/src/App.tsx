import {
  FormEvent,
  Suspense,
  lazy,
  startTransition,
  useDeferredValue,
  useEffect,
  useRef,
  useState,
} from "react";
// @ts-ignore - react-grid-layout v2 removed WidthProvider; ResponsiveGridLayout is the composed replacement
import { ResponsiveGridLayout as _ResponsiveGridLayout } from "react-grid-layout";

const Heatmap3D = lazy(() => import("./Heatmap3D"));

// v2 API: ResponsiveGridLayout already wraps width detection internally.
// Cast to any to avoid type mismatches from incomplete @types/react-grid-layout v1 stubs.
const ResponsiveGridLayout = _ResponsiveGridLayout as any;

type ProviderHealth = {
  name: string;
  role: string;
  enabled: boolean;
  healthy: boolean;
  message: string;
  latency_ms?: number | null;
};

type ReasoningTraceEntry = {
  provider: string;
  healthy: boolean;
  health_message: string;
  attempted: boolean;
  success: boolean;
  latency_ms?: number | null;
  error?: string | null;
};

type Observation = {
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

type SearchHit = {
  observation_id: string;
  score: number;
  summary?: string | null;
  thumbnail_path: string;
  session_id: string;
  created_at: string;
  tags: string[];
};

type Answer = {
  text: string;
  provider: string;
  confidence: number;
};

type BoundingBox = {
  x: number;
  y: number;
  width: number;
  height: number;
  label?: string | null;
  score?: number | null;
};

type ContinuitySignal = {
  stable_elements: string[];
  changed_elements: string[];
  continuity_score: number;
  predicted_support: number;
  nearest_memory_score: number;
};

type PersistenceSignal = {
  visible_track_ids: string[];
  occluded_track_ids: string[];
  recovered_track_ids: string[];
  disappeared_track_ids: string[];
  violated_track_ids: string[];
  persistence_confidence: number;
};

type WorldModelMetrics = {
  prediction_consistency: number;
  surprise_score: number;
  temporal_continuity_score: number;
  persistence_confidence: number;
  occlusion_recovery_score: number;
  continuity_signal: ContinuitySignal;
  persistence_signal: PersistenceSignal;
};

type PredictionWindow = {
  previous_observation_id?: string | null;
  context_observation_ids: string[];
  expected_track_ids: string[];
  predicted_tags: string[];
  predicted_summary: string;
  stable_elements: string[];
  confidence: number;
};

type EntityTrack = {
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

type SceneState = {
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

type BaselineModeScore = {
  continuity: number;
  persistence: number;
  surprise_separation: number;
  composite: number;
};

type BaselineComparison = {
  winner: string;
  jepa_hybrid: BaselineModeScore;
  frame_captioning: BaselineModeScore;
  embedding_retrieval: BaselineModeScore;
  summary: string;
};

type ChallengeRun = {
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

type AnalyzeResponse = {
  observation: Observation;
  hits: SearchHit[];
  answer?: Answer | null;
  provider_health: ProviderHealth[];
  reasoning_trace: ReasoningTraceEntry[];
};

type LivingLensTickResponse = AnalyzeResponse & {
  scene_state: SceneState;
  entity_tracks: EntityTrack[];
  baseline_comparison?: BaselineComparison | null;
};

type QueryResponse = {
  hits: SearchHit[];
  answer?: Answer | null;
  provider_health: ProviderHealth[];
  reasoning_trace: ReasoningTraceEntry[];
};

type WorldStateResponse = {
  session_id: string;
  current?: SceneState | null;
  history: SceneState[];
  entity_tracks: EntityTrack[];
  challenges: ChallengeRun[];
};

type Settings = Record<string, any>;

type CameraDeviceOption = {
  deviceId: string;
  label: string;
};

type CameraDiagnostics = {
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

type CameraAccessState = {
  status: string;
  granted: boolean;
  canPrompt: boolean;
};

const tabs = [
  "Living Lens",
  "Live Lens",
  "Memory Search",
  "Session Replay",
  "Integrations",
  "Settings",
] as const;

const DEFAULT_CHALLENGE_STEPS = [
  "Hold one stable object in view for a few seconds so the model can learn a baseline scene.",
  "Partially cover that object while keeping the rest of the scene stable.",
  "Fully hide the object for at least one update cycle.",
  "Reveal the object again in roughly the same area.",
  "Move the camera away and return to the original scene.",
  "Introduce a distractor or new object and watch the surprise score react.",
] as const;

const DEFAULT_CAMERA_DIAGNOSTICS: CameraDiagnostics = {
  phase: "idle",
  selectedDeviceId: "default",
  selectedLabel: "Auto camera",
  permissionStatus: "unknown",
  resolution: "0 x 0",
  readyState: "idle",
  trackState: "idle",
  trackMuted: false,
  trackEnabled: false,
  lastFrameAt: null,
  frameLuma: null,
  blackFrameDetected: false,
  message: "Camera not started",
  error: null,
};

type DesktopBridge = {
  request: (
    path: string,
    options?: {
      method?: string;
      headers?: Record<string, string>;
      body?: unknown;
    },
  ) => Promise<{ ok: boolean; status: number; data: any }>;
  pickFile?: () => Promise<string | null>;
  getCameraAccess?: () => Promise<{ status: string; granted: boolean; canPrompt: boolean }>;
  requestCameraAccess?: () => Promise<{ status: string; granted: boolean; canPrompt: boolean }>;
  openCameraSettings?: () => Promise<boolean>;
};

const BROWSER_RUNTIME_URL = "http://127.0.0.1:7777";

function browserBridge(): DesktopBridge {
  return {
    async request(path, options = {}) {
      const requestPath = path.startsWith("/") ? path : `/${path}`;
      const response = await fetch(`${BROWSER_RUNTIME_URL}${requestPath}`, {
        method: options.method || "GET",
        headers: {
          "Content-Type": "application/json",
          ...(options.headers || {}),
        },
        body: options.body ? JSON.stringify(options.body) : undefined,
      });
      const contentType = response.headers.get("content-type") || "";
      const data = contentType.includes("application/json")
        ? await response.json()
        : await response.text();
      return {
        ok: response.ok,
        status: response.status,
        data,
      };
    },
  };
}

function getDesktopBridge(): DesktopBridge {
  return (window as Window & { tooriDesktop?: DesktopBridge }).tooriDesktop || browserBridge();
}

function isDesktopBridgeAvailable(): boolean {
  return Boolean((window as Window & { tooriDesktop?: DesktopBridge }).tooriDesktop);
}

function assetUrl(filePath: string): string {
  if (isDesktopBridgeAvailable()) {
    return `file://${filePath}`;
  }
  return `${BROWSER_RUNTIME_URL}/v1/file?path=${encodeURIComponent(filePath)}`;
}

async function readFileAsBase64(file: File): Promise<string> {
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Unable to read file"));
    reader.onload = () => {
      const result = String(reader.result || "");
      const [, payload = ""] = result.split(",", 2);
      resolve(payload);
    };
    reader.readAsDataURL(file);
  });
}

async function pickImagePayload(): Promise<{ filePath?: string; imageBase64?: string } | null> {
  const bridge = getDesktopBridge();
  if (bridge.pickFile) {
    const filePath = await bridge.pickFile();
    return filePath ? { filePath } : null;
  }
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/png,image/jpeg,image/webp";
  return await new Promise((resolve) => {
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) {
        resolve(null);
        return;
      }
      resolve({ imageBase64: await readFileAsBase64(file) });
    };
    input.click();
  });
}

async function getCameraAccessStateFallback(): Promise<{ status: string; granted: boolean; canPrompt: boolean }> {
  try {
    if (!("permissions" in navigator) || !(navigator.permissions as any).query) {
      return { status: "unknown", granted: true, canPrompt: true };
    }
    const result = await (navigator.permissions as any).query({ name: "camera" });
    return {
      status: result.state,
      granted: result.state === "granted",
      canPrompt: result.state === "prompt",
    };
  } catch {
    return { status: "unknown", granted: true, canPrompt: true };
  }
}

async function requestCameraAccessFallback(): Promise<{ status: string; granted: boolean; canPrompt: boolean }> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    stream.getTracks().forEach((track) => track.stop());
    return { status: "granted", granted: true, canPrompt: false };
  } catch (error) {
    const message = (error as Error).message.toLowerCase();
    if (message.includes("denied") || message.includes("permission")) {
      return { status: "denied", granted: false, canPrompt: false };
    }
    return { status: "unknown", granted: false, canPrompt: true };
  }
}

async function runtimeRequest<T>(path: string, method = "GET", body?: unknown): Promise<T> {
  const response = await getDesktopBridge().request(path, { method, body });
  if (!response.ok) {
    const message =
      typeof response.data === "string"
        ? response.data
        : response.data?.detail || `Request failed with status ${response.status}`;
    throw new Error(message);
  }
  return response.data as T;
}

function cameraScore(label: string): number {
  const normalized = label.toLowerCase();
  let score = 0;
  if (normalized.includes("facetime")) score += 8;
  if (normalized.includes("continuity")) score += 7;
  if (normalized.includes("iphone")) score += 6;
  if (normalized.includes("usb")) score += 4;
  if (normalized.includes("camera")) score += 2;
  if (normalized.includes("virtual")) score -= 4;
  if (normalized.includes("obs")) score -= 4;
  return score;
}

function sortCameras(devices: CameraDeviceOption[]): CameraDeviceOption[] {
  return [...devices].sort((left, right) => {
    const scoreDelta = cameraScore(right.label) - cameraScore(left.label);
    if (scoreDelta !== 0) {
      return scoreDelta;
    }
    return left.label.localeCompare(right.label);
  });
}

function pickBestCamera(
  devices: CameraDeviceOption[],
  preferredDeviceId?: string,
  currentDeviceId?: string,
): string | undefined {
  if (preferredDeviceId && devices.some((device) => device.deviceId === preferredDeviceId)) {
    return preferredDeviceId;
  }
  if (currentDeviceId && devices.some((device) => device.deviceId === currentDeviceId)) {
    return currentDeviceId;
  }
  return sortCameras(devices)[0]?.deviceId;
}

function formatLatency(latencyMs?: number | null): string {
  if (latencyMs == null) {
    return "n/a";
  }
  return `${latencyMs.toFixed(0)} ms`;
}

function diagnosticsMessage(phase: string, error?: string | null): string {
  if (error) {
    return error;
  }
  switch (phase) {
    case "requesting permission":
      return "Requesting camera permission";
    case "camera retrying":
      return "Retrying camera stream";
    case "stream attached":
      return "Stream attached, waiting for frames";
    case "preview ready":
      return "Live preview ready";
    case "video stalled":
      return "Camera stream stalled";
    case "black frame detected":
      return "Camera is active but frames are effectively black";
    case "camera error":
      return "Unable to start the camera";
    default:
      return "Camera not started";
  }
}

async function waitForVideoReadiness(
  video: HTMLVideoElement,
  stream: MediaStream,
): Promise<{ ready: boolean; message: string }> {
  const track = stream.getVideoTracks()[0];
  const currentWidth = () => video.videoWidth || Number(track.getSettings().width || 0);
  const currentHeight = () => video.videoHeight || Number(track.getSettings().height || 0);
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA && currentWidth() > 0 && currentHeight() > 0) {
    return { ready: true, message: "Camera metadata loaded" };
  }
  try {
    await video.play();
  } catch {
    // Electron may reject autoplay before the media element settles.
  }
  return await new Promise<{ ready: boolean; message: string }>((resolve) => {
    const finish = (ready: boolean, message: string) => {
      cleanup();
      resolve({ ready, message });
    };
    const evaluate = () => {
      if (track.readyState === "live" && currentWidth() > 0 && currentHeight() > 0) {
        finish(true, "Camera stream attached");
      }
    };
    const timeout = window.setTimeout(() => {
      if (track.readyState === "live") {
        finish(false, "Camera metadata is delayed; keeping the stream attached");
        return;
      }
      finish(false, "Timed out waiting for camera metadata");
    }, 4500);
    const interval = window.setInterval(evaluate, 180);
    const handleReady = () => evaluate();
    const handleError = () => finish(false, "Camera metadata failed to load");
    function cleanup() {
      window.clearTimeout(timeout);
      window.clearInterval(interval);
      video.removeEventListener("loadedmetadata", handleReady);
      video.removeEventListener("loadeddata", handleReady);
      video.removeEventListener("canplay", handleReady);
      video.removeEventListener("resize", handleReady);
      video.removeEventListener("error", handleError);
    }
    video.addEventListener("loadedmetadata", handleReady);
    video.addEventListener("loadeddata", handleReady);
    video.addEventListener("canplay", handleReady);
    video.addEventListener("resize", handleReady);
    video.addEventListener("error", handleError);
    evaluate();
  });
}

function renderReasoningTrace(trace: ReasoningTraceEntry[]) {
  if (!trace.length) {
    return null;
  }
  return (
    <div className="trace-list">
      {trace.map((entry) => (
        <div key={`${entry.provider}-${entry.attempted}-${entry.success}-${entry.error || "ok"}`} className="trace-item">
          <div className="trace-head">
            <strong>{entry.provider}</strong>
            <span data-success={entry.success}>
              {entry.success ? "answered" : entry.attempted ? "failed" : "skipped"}
            </span>
          </div>
          <p>{entry.success ? `${entry.health_message} in ${formatLatency(entry.latency_ms)}` : entry.error || entry.health_message}</p>
        </div>
      ))}
    </div>
  );
}

function normalizeBoxes(
  boxes: unknown,
  observation?: Observation | null,
): BoundingBox[] {
  if (!Array.isArray(boxes)) {
    return [];
  }
  const width = Math.max(typeof observation?.width === "number" ? observation.width : 1, 1);
  const height = Math.max(typeof observation?.height === "number" ? observation.height : 1, 1);
  return boxes.flatMap((item) => {
    if (!item || typeof item !== "object") {
      return [];
    }
    const raw = item as Record<string, unknown>;
    const x = Number(raw.x);
    const y = Number(raw.y);
    const boxWidth = Number(raw.width);
    const boxHeight = Number(raw.height);
    if (![x, y, boxWidth, boxHeight].every(Number.isFinite)) {
      return [];
    }
    const normalized = boxWidth <= 1.2 && boxHeight <= 1.2 && x <= 1.2 && y <= 1.2
      ? { x, y, width: boxWidth, height: boxHeight }
      : { x: x / width, y: y / height, width: boxWidth / width, height: boxHeight / height };
    return [{
      ...normalized,
      label: raw.label ? String(raw.label) : null,
      score: raw.score == null ? null : Number(raw.score),
    }];
  });
}

function explainSummary(observation?: Observation | null, answerText?: string | null): string {
  if (answerText && answerText.trim()) {
    return answerText.trim();
  }
  if (!observation) {
    return "Waiting for live scene updates.";
  }
  const rawSummary = String(observation.summary || "").trim();
  const metadata = (observation.metadata || {}) as Record<string, unknown>;
  const perception = metadata.perception && typeof metadata.perception === "object"
    ? metadata.perception as Record<string, unknown>
    : {};
  const topLabel = String(perception.top_label || observation.tags?.[0] || "").replace(/_/g, " ").trim();
  const color = String(perception.dominant_color || "").trim();
  const brightness = String(perception.brightness_label || "").trim();
  const edge = String(perception.edge_label || "").trim();
  if (rawSummary && !/(dominant|balanced|textured|smooth)\s+scene$/i.test(rawSummary)) {
    return rawSummary;
  }
  const parts = [];
  if (topLabel) {
    parts.push(`Likely primary object: ${topLabel}.`);
  }
  if (color || brightness || edge) {
    parts.push(
      `Visual cues: ${[color && `${color} color`, brightness && `${brightness} lighting`, edge && `${edge} detail`]
        .filter(Boolean)
        .join(", ")}.`,
    );
  }
  if (!parts.length && rawSummary) {
    return `Local visual reading: ${rawSummary}.`;
  }
  return parts.join(" ");
}

function explainWorldModel(sceneState?: SceneState | null): string {
  if (!sceneState) {
    return "The world model is waiting for a few frames before it can tell you what stayed stable and what changed.";
  }
  const stable = sceneState.stable_elements.slice(0, 3);
  const changed = sceneState.changed_elements.slice(0, 2);
  if (stable.length && changed.length) {
    return `Expected the scene to stay consistent around ${stable.join(", ")}. The biggest change detected was ${changed.join(", ")}.`;
  }
  if (stable.length) {
    return `The model expects continuity around ${stable.join(", ")} based on the recent scene history.`;
  }
  if (changed.length) {
    return `The model detected change in ${changed.join(", ")} and is updating its scene memory.`;
  }
  return "The model is still building a stable scene hypothesis from the recent frames.";
}

function ObservationThumbnail({ src, alt }: { src: string; alt: string }) {
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setFailed(false);
  }, [src]);

  if (failed) {
    return <div className="image-fallback">Preview unavailable</div>;
  }

  return (
    <img
      src={src}
      alt=""
      title={alt}
      crossOrigin="anonymous"
      onError={() => setFailed(true)}
    />
  );
}

const DETECTION_COLORS = [
  "rgba(67, 216, 201, 0.85)",   // Teal
  "rgba(255, 140, 66, 0.85)",   // Orange
  "rgba(130, 171, 255, 0.85)",  // Blue
  "rgba(246, 199, 106, 0.85)",  // Yellow
  "rgba(255, 107, 107, 0.85)",  // Red
  "rgba(155, 107, 255, 0.85)",  // Purple
  "rgba(84, 206, 176, 0.85)",   // Green
];

function getLabelColor(label: string): string {
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = label.charCodeAt(i) + ((hash << 5) - hash);
  }
  return DETECTION_COLORS[Math.abs(hash) % DETECTION_COLORS.length];
}

function DetectionOverlay({ boxes }: { boxes: BoundingBox[] }) {
  if (!boxes.length) {
    return null;
  }
  return (
    <div className="detection-overlay" aria-hidden="true">
      {boxes.map((box, index) => {
        const fallbackLabel = box.label || "tracked region";
        const color = getLabelColor(fallbackLabel);
        return (
          <div
            key={`${fallbackLabel}-${index}-${box.x}-${box.y}`}
            className="detection-box"
            style={{
              left: `${Math.max(0, Math.min(box.x, 0.96)) * 100}%`,
              top: `${Math.max(0, Math.min(box.y, 0.96)) * 100}%`,
              width: `${Math.max(0.04, Math.min(box.width, 1 - box.x)) * 100}%`,
              height: `${Math.max(0.04, Math.min(box.height, 1 - box.y)) * 100}%`,
              borderColor: color,
              backgroundColor: color.replace('0.85)', '0.12)'),
            }}
          >
            <span style={{ backgroundColor: color }}>
              {fallbackLabel}
              {box.score != null ? ` ${box.score.toFixed(2)}` : ""}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function ChallengeMetricChart({ history }: { history: SceneState[] }) {
  if (!history.length) {
    return (
      <div className="metric-chart empty">
        <p className="muted">Metric trend lines will appear once Living Lens has recorded a few world-state updates.</p>
      </div>
    );
  }
  const points = [...history].slice(0, 8).reverse();
  return (
    <div className="metric-chart-shell">
      <div className="metric-chart-legend" aria-hidden="true">
        <span className="legend-dot continuity">Continuity</span>
        <span className="legend-dot surprise">Surprise</span>
        <span className="legend-dot persistence">Persistence</span>
      </div>
      <div className="metric-chart" aria-label="World-model metrics over time">
        {points.map((state, index) => (
          <div
            key={state.id}
            className={index === points.length - 1 ? "metric-column is-latest" : "metric-column"}
          >
            <div className="metric-bars">
              <span className="metric-lane" />
              <span className="metric-lane" />
              <span className="metric-lane" />
              <span className="metric-lane" />
              <span className="metric-lane" />
            <span
              className="metric-bar continuity"
              style={{ height: `${Math.max(8, Math.round(state.metrics.temporal_continuity_score * 100))}%` }}
              title={`Continuity ${state.metrics.temporal_continuity_score.toFixed(2)}`}
            />
            <span
              className="metric-bar surprise"
              style={{ height: `${Math.max(8, Math.round(state.metrics.surprise_score * 100))}%` }}
              title={`Surprise ${state.metrics.surprise_score.toFixed(2)}`}
            />
            <span
              className="metric-bar persistence"
              style={{ height: `${Math.max(8, Math.round(state.metrics.persistence_confidence * 100))}%` }}
              title={`Persistence ${state.metrics.persistence_confidence.toFixed(2)}`}
            />
            </div>
            <small>t{index + 1}</small>
          </div>
        ))}
      </div>
    </div>
  );
}

function BaselineScoreboard({ comparison }: { comparison: BaselineComparison }) {
  const rows = [
    { label: "JEPA / Hybrid", tone: "accent", value: comparison.jepa_hybrid },
    { label: "Frame captioning", tone: "neutral", value: comparison.frame_captioning },
    { label: "Embedding retrieval", tone: "neutral", value: comparison.embedding_retrieval },
  ];
  const winner = rows.reduce((best, row) => (row.value.composite > best.value.composite ? row : best), rows[0]);
  return (
    <div className="baseline-scoreboard">
      {rows.map((row) => (
        <div
          key={row.label}
          className={row.label === winner.label ? "baseline-row is-winner" : "baseline-row"}
          data-tone={row.tone}
        >
          <div className="baseline-copy">
            <strong>{row.label}</strong>
            <p className="muted">{row.label === winner.label ? "Current best performer on this sequence." : "Comparison baseline for the same sequence."}</p>
          </div>
          <div className="baseline-metrics">
            <div className="baseline-metric">
              <span>Continuity</span>
              <div className="baseline-mini-bar continuity">
                <span style={{ width: `${Math.max(8, Math.round(row.value.continuity * 100))}%` }} />
              </div>
              <strong>{row.value.continuity.toFixed(2)}</strong>
            </div>
            <div className="baseline-metric">
              <span>Persistence</span>
              <div className="baseline-mini-bar persistence">
                <span style={{ width: `${Math.max(8, Math.round(row.value.persistence * 100))}%` }} />
              </div>
              <strong>{row.value.persistence.toFixed(2)}</strong>
            </div>
            <div className="baseline-metric">
              <span>Change separation</span>
              <div className="baseline-mini-bar surprise">
                <span style={{ width: `${Math.max(8, Math.round(row.value.surprise_separation * 100))}%` }} />
              </div>
              <strong>{row.value.surprise_separation.toFixed(2)}</strong>
            </div>
          </div>
          <div className="baseline-composite">
            <span>Overall</span>
            <strong>{row.value.composite.toFixed(2)}</strong>
          </div>
        </div>
      ))}
      <p className="muted baseline-summary">{comparison.summary}</p>
    </div>
  );
}

function challengeStepExpectation(index: number): string {
  switch (index) {
    case 0:
      return "Give the model a stable baseline. Continuity should climb while novelty and surprise stay low.";
    case 1:
      return "A partial occlusion should keep persistence high while continuity dips only slightly.";
    case 2:
      return "A full occlusion should move the tracked entity to an occluded state without erasing it from memory.";
    case 3:
      return "When the object returns, the persistence graph should recover the same track instead of inventing a new one.";
    case 4:
      return "Moving away and back should preserve the main scene hypothesis and reconnect the remembered state.";
    case 5:
      return "An unexpected distractor should produce a visible surprise spike while stable objects remain persistent.";
    default:
      return "Follow the live instruction and watch the continuity, surprise, and persistence signals update.";
  }
}

export default function App() {
  const desktopBridgeAvailable = isDesktopBridgeAvailable();
  const [activeTab, setActiveTab] = useState<(typeof tabs)[number]>("Living Lens");
  const [livingSection, setLivingSection] = useState<"overview" | "memory" | "challenge">("overview");
  const [settings, setSettings] = useState<Settings | null>(null);
  const [health, setHealth] = useState<ProviderHealth[]>([]);
  const [observations, setObservations] = useState<Observation[]>([]);
  const [worldState, setWorldState] = useState<WorldStateResponse | null>(null);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [prompt, setPrompt] = useState("");
  const [searchText, setSearchText] = useState("");
  const [sessionId, setSessionId] = useState("desktop-live");
  const [status, setStatus] = useState("Connecting to runtime");
  const [eventsState, setEventsState] = useState("disconnected");
  const [savingSettings, setSavingSettings] = useState(false);
  const [cameraDevices, setCameraDevices] = useState<CameraDeviceOption[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState("default");
  const [cameraDiagnostics, setCameraDiagnostics] = useState<CameraDiagnostics>(DEFAULT_CAMERA_DIAGNOSTICS);
  const [cameraAccess, setCameraAccess] = useState<CameraAccessState>({
    status: "unknown",
    granted: false,
    canPrompt: false,
  });
  const [cameraBusy, setCameraBusy] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraStreamLive, setCameraStreamLive] = useState(false);
  const [streamEpoch, setStreamEpoch] = useState(0);
  const [livingLensEnabled, setLivingLensEnabled] = useState(true);
  const [showEnergyMap, setShowEnergyMap] = useState<boolean>(true);
  const [showEntities, setShowEntities] = useState(true);
  const [lensMaximized, setLensMaximized] = useState(false);
  const [layoutMode, setLayoutMode] = useState<"grid" | "sidebar" | "stacked" | "focus">("grid");
  
  const defaultLayouts = {
    grid: {
      lg: [
        { i: 'monitor', x: 0, y: 0, w: 8, h: 6, minW: 4, minH: 4 },
        { i: 'understanding', x: 8, y: 0, w: 4, h: 3 },
        { i: 'pulse', x: 8, y: 3, w: 4, h: 3 },
        { i: 'matches', x: 0, y: 6, w: 12, h: 4 },
      ]
    },
    sidebar: {
      lg: [
        { i: 'monitor', x: 0, y: 0, w: 4, h: 10 },
        { i: 'understanding', x: 4, y: 0, w: 8, h: 5 },
        { i: 'pulse', x: 4, y: 5, w: 8, h: 5 },
        { i: 'matches', x: 0, y: 10, w: 12, h: 4 },
      ]
    },
    stacked: {
      lg: [
        { i: 'monitor', x: 0, y: 0, w: 12, h: 6 },
        { i: 'understanding', x: 0, y: 6, w: 12, h: 3 },
        { i: 'pulse', x: 0, y: 9, w: 12, h: 3 },
        { i: 'matches', x: 0, y: 12, w: 12, h: 4 },
      ]
    },
    focus: {
      lg: [
        { i: 'monitor', x: 0, y: 0, w: 12, h: 10 },
        { i: 'understanding', x: 0, y: 10, w: 6, h: 4 },
        { i: 'pulse', x: 6, y: 10, w: 6, h: 4 },
        { i: 'matches', x: 0, y: 14, w: 12, h: 4 },
      ]
    }
  };

  const [currentLayouts, setCurrentLayouts] = useState(defaultLayouts.grid);

  useEffect(() => {
    setCurrentLayouts(defaultLayouts[layoutMode]);
  }, [layoutMode]);

  const [livingLensIntervalS, setLivingLensIntervalS] = useState(6);
  const [livingLensPrompt, setLivingLensPrompt] = useState("");
  const [livingLensBusy, setLivingLensBusy] = useState(false);
  const [livingLensStatus, setLivingLensStatus] = useState("Continuous monitoring is ready");
  const [livingLensResult, setLivingLensResult] = useState<LivingLensTickResponse | null>(null);
  const [challengeRun, setChallengeRun] = useState<ChallengeRun | null>(null);
  const [challengeBusy, setChallengeBusy] = useState(false);
  const [challengeGuideActive, setChallengeGuideActive] = useState(false);
  const [challengeStepIndex, setChallengeStepIndex] = useState(0);
  const [showAllTracks, setShowAllTracks] = useState(false);
  const liveVideoRef = useRef<HTMLVideoElement | null>(null);
  const livingVideoRef = useRef<HTMLVideoElement | null>(null);
  const liveCaptureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const liveDiagnosticsCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const livingCaptureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const livingDiagnosticsCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const deferredSearchText = useDeferredValue(searchText);
  const streamRef = useRef<MediaStream | null>(null);
  const activeCameraDeviceIdRef = useRef("default");
  const livingLensInFlightRef = useRef(false);
  const livingLensLastTickRef = useRef(0);

  function activeVideoElement() {
    return activeTab === "Living Lens" ? livingVideoRef.current : liveVideoRef.current;
  }

  function activeDiagnosticsCanvas() {
    return activeTab === "Living Lens" ? livingDiagnosticsCanvasRef.current : liveDiagnosticsCanvasRef.current;
  }

  function frameElements(mode: "live" | "living") {
    if (mode === "living") {
      return {
        video: livingVideoRef.current,
        canvas: livingCaptureCanvasRef.current,
      };
    }
    return {
      video: liveVideoRef.current,
      canvas: liveCaptureCanvasRef.current,
    };
  }

  async function refreshAll() {
    const [settingsResponse, healthResponse, observationsResponse, worldStateResponse] = await Promise.all([
      runtimeRequest<Settings>("/v1/settings"),
      runtimeRequest<{ providers: ProviderHealth[] }>("/v1/providers/health"),
      runtimeRequest<{ observations: Observation[] }>(`/v1/observations?session_id=${encodeURIComponent(sessionId)}&limit=48`),
      runtimeRequest<WorldStateResponse>(`/v1/world-state?session_id=${encodeURIComponent(sessionId)}`),
    ]);
    setSettings(settingsResponse);
    setHealth(healthResponse.providers);
    setObservations(observationsResponse.observations);
    setWorldState(worldStateResponse);
    setChallengeRun(worldStateResponse.challenges?.[0] || null);
    setStatus("Runtime ready");
  }

  useEffect(() => {
    refreshAll().catch((error) => setStatus(error.message));
  }, [sessionId]);

  useEffect(() => {
    const socket = new WebSocket("ws://127.0.0.1:7777/v1/events");
    socket.onopen = () => setEventsState("connected");
    socket.onclose = () => setEventsState("disconnected");
    socket.onmessage = () => {
      startTransition(() => {
        refreshAll().catch(() => undefined);
      });
    };
    return () => socket.close();
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    const preference = settings?.theme_preference || "system";
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const applyTheme = () => {
      const resolved = preference === "system"
        ? (mediaQuery.matches ? "dark" : "light")
        : preference;
      root.dataset.theme = resolved;
      root.style.colorScheme = resolved;
    };
    applyTheme();
    if (preference !== "system") {
      return () => undefined;
    }
    const handleChange = () => applyTheme();
    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, [settings?.theme_preference]);

  useEffect(() => {
    if (!deferredSearchText.trim()) {
      return;
    }
    const timeout = window.setTimeout(() => {
      runtimeRequest<QueryResponse>("/v1/query", "POST", {
        query: deferredSearchText,
        session_id: sessionId,
        top_k: settings?.top_k || 6,
      })
        .then(setQueryResult)
        .catch((error) => setStatus(error.message));
    }, 250);
    return () => window.clearTimeout(timeout);
  }, [deferredSearchText, sessionId, settings?.top_k]);

  useEffect(() => {
    if (activeTab !== "Living Lens" || !livingLensEnabled || !cameraStreamLive) {
      return () => undefined;
    }
    const tick = () => {
      if (livingLensInFlightRef.current) {
        return;
      }
      const intervalMs = Math.max(2, livingLensIntervalS) * 1000;
      if (Date.now() - livingLensLastTickRef.current < intervalMs) {
        return;
      }
      livingLensInFlightRef.current = true;
      livingLensLastTickRef.current = Date.now();
      setLivingLensBusy(true);
      setLivingLensStatus("Analyzing live scene");
      runLivingLensTick({
        query: livingLensPrompt.trim() || undefined,
        decodeMode: livingLensPrompt.trim() ? "force" : "auto",
        topK: settings?.top_k || 6,
      })
        .then((result) => {
          if (!result) {
            setLivingLensStatus("Waiting for a usable frame");
            return;
          }
          setLivingLensResult(result);
          setLivingLensStatus(`Updated ${new Date().toLocaleTimeString()}`);
        })
        .catch((error) => {
          setLivingLensStatus((error as Error).message);
        })
        .finally(() => {
          livingLensInFlightRef.current = false;
          setLivingLensBusy(false);
        });
    };

    tick();
    const interval = window.setInterval(tick, 1200);
    return () => window.clearInterval(interval);
  }, [activeTab, cameraStreamLive, livingLensEnabled, livingLensIntervalS, livingLensPrompt, sessionId, settings?.top_k]);

  useEffect(() => {
    if (!settings) {
      return;
    }
    const desiredDeviceId = settings.camera_device && settings.camera_device !== "default"
      ? settings.camera_device
      : undefined;
    if (streamRef.current && (!desiredDeviceId || activeCameraDeviceIdRef.current === desiredDeviceId)) {
      setSelectedCameraId(desiredDeviceId || activeCameraDeviceIdRef.current || "default");
      return;
    }
    startCamera({ preferredDeviceId: desiredDeviceId, phase: "requesting permission" }).catch(() => undefined);
    return () => undefined;
  }, [settings?.camera_device]);

  useEffect(() => {
    return () => stopStream(streamRef.current);
  }, []);

  useEffect(() => {
    const stream = streamRef.current;
    const video = activeVideoElement();
    if (!stream || !video) {
      return () => undefined;
    }
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    video.autoplay = true;
    video.play().catch(() => undefined);
    return () => undefined;
  }, [activeTab, streamEpoch]);

  useEffect(() => {
    const stream = streamRef.current;
    const video = activeVideoElement();
    const canvas = activeDiagnosticsCanvas();
    if (!stream || !video || !canvas) {
      return;
    }
    const track = stream.getVideoTracks()[0];
    let lastCurrentTime = video.currentTime;
    let lastFrameMs = Date.now();
    let darkFrameCount = 0;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    const interval = window.setInterval(() => {
      if (!context) {
        return;
      }
      const width = video.videoWidth || Number(track.getSettings().width || 0);
      const height = video.videoHeight || Number(track.getSettings().height || 0);
      const now = Date.now();
      if (video.currentTime !== lastCurrentTime) {
        lastCurrentTime = video.currentTime;
        lastFrameMs = now;
      }
      let frameLuma: number | null = null;
      if (width > 0 && height > 0 && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        canvas.width = 64;
        canvas.height = 36;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
        let totalLuma = 0;
        for (let index = 0; index < pixels.length; index += 4) {
          totalLuma += (pixels[index] * 0.299) + (pixels[index + 1] * 0.587) + (pixels[index + 2] * 0.114);
        }
        frameLuma = totalLuma / (pixels.length / 4);
      }
      if (frameLuma != null && frameLuma < 6) {
        darkFrameCount += 1;
      } else {
        darkFrameCount = 0;
      }
      const blackFrameDetected = darkFrameCount >= 3 && width > 0 && height > 0;
      const stalled = now - lastFrameMs > 2500;
      setCameraStreamLive(track.readyState === "live");
      const phase = track.readyState !== "live"
        ? "video stalled"
        : blackFrameDetected
          ? "black frame detected"
          : stalled
            ? "video stalled"
            : "preview ready";
      const resolution = width > 0 && height > 0 ? `${width} x ${height}` : "0 x 0";
      const lastFrameAt = width > 0 && height > 0 ? new Date(lastFrameMs).toLocaleTimeString() : null;
      setCameraReady(track.readyState === "live" && !blackFrameDetected && !stalled && width > 0 && height > 0);
      setCameraDiagnostics((current) => ({
        ...current,
        phase,
        resolution,
        readyState: String(video.readyState),
        trackState: track.readyState,
        trackMuted: track.muted,
        trackEnabled: track.enabled,
        lastFrameAt,
        frameLuma,
        blackFrameDetected,
        message: diagnosticsMessage(phase, current.error),
      }));
    }, 700);
    return () => window.clearInterval(interval);
  }, [streamEpoch, activeTab]);

  function stopStream(stream: MediaStream | null) {
    if (!stream) {
      return;
    }
    for (const track of stream.getTracks()) {
      track.stop();
    }
  }

  async function enumerateVideoDevices(): Promise<CameraDeviceOption[]> {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return sortCameras(
      devices
        .filter((device) => device.kind === "videoinput")
        .map((device, index) => ({
          deviceId: device.deviceId,
          label: device.label || `Camera ${index + 1}`,
        })),
    );
  }

  async function persistCameraDevice(deviceId: string) {
    if (!deviceId || deviceId === "default") {
      return;
    }
    const latest = await runtimeRequest<Settings>("/v1/settings");
    if (latest.camera_device === deviceId) {
      setSettings((current) => (current ? { ...current, camera_device: deviceId } : current));
      return;
    }
    latest.camera_device = deviceId;
    const saved = await runtimeRequest<Settings>("/v1/settings", "PUT", latest);
    setSettings(saved);
  }

  async function attachStream(
    stream: MediaStream,
    devices: CameraDeviceOption[],
    requestedDeviceId?: string,
    permissionStatus = "unknown",
  ): Promise<{ ready: boolean; message: string }> {
    const video = activeVideoElement();
    if (!video) {
      throw new Error("Video element is unavailable");
    }
    stopStream(streamRef.current);
    streamRef.current = stream;
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    video.autoplay = true;
    setCameraDiagnostics((current) => ({
      ...current,
      phase: "stream attached",
      message: diagnosticsMessage("stream attached"),
      error: null,
    }));
    const readiness = await waitForVideoReadiness(video, stream);
    const track = stream.getVideoTracks()[0];
    if (track.readyState !== "live") {
      throw new Error("Camera track never became live");
    }
    setCameraStreamLive(true);
    const effectiveDeviceId = String(track.getSettings().deviceId || requestedDeviceId || "default");
    const effectiveLabel = devices.find((device) => device.deviceId === effectiveDeviceId)?.label || track.label || "Auto camera";
    activeCameraDeviceIdRef.current = effectiveDeviceId;
    setSelectedCameraId(effectiveDeviceId);
    setCameraDiagnostics({
      phase: readiness.ready ? "stream attached" : "camera retrying",
      selectedDeviceId: effectiveDeviceId,
      selectedLabel: effectiveLabel,
      permissionStatus,
      resolution: `${video.videoWidth || track.getSettings().width || 0} x ${video.videoHeight || track.getSettings().height || 0}`,
      readyState: String(video.readyState),
      trackState: track.readyState,
      trackMuted: track.muted,
      trackEnabled: track.enabled,
      lastFrameAt: new Date().toLocaleTimeString(),
      frameLuma: null,
      blackFrameDetected: false,
      message: readiness.message,
      error: readiness.ready ? null : readiness.message,
    });
    setStreamEpoch((current) => current + 1);
    await persistCameraDevice(effectiveDeviceId);
    return readiness;
  }

  async function startCamera(options: { preferredDeviceId?: string; phase: string; forceAuto?: boolean }) {
    if (!navigator.mediaDevices?.getUserMedia) {
      const message = "This environment does not support camera capture";
      setCameraStreamLive(false);
      setCameraDiagnostics({
        ...DEFAULT_CAMERA_DIAGNOSTICS,
        phase: "camera error",
        message,
        error: message,
      });
      setStatus(message);
      return;
    }
    setCameraBusy(true);
    setCameraReady(false);
    setCameraStreamLive(false);
    const bridge = getDesktopBridge();
    let access = bridge.getCameraAccess
      ? await bridge.getCameraAccess()
      : await getCameraAccessStateFallback();
    if (!access.granted && access.canPrompt) {
      access = bridge.requestCameraAccess
        ? await bridge.requestCameraAccess()
        : await requestCameraAccessFallback();
    }
    setCameraAccess(access);
    if (!access.granted) {
      setCameraStreamLive(false);
      const message = access.status === "denied"
        ? "Camera permission denied in macOS Privacy & Security"
        : access.status === "restricted"
          ? "Camera access is restricted by macOS"
          : access.status === "not-determined"
            ? "macOS did not show the camera prompt"
            : "Camera access is not granted";
      setCameraDiagnostics({
        ...DEFAULT_CAMERA_DIAGNOSTICS,
        phase: "camera error",
        permissionStatus: access.status,
        message,
        error: message,
      });
      setStatus(message);
      setCameraBusy(false);
      return;
    }
    setCameraDiagnostics((current) => ({
      ...current,
      phase: options.phase,
      permissionStatus: access.status,
      message: diagnosticsMessage(options.phase),
      error: null,
      blackFrameDetected: false,
    }));
    let provisionalStream: MediaStream | null = null;
    let selectedStream: MediaStream | null = null;
    try {
      provisionalStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      const devices = await enumerateVideoDevices();
      setCameraDevices(devices);
      const currentDeviceId = String(provisionalStream.getVideoTracks()[0]?.getSettings().deviceId || "");
      const desiredDeviceId = options.forceAuto
        ? pickBestCamera(devices, undefined, currentDeviceId)
        : pickBestCamera(devices, options.preferredDeviceId, currentDeviceId);
      if (desiredDeviceId && desiredDeviceId !== currentDeviceId) {
        try {
          selectedStream = await navigator.mediaDevices.getUserMedia({
            video: {
              deviceId: { exact: desiredDeviceId },
              width: { ideal: 1280 },
              height: { ideal: 720 },
            },
            audio: false,
          });
        } catch {
          setCameraDiagnostics((current) => ({
            ...current,
            phase: "camera retrying",
            message: diagnosticsMessage("camera retrying"),
          }));
          selectedStream = provisionalStream;
          provisionalStream = null;
        }
      } else {
        selectedStream = provisionalStream;
        provisionalStream = null;
      }
      if (!selectedStream) {
        throw new Error("Unable to attach a camera stream");
      }
      const readiness = await attachStream(selectedStream, devices, desiredDeviceId, access.status);
      stopStream(provisionalStream);
      setStatus(readiness.ready ? "Camera preview ready" : readiness.message);
    } catch (error) {
      stopStream(selectedStream);
      stopStream(provisionalStream);
      setCameraStreamLive(false);
      const message = (error as Error).message;
      setCameraDiagnostics((current) => ({
        ...current,
        phase: "camera error",
        permissionStatus: access.status,
        message: diagnosticsMessage("camera error", message),
        error: message,
      }));
      setStatus(`Camera unavailable: ${message}`);
    } finally {
      setCameraBusy(false);
    }
  }

  async function requestCameraPermission() {
    setCameraBusy(true);
    try {
      const bridge = getDesktopBridge();
      const access = bridge.requestCameraAccess
        ? await bridge.requestCameraAccess()
        : await requestCameraAccessFallback();
      setCameraAccess(access);
      if (access.granted) {
        setStatus("Camera permission granted");
        await retryCamera(true);
        return;
      }
      const message = access.status === "denied"
        ? "Camera permission denied in macOS Privacy & Security"
        : access.status === "restricted"
          ? "Camera access is restricted by macOS"
          : access.status === "not-determined"
            ? "macOS did not show the camera prompt"
            : "Camera access is not granted";
      setCameraDiagnostics((current) => ({
        ...current,
        phase: "camera error",
        permissionStatus: access.status,
        message,
        error: message,
      }));
      setCameraStreamLive(false);
      setStatus(message);
    } finally {
      setCameraBusy(false);
    }
  }

  async function retryCamera(forceAuto = false) {
    const preferredDeviceId = forceAuto
      ? undefined
      : selectedCameraId !== "default"
        ? selectedCameraId
        : settings?.camera_device !== "default"
          ? settings?.camera_device
          : undefined;
    await startCamera({
      preferredDeviceId,
      phase: forceAuto ? "requesting permission" : "camera retrying",
      forceAuto,
    });
  }

  async function switchCamera() {
    if (cameraDevices.length < 2) {
      setStatus("No alternate camera is available");
      return;
    }
    const currentIndex = cameraDevices.findIndex((device) => device.deviceId === selectedCameraId);
    const nextDevice = cameraDevices[(currentIndex + 1 + cameraDevices.length) % cameraDevices.length];
    await startCamera({ preferredDeviceId: nextDevice.deviceId, phase: "camera retrying" });
  }

  function currentFrameBase64(mode: "live" | "living"): string | null {
    const { video, canvas } = frameElements(mode);
    if (!video || !canvas) {
      return null;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx || !video.videoWidth || !video.videoHeight) {
      return null;
    }
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/png").split(",")[1];
  }

  async function analyzeCurrentFrame(options: {
    query?: string;
    decodeMode: "auto" | "force";
    topK?: number;
  }): Promise<AnalyzeResponse | null> {
    const imageBase64 = currentFrameBase64("live");
    if (!imageBase64) {
      return null;
    }
    const result = await runtimeRequest<AnalyzeResponse>("/v1/analyze", "POST", {
      image_base64: imageBase64,
      session_id: sessionId,
      query: options.query || undefined,
      decode_mode: options.decodeMode,
      top_k: options.topK || settings?.top_k || 6,
    });
    setHealth(result.provider_health);
    startTransition(() => {
      refreshAll().catch(() => undefined);
    });
    return result;
  }

  async function runLivingLensTick(options: {
    query?: string;
    decodeMode: "auto" | "force";
    topK?: number;
  }): Promise<LivingLensTickResponse | null> {
    const imageBase64 = currentFrameBase64("living");
    if (!imageBase64) {
      return null;
    }
    const result = await runtimeRequest<LivingLensTickResponse>("/v1/living-lens/tick", "POST", {
      image_base64: imageBase64,
      session_id: sessionId,
      query: options.query || undefined,
      decode_mode: options.decodeMode,
      top_k: options.topK || settings?.top_k || 6,
      proof_mode: "both",
    });
    setHealth(result.provider_health);
    startTransition(() => {
      refreshAll().catch(() => undefined);
    });
    return result;
  }

  async function runChallengeEvaluation(challengeSet: "live" | "curated" | "both" = "both") {
    setChallengeBusy(true);
    setStatus("Evaluating JEPA challenge sequence");
    setLivingSection("challenge");
    try {
      const result = await runtimeRequest<ChallengeRun>("/v1/challenges/evaluate", "POST", {
        session_id: sessionId,
        challenge_set: challengeSet,
        proof_mode: "both",
        limit: 10,
      });
      setChallengeRun(result);
      setStatus(result.summary);
      startTransition(() => {
        refreshAll().catch(() => undefined);
      });
    } catch (error) {
      setStatus((error as Error).message);
    } finally {
      setChallengeBusy(false);
    }
  }

  function startGuidedChallenge() {
    setChallengeGuideActive(true);
    setChallengeStepIndex(0);
    setLivingSection("challenge");
    setActiveTab("Living Lens");
    setLivingLensStatus("Guided challenge started. Follow the first instruction below.");
  }

  function advanceGuidedChallenge() {
    const steps = challengeRun?.guide_steps?.length ? challengeRun.guide_steps : DEFAULT_CHALLENGE_STEPS;
    setChallengeStepIndex((current) => {
      const next = Math.min(current + 1, steps.length - 1);
      if (next === steps.length - 1) {
        setLivingLensStatus("Final guided step reached. Run scoring when you are ready.");
      }
      return next;
    });
  }

  function resetGuidedChallenge() {
    setChallengeGuideActive(false);
    setChallengeStepIndex(0);
    setLivingLensStatus("Continuous monitoring is ready");
  }

  async function captureFrame() {
    if (!cameraReady) {
      setStatus("Video stream is not ready for capture");
      return;
    }
    setStatus("Analyzing current frame");
    try {
      const result = await analyzeCurrentFrame({
        query: prompt || undefined,
        decodeMode: "auto",
        topK: settings?.top_k || 6,
      });
      if (!result) {
        setStatus("Video stream is not ready for capture");
        return;
      }
      setAnalysis(result);
      setStatus(`Captured ${result.observation.id}`);
    } catch (error) {
      setStatus((error as Error).message);
    }
  }

  async function analyzeFile() {
    const payload = await pickImagePayload();
    if (!payload) {
      return;
    }
    setStatus(
      payload.filePath
        ? `Analyzing ${payload.filePath}`
        : "Analyzing selected image",
    );
    try {
      const result = await runtimeRequest<AnalyzeResponse>("/v1/analyze", "POST", {
        file_path: payload.filePath,
        image_base64: payload.imageBase64,
        session_id: sessionId,
        query: prompt || undefined,
        decode_mode: "force",
      });
      setAnalysis(result);
      startTransition(() => {
        refreshAll().catch(() => undefined);
      });
    } catch (error) {
      setStatus((error as Error).message);
    }
  }

  async function saveSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!settings) {
      return;
    }
    setSavingSettings(true);
    try {
      const updated = await runtimeRequest<Settings>("/v1/settings", "PUT", settings);
      setSettings(updated);
      setStatus("Settings saved");
    } catch (error) {
      setStatus((error as Error).message);
    } finally {
      setSavingSettings(false);
    }
  }

  function mutateSetting(path: string[], value: string | number | boolean) {
    setSettings((current) => {
      if (!current) {
        return current;
      }
      const clone = structuredClone(current);
      let pointer = clone;
      for (const key of path.slice(0, -1)) {
        pointer = pointer[key];
      }
      pointer[path[path.length - 1]] = value;
      return clone;
    });
  }

  function mutateProviderEnabled(name: string, enabled: boolean) {
    mutateSetting(["providers", name, "enabled"], enabled);
  }

  const latestObservation = analysis?.observation || observations[0];
  const latestObservationSummary = latestObservation
    ? explainSummary(
      latestObservation,
      analysis?.observation?.id === latestObservation.id ? analysis?.answer?.text : undefined,
    )
    : null;
  const readyProviders = health.filter((item) => item.healthy).length;
  const degradedProviders = Math.max(health.length - readyProviders, 0);
  const currentSceneState = livingLensResult?.scene_state || worldState?.current || null;
  const livingTracks = livingLensResult?.entity_tracks || worldState?.entity_tracks || [];
  const livingObservation = livingLensResult?.observation || latestObservation;
  const livingAnswer = explainSummary(livingObservation, livingLensResult?.answer?.text || currentSceneState?.observed_state_summary);
  const livingContinuity = currentSceneState?.metrics.temporal_continuity_score ?? (livingObservation ? Math.max(0, 1 - livingObservation.novelty) : null);
  const livingNearest = livingLensResult?.hits?.[0] || null;
  const livingMatches = livingLensResult?.hits || [];
  const livingBaseline = livingLensResult?.baseline_comparison || challengeRun?.baseline_comparison || worldState?.challenges?.[0]?.baseline_comparison || null;
  const worldHistory = worldState?.history || [];
  const continuitySignal = currentSceneState?.metrics.continuity_signal;
  const persistenceSignal = currentSceneState?.metrics.persistence_signal;
  const liveObservationMetadata = (analysis?.observation?.metadata as Record<string, unknown> | undefined)
    || (latestObservation?.metadata as Record<string, unknown> | undefined);
  const liveBoxes = normalizeBoxes(
    liveObservationMetadata?.object_proposals
      || liveObservationMetadata?.proposal_boxes
      || liveObservationMetadata?.bounding_boxes,
    analysis?.observation || latestObservation,
  );
  const livingBoxes = normalizeBoxes(
    (livingObservation?.metadata as Record<string, unknown> | undefined)?.bounding_boxes
      || (livingObservation?.metadata as Record<string, unknown> | undefined)?.object_proposals
      || (livingObservation?.metadata as Record<string, unknown> | undefined)?.proposal_boxes
      || (currentSceneState?.metadata as Record<string, unknown> | undefined)?.bounding_boxes
      || (currentSceneState?.metadata as Record<string, unknown> | undefined)?.object_proposals
      || (currentSceneState?.metadata as Record<string, unknown> | undefined)?.proposal_boxes,
    livingObservation,
  );
  const challengeSteps = (challengeRun?.guide_steps?.length ? challengeRun.guide_steps : Array.from(DEFAULT_CHALLENGE_STEPS));
  const currentChallengeStep = challengeGuideActive ? challengeSteps[challengeStepIndex] : null;
  const worldModelSummary = explainWorldModel(currentSceneState);
  const cameraStatusLabel = cameraReady ? "camera ready" : cameraStreamLive ? "camera connected" : "camera idle";
  const livingHistory = [livingLensResult?.scene_state, ...worldHistory]
    .filter((state): state is SceneState => Boolean(state))
    .filter((state, index, list) => list.findIndex((candidate) => candidate.id === state.id) === index);
  const challengeHistory = livingHistory.slice(0, 8);
  const trackStatusRank: Record<string, number> = {
    visible: 0,
    "re-identified": 1,
    recovered: 1,
    occluded: 2,
    disappeared: 3,
    violated: 4,
  };
  const sortedTracks = [...livingTracks].sort((left, right) => {
    const rankDelta = (trackStatusRank[left.status] ?? 5) - (trackStatusRank[right.status] ?? 5);
    if (rankDelta !== 0) {
      return rankDelta;
    }
    return right.continuity_score - left.continuity_score;
  });
  const featuredTracks = sortedTracks.slice(0, 4);
  const displayedTracks = showAllTracks ? sortedTracks : sortedTracks.slice(0, 8);
  const topHealth = health.slice(0, 4);

  return (
    <div className="shell">
      <aside className="nav">
        <div className="nav-header">
          <p className="eyebrow">Toori</p>
          <h1>Lens Assistant</h1>
          <p className="muted nav-copy">
            Camera-first memory, live reasoning, and plugin-ready runtime across desktop, iOS, and Android.
          </p>
        </div>
        <div className="nav-scroll">
          <div className="status-card">
            <div className="status-line">
              <span className="status-dot" data-online={eventsState === "connected"} />
              <span>Event stream: {eventsState}</span>
            </div>
            <div className="muted">{status}</div>
            <div className="status-grid">
              <div className="status-metric">
                <span>Session</span>
                <strong>{sessionId}</strong>
              </div>
              <div className="status-metric">
                <span>Providers</span>
                <strong>{readyProviders} ready / {health.length || 0}</strong>
              </div>
              <div className="status-metric">
                <span>Fallbacks</span>
                <strong>{degradedProviders}</strong>
              </div>
            </div>
            <button className="secondary-link" onClick={() => setActiveTab("Integrations")}>
              View provider health
            </button>
          </div>
          <nav className="tabs">
            {tabs.map((tab) => (
              <button
                key={tab}
                className={tab === activeTab ? "tab active" : "tab"}
                onClick={() => setActiveTab(tab)}
              >
                {tab}
              </button>
            ))}
          </nav>
          <div className="sidebar-health">
            {topHealth.map((item) => (
              <div key={item.name} className="health-badge" data-healthy={item.healthy}>
                <span>{item.name}</span>
                <strong>{item.healthy ? "ready" : "degraded"}</strong>
              </div>
            ))}
          </div>
        </div>
      </aside>

      <main className="workspace">
        <section className="hero">
          <div>
            <p className="eyebrow">{activeTab === "Living Lens" ? "Continuous scene mode" : "Live runtime"}</p>
            <h2>{activeTab}</h2>
            <div className="hero-meta">
              <span>{settings?.primary_perception_provider || "onnx"} perception</span>
              <span>{settings?.reasoning_backend || "cloud"} reasoning</span>
              <span>{cameraStatusLabel}</span>
            </div>
          </div>
          <div className="hero-actions">
            <button className="primary" onClick={captureFrame} disabled={!cameraReady || cameraBusy}>
              Capture Frame
            </button>
            <button onClick={analyzeFile}>Analyze File</button>
          </div>
        </section>

        {activeTab === "Live Lens" && (
          <section className="panel-grid lens-grid">
            <article className="panel camera-panel">
              <div className="panel-head">
                <h3>Camera</h3>
                <span>{settings?.primary_perception_provider || "onnx"} {"->"} {settings?.reasoning_backend || "cloud"}</span>
              </div>
              <div className="camera-controls">
                <label className="field">
                  <span>Camera Device</span>
                  <select
                    value={selectedCameraId}
                    onChange={(event) => {
                      setSelectedCameraId(event.target.value);
                      startCamera({ preferredDeviceId: event.target.value, phase: "camera retrying" }).catch(() => undefined);
                    }}
                  >
                    {cameraDevices.length ? (
                      cameraDevices.map((device) => (
                        <option key={device.deviceId} value={device.deviceId}>
                          {device.label}
                        </option>
                      ))
                    ) : (
                      <option value="default">Auto camera</option>
                    )}
                  </select>
                </label>
                <div className="camera-actions">
                  <button onClick={() => retryCamera(false)} disabled={cameraBusy || (desktopBridgeAvailable && !cameraAccess.granted && cameraAccess.status !== "unknown")}>Retry Camera</button>
                  <button onClick={switchCamera} disabled={cameraBusy || cameraDevices.length < 2 || (desktopBridgeAvailable && !cameraAccess.granted && cameraAccess.status !== "unknown")}>Switch Camera</button>
                  <button onClick={() => retryCamera(true)} disabled={cameraBusy || (desktopBridgeAvailable && !cameraAccess.granted && cameraAccess.status !== "unknown")}>Auto Pick</button>
                </div>
              </div>
              <div className="preview-surface">
                <video ref={liveVideoRef} autoPlay muted playsInline />
                {showEntities && <DetectionOverlay boxes={liveBoxes} />}
                {showEnergyMap && <Suspense fallback={null}><Heatmap3D /></Suspense>}
              </div>
              <canvas ref={liveCaptureCanvasRef} hidden />
              <canvas ref={liveDiagnosticsCanvasRef} hidden />
              <div className="camera-diagnostics">
                <div className="diagnostic-card">
                  <span>Phase</span>
                  <strong>{cameraDiagnostics.phase}</strong>
                </div>
                <div className="diagnostic-card">
                  <span>Camera</span>
                  <strong>{cameraDiagnostics.selectedLabel}</strong>
                </div>
                <div className="diagnostic-card">
                  <span>Permission</span>
                  <strong>{cameraDiagnostics.permissionStatus}</strong>
                </div>
                <div className="diagnostic-card">
                  <span>Resolution</span>
                  <strong>{cameraDiagnostics.resolution}</strong>
                </div>
                <div className="diagnostic-card">
                  <span>Track</span>
                  <strong>{cameraDiagnostics.trackState}</strong>
                </div>
                <div className="diagnostic-card">
                  <span>Ready State</span>
                  <strong>{cameraDiagnostics.readyState}</strong>
                </div>
                <div className="diagnostic-card">
                  <span>Last Frame</span>
                  <strong>{cameraDiagnostics.lastFrameAt || "n/a"}</strong>
                </div>
              </div>
              <div className={cameraDiagnostics.blackFrameDetected ? "camera-health warning" : "camera-health"}>
                <strong>{cameraDiagnostics.message}</strong>
                <p>
                  track enabled: {String(cameraDiagnostics.trackEnabled)} | muted: {String(cameraDiagnostics.trackMuted)}
                  {cameraDiagnostics.frameLuma != null ? ` | luma ${cameraDiagnostics.frameLuma.toFixed(1)}` : ""}
                </p>
              </div>
              {cameraAccess.status !== "unknown" && !cameraAccess.granted ? (
                <div className="camera-health warning">
                  <strong>
                    {desktopBridgeAvailable
                      ? "Electron does not currently have camera permission."
                      : "Camera permission is not currently granted."}
                  </strong>
                  <p>
                    Status: {cameraAccess.status}. Use the buttons below to request access again
                    {desktopBridgeAvailable
                      ? ", open the macOS camera privacy settings, or switch to the browser runtime while the packaged macOS app identity is being fixed."
                      : "."}
                  </p>
                  <div className="camera-actions">
                    <button onClick={requestCameraPermission} disabled={cameraBusy}>
                      Request Camera Access
                    </button>
                    {desktopBridgeAvailable ? (
                      <button
                        onClick={() => {
                          const openCameraSettings = getDesktopBridge().openCameraSettings;
                          if (openCameraSettings) {
                            openCameraSettings().catch(() => undefined);
                          }
                        }}
                      >
                        Open Camera Settings
                      </button>
                    ) : null}
                    {desktopBridgeAvailable ? (
                      <button onClick={() => window.open("http://127.0.0.1:4173/", "_blank", "noopener,noreferrer")}>
                        Open Browser Runtime
                      </button>
                    ) : null}
                  </div>
                </div>
              ) : null}
              <label className="field">
                <span>Prompt</span>
                <textarea
                  rows={3}
                  placeholder="Ask a question or leave blank to let auto-decoding describe the scene."
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                />
              </label>
            </article>

            <article className="panel panel-scroll">
              <div className="panel-head">
                <h3>Latest Observation</h3>
                <span>{latestObservation ? latestObservation.id : "none"}</span>
              </div>
              {latestObservation ? (
                <div className="observation-card-stack">
                  <div className="observation-card">
                    <ObservationThumbnail src={assetUrl(latestObservation.thumbnail_path)} alt={latestObservation.id} />
                    <div>
                      <p className="muted">{new Date(latestObservation.created_at).toLocaleString()}</p>
                      <p>{latestObservationSummary}</p>
                      <div className="chips">
                        <span>novelty {latestObservation.novelty.toFixed(2)}</span>
                        <span>confidence {latestObservation.confidence.toFixed(2)}</span>
                        <span>{latestObservation.providers.join(" + ")}</span>
                      </div>
                    </div>
                  </div>
                  {renderReasoningTrace(analysis?.reasoning_trace || [])}
                </div>
              ) : (
                <p className="muted">No observations captured yet.</p>
              )}
            </article>

            <article className="panel panel-scroll">
              <div className="panel-head">
                <h3>Nearest Memory</h3>
                <span>{analysis?.hits.length || 0} related scenes</span>
              </div>
              <div className="stack scroll-stack">
                {(analysis?.hits || []).map((hit) => (
                  <div key={hit.observation_id} className="list-row">
                    <ObservationThumbnail src={assetUrl(hit.thumbnail_path)} alt={hit.observation_id} />
                    <div>
                      <strong>{hit.observation_id}</strong>
                      <p>{hit.summary || "No summary"}</p>
                    </div>
                    <span>{hit.score.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </article>
          </section>
        )}

        {activeTab === "Living Lens" && (
          <>
          {/* ── Global Controls Header (z-100, never overlapped) ── */}
          <header className="global-controls-header">
            <div className="global-controls-left">
              <label className="field checkbox compact-check">
                <input type="checkbox" checked={livingLensEnabled} onChange={(e) => setLivingLensEnabled(e.target.checked)} />
                <span>Auto analyze</span>
              </label>
              <label className="field checkbox compact-check">
                <input type="checkbox" checked={showEnergyMap} onChange={(e) => setShowEnergyMap(e.target.checked)} />
                <span>3D Heatmap</span>
              </label>
              <label className="field checkbox compact-check">
                <input type="checkbox" checked={showEntities} onChange={(e) => setShowEntities(e.target.checked)} />
                <span>Entities</span>
              </label>
              <div className="interval-control">
                <span className="interval-label">Interval</span>
                <input type="number" min="2" max="30" value={livingLensIntervalS} onChange={(e) => setLivingLensIntervalS(Number(e.target.value) || 6)} />
                <small>s</small>
              </div>
            </div>
            <div className="layout-switcher">
              {[
                { id: "grid", tooltip: "Grid Layout", icon: <svg viewBox="0 0 24 24" fill="currentColor"><rect x="3" y="3" width="7" height="12" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/></svg> },
                { id: "sidebar", tooltip: "Sidebar Focus", icon: <svg viewBox="0 0 24 24" fill="currentColor"><rect x="3" y="3" width="8" height="18" rx="1"/><rect x="14" y="3" width="7" height="18" rx="1"/></svg> },
                { id: "stacked", tooltip: "Stacked Column", icon: <svg viewBox="0 0 24 24" fill="currentColor"><rect x="3" y="3" width="18" height="7" rx="1"/><rect x="3" y="14" width="18" height="7" rx="1"/></svg> },
                { id: "focus", tooltip: "Immersive View", icon: <svg viewBox="0 0 24 24" fill="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><rect x="7" y="7" width="10" height="10" rx="1"/></svg> },
              ].map((l) => (
                <button
                  key={l.id}
                  className={`layout-btn ${layoutMode === l.id ? 'active' : ''}`}
                  onClick={() => setLayoutMode(l.id as any)}
                  title={l.tooltip}
                >
                  {l.icon}
                </button>
              ))}
            </div>
          </header>

          <section className={`living-shell layout-${layoutMode}`}>
            <div className="living-subnav-container">
              <div className="living-subnav" role="tablist" aria-label="Living Lens workspaces">
                {[
                  { id: "overview", label: "Overview", detail: "Scene delta and live metrics" },
                  { id: "memory", label: "Memory", detail: "Continuity memory and entity tracks" },
                  { id: "challenge", label: "Challenge Lab", detail: "Guided proof run and baselines" },
                ].map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    role="tab"
                    aria-selected={livingSection === item.id}
                    className={livingSection === item.id ? "living-subtab active" : "living-subtab"}
                    onClick={() => setLivingSection(item.id as "overview" | "memory" | "challenge")}
                  >
                    <strong>{item.label}</strong>
                    <span>{item.detail}</span>
                  </button>
                ))}
              </div>
            </div>

            <ResponsiveGridLayout
              className="living-stage-grid"
              layouts={currentLayouts}
              breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xss: 0 }}
              cols={{ lg: 12, md: 10, sm: 6, xs: 4, xss: 2 }}
              rowHeight={80}
              onLayoutChange={(current: any, all: any) => {
                if (layoutMode === "grid") setCurrentLayouts(all);
              }}
              draggableHandle=".panel-head"
              margin={[20, 20]}
            >
              <div key="monitor">
                <article className="panel panel--live living-stage-panel" style={{ height: '100%', margin: 0, display: 'flex', flexDirection: 'column' }}>
                  <div className="panel-head">
                    <h3>Scene Monitor</h3>
                  </div>
                  <div className={`living-preview preview-surface ${lensMaximized ? 'maximized' : ''}`} style={{ flex: 1, minHeight: 0 }}>
                    <button 
                      className="maximize-btn" 
                      onClick={(e) => { e.stopPropagation(); setLensMaximized(!lensMaximized); }}
                      title={lensMaximized ? "Restore view" : "Maximize view"}
                    >
                      {lensMaximized ? "⤓ Restore" : "⤢ Maximize"}
                    </button>
                    <div className="video-aspect-container">
                      <video ref={livingVideoRef} autoPlay muted playsInline />
                      {showEntities && <DetectionOverlay boxes={livingBoxes} />}
                      {showEnergyMap && <Suspense fallback={null}><Heatmap3D /></Suspense>}
                      <div className="living-overlay">
                        <span className="overlay-pill">{livingLensEnabled ? "Live" : "Paused"}</span>
                        {currentSceneState ? <span className="overlay-pill">scene {currentSceneState.id.substring(0,8)}</span> : null}
                      </div>
                    </div>
                  </div>
                  <div className="signal-grid living-primary-kpi-grid" style={{ marginTop: 'auto', paddingTop: '1rem' }}>
                    <div className="diagnostic-card panel--live">
                      <span>Status</span>
                      <strong>{livingLensStatus}</strong>
                    </div>
                    <div className="diagnostic-card panel--stable">
                      <span>Prediction</span>
                      <strong>{currentSceneState ? currentSceneState.metrics.prediction_consistency.toFixed(2) : "n/a"}</strong>
                    </div>
                    <div className="diagnostic-card panel--persistence">
                      <span>Persistence</span>
                      <strong>{currentSceneState ? currentSceneState.metrics.persistence_confidence.toFixed(2) : "n/a"}</strong>
                    </div>
                  </div>
                </article>
              </div>

              <div key="understanding">
                <article className="panel panel--stable living-summary-panel" style={{ height: '100%', margin: 0, display: 'flex', flexDirection: 'column' }}>
                  <div className="panel-head">
                    <h3>Live Understanding</h3>
                  </div>
                  <div style={{ flex: 1, overflow: 'auto' }}>
                    {livingObservation ? (
                      <div className="stack">
                        <div className="observation-card">
                          <ObservationThumbnail src={assetUrl(livingObservation.thumbnail_path)} alt={livingObservation.id} />
                          <div>
                            <p style={{ fontSize: '0.9rem' }}>{livingAnswer}</p>
                            <div className="chips chips--stable">
                              <span>conf {livingObservation.confidence.toFixed(2)}</span>
                              <span>{livingObservation.providers[0]}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <p className="muted">Monitoring...</p>
                    )}
                  </div>
                </article>
              </div>

              <div key="pulse">
                <article className="panel panel--change living-summary-panel" style={{ height: '100%', margin: 0, display: 'flex', flexDirection: 'column' }}>
                  <div className="panel-head">
                    <h3>Scene Pulse</h3>
                  </div>
                  <div style={{ flex: 1, overflow: 'auto' }}>
                    {currentSceneState ? (
                      <div className="stack">
                        <div className="camera-health panel--stable" style={{ padding: '0.5rem' }}>
                          <small>Stable Anchors</small>
                          <div className="chips chips--stable">
                            {(continuitySignal?.stable_elements || []).slice(0, 3).map((item) => (
                              <span key={item}>{item}</span>
                            ))}
                          </div>
                        </div>
                        <div className="camera-health panel--change" style={{ padding: '0.5rem' }}>
                          <small>Changes</small>
                          <div className="chips chips--changed">
                            {(continuitySignal?.changed_elements || []).slice(0, 3).map((item) => (
                              <span key={item}>{item}</span>
                            ))}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <p className="muted">Waiting for state...</p>
                    )}
                  </div>
                </article>
              </div>

              <div key="matches">
                <article className="panel panel--persistence living-summary-panel" style={{ height: '100%', margin: 0, display: 'flex', flexDirection: 'column' }}>
                  <div className="panel-head">
                    <h3>Memory Relinking</h3>
                  </div>
                  <div className="stack" style={{ flex: 1, overflow: 'auto' }}>
                    {(livingLensResult?.hits || []).length > 0 ? (
                      livingLensResult!.hits.slice(0, 3).map((hit: any) => (
                        <div key={hit.observation_id} className="observation-card miniature" style={{ padding: '0.4rem', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', marginBottom: '4px' }}>
                           <span style={{ fontSize: '0.8rem' }}>{hit.observation_id.substring(0, 12)}</span>
                           <span className="accent-2" style={{ fontSize: '0.8rem', marginLeft: 'auto' }}>{(hit.score * 100).toFixed(0)}%</span>
                        </div>
                      ))
                    ) : (
                      <p className="muted">Searching history...</p>
                    )}
                  </div>
                </article>
              </div>
            </ResponsiveGridLayout>

            {livingSection === "overview" && (
              <div className="living-view-grid living-view-grid--overview">
                <article className="panel panel--stable living-panel-span-2">
                  <div className="panel-head">
                    <h3>Scene Delta</h3>
                    <span>{currentSceneState ? currentSceneState.id : "waiting"}</span>
                  </div>
                  {currentSceneState ? (
                    <div className="stack">
                      <div className="proof-split">
                        <div className="camera-health panel--stable">
                          <strong>Predicted state</strong>
                          <p>{currentSceneState.predicted_state_summary}</p>
                          <div className="chips chips--stable">
                            {currentSceneState.prediction_window.predicted_tags.map((item) => (
                              <span key={item}>{item}</span>
                            ))}
                          </div>
                        </div>
                        <div className="camera-health panel--change">
                          <strong>Observed state</strong>
                          <p>{currentSceneState.observed_state_summary}</p>
                          <div className="chips chips--changed">
                            {currentSceneState.observed_elements.map((item) => (
                              <span key={item}>{item}</span>
                            ))}
                          </div>
                        </div>
                      </div>
                      <div className="proof-split">
                        <div className="camera-health panel--stable">
                          <strong>What stayed stable</strong>
                          <div className="chips chips--stable">
                            {(continuitySignal?.stable_elements || []).map((item) => (
                              <span key={item}>{item}</span>
                            ))}
                            {!continuitySignal?.stable_elements?.length ? <span>No stable anchors yet</span> : null}
                          </div>
                        </div>
                        <div className="camera-health panel--change">
                          <strong>What changed</strong>
                          <div className="chips chips--changed">
                            {(continuitySignal?.changed_elements || []).map((item) => (
                              <span key={item}>{item}</span>
                            ))}
                            {!continuitySignal?.changed_elements?.length ? <span>Scene is currently stable</span> : null}
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p className="muted">Turn on automatic monitoring and point the lens at a scene to start building a temporal world model.</p>
                  )}
                </article>

                <article className="panel panel--comparison">
                  <div className="panel-head">
                    <h3>Metric Trend</h3>
                    <span>{challengeHistory.length} samples</span>
                  </div>
                  <p className="muted">Continuity should stay high through stable moments, surprise should spike on violations, and persistence should recover after occlusion.</p>
                  <ChallengeMetricChart history={challengeHistory} />
                </article>

                <article className="panel panel--persistence panel-scroll">
                  <div className="panel-head">
                    <h3>Persistence Snapshot</h3>
                    <span>{livingTracks.length} tracked threads</span>
                  </div>
                  <div className="stack scroll-stack">
                    <div className="camera-health panel--persistence">
                      <strong>Active persistence</strong>
                      <div className="chips chips--persist">
                        {(persistenceSignal?.visible_track_ids || []).map((item) => (
                          <span key={item}>visible {item.slice(-4)}</span>
                        ))}
                        {(persistenceSignal?.recovered_track_ids || []).map((item) => (
                          <span key={item}>recovered {item.slice(-4)}</span>
                        ))}
                        {(persistenceSignal?.occluded_track_ids || []).map((item) => (
                          <span key={item}>occluded {item.slice(-4)}</span>
                        ))}
                      </div>
                    </div>
                    {featuredTracks.map((track) => (
                      <div key={track.id} className="track-card" data-status={track.status}>
                        <div>
                          <strong>{track.label}</strong>
                          <p>{track.status}</p>
                        </div>
                        <div className="chips chips--persist">
                          <span>continuity {track.continuity_score.toFixed(2)}</span>
                          <span>persistence {track.persistence_confidence.toFixed(2)}</span>
                          <span>re-id {track.reidentification_count}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </article>
              </div>
            )}

            {livingSection === "memory" && (
              <div className="living-view-grid living-view-grid--memory">
                <article className="panel panel--memory">
                  <div className="panel-head">
                    <h3>Continuity Memory</h3>
                    <span>{livingNearest ? "nearest scene" : "waiting"}</span>
                  </div>
                  {livingNearest ? (
                    <div className="stack">
                      <div className="observation-card">
                        <ObservationThumbnail src={assetUrl(livingNearest.thumbnail_path)} alt={livingNearest.observation_id} />
                        <div>
                          <strong>{livingNearest.observation_id}</strong>
                          <p>{livingNearest.summary || "Nearest continuity match"}</p>
                          <div className="chips chips--persist">
                            <span>match {livingNearest.score.toFixed(2)}</span>
                            {livingObservation ? <span>current novelty {livingObservation.novelty.toFixed(2)}</span> : null}
                          </div>
                        </div>
                      </div>
                      <p className="muted">
                        This is the strongest remembered scene anchor for the current live state. Use it to see whether Living Lens is reconnecting to prior context instead of describing each frame from scratch.
                      </p>
                    </div>
                  ) : (
                    <p className="muted">Continuity Memory will populate once the live scene has at least one relevant prior observation.</p>
                  )}
                </article>

                <article className="panel panel--persistence panel-scroll">
                  <div className="panel-head">
                    <div>
                      <h3>Entity Tracks</h3>
                      <p className="muted">Persistent threads across visibility, occlusion, recovery, and re-identification.</p>
                    </div>
                    <span>{livingTracks.length} tracks</span>
                  </div>
                  <div className="stack scroll-stack">
                    {displayedTracks.map((track) => (
                      <div key={track.id} className="track-card" data-status={track.status}>
                        <div>
                          <strong>{track.label}</strong>
                          <p>{track.status}</p>
                        </div>
                        <div className="chips chips--persist">
                          <span>continuity {track.continuity_score.toFixed(2)}</span>
                          <span>persistence {track.persistence_confidence.toFixed(2)}</span>
                          <span>re-id {track.reidentification_count}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                  {sortedTracks.length > 8 ? (
                    <button type="button" className="secondary-link" onClick={() => setShowAllTracks((value) => !value)}>
                      {showAllTracks ? "Show fewer tracks" : `Show all ${sortedTracks.length} tracks`}
                    </button>
                  ) : null}
                </article>

                <article className="panel panel--memory panel-scroll">
                  <div className="panel-head">
                    <h3>Related Memory Matches</h3>
                    <span>{livingMatches.length} scenes</span>
                  </div>
                  <div className="stack scroll-stack">
                    {livingMatches.length ? livingMatches.map((hit) => (
                      <div key={hit.observation_id} className="list-row">
                        <ObservationThumbnail src={assetUrl(hit.thumbnail_path)} alt={hit.observation_id} />
                        <div>
                          <strong>{hit.observation_id}</strong>
                          <p>{hit.summary || "No summary"}</p>
                        </div>
                        <span>{hit.score.toFixed(2)}</span>
                      </div>
                    )) : <p className="muted">No related memory scenes yet. Continue monitoring to build continuity anchors.</p>}
                  </div>
                </article>
              </div>
            )}

            {livingSection === "challenge" && (
              <div className="living-view-grid living-view-grid--challenge">
                <article className="panel panel--challenge">
                  <div className="panel-head">
                    <h3>Guided JEPA Challenge</h3>
                    <span>{challengeGuideActive ? `step ${challengeStepIndex + 1} / ${challengeSteps.length}` : "ready"}</span>
                  </div>
                  <p className="muted">
                    Run one guided occlusion-and-change sequence. The app compares JEPA / Hybrid mode against frame captioning and embedding retrieval on the same session window.
                  </p>
                  <div className="camera-actions">
                    <button onClick={startGuidedChallenge} disabled={challengeBusy}>
                      {challengeGuideActive ? "Guided run active" : "Start Guided Run"}
                    </button>
                    <button onClick={advanceGuidedChallenge} disabled={!challengeGuideActive || challengeStepIndex >= challengeSteps.length - 1}>
                      Next Step
                    </button>
                    <button onClick={resetGuidedChallenge} disabled={!challengeGuideActive}>
                      Reset
                    </button>
                    <button onClick={() => runChallengeEvaluation("live")} disabled={challengeBusy}>
                      {challengeBusy ? "Scoring" : "Score Live Sequence"}
                    </button>
                    <button onClick={() => runChallengeEvaluation("curated")} disabled={challengeBusy}>
                      Score Stored Window
                    </button>
                  </div>
                  <div className="challenge-stepper">
                    {challengeSteps.map((step, index) => (
                      <div
                        key={step}
                        className={
                          index === challengeStepIndex && challengeGuideActive
                            ? "challenge-step active"
                            : index < challengeStepIndex && challengeGuideActive
                              ? "challenge-step complete"
                              : "challenge-step"
                        }
                      >
                        <span>Step {index + 1}</span>
                        <strong>{step}</strong>
                      </div>
                    ))}
                  </div>
                  <div className="camera-health panel--challenge challenge-current">
                    <strong>
                      {challengeGuideActive
                        ? `Current instruction: step ${challengeStepIndex + 1} of ${challengeSteps.length}`
                        : "Current instruction"}
                    </strong>
                    <p>{currentChallengeStep || challengeSteps[0]}</p>
                    <p className="muted">{challengeStepExpectation(challengeGuideActive ? challengeStepIndex : 0)}</p>
                  </div>
                </article>

                <article className="panel panel--comparison">
                  <div className="panel-head">
                    <h3>Baseline Scoreboard</h3>
                    <span>{livingBaseline ? "scored" : "waiting"}</span>
                  </div>
                  {livingBaseline ? (
                    <BaselineScoreboard comparison={livingBaseline} />
                  ) : (
                    <p className="muted">Run the guided challenge or score a stored window to compare JEPA / Hybrid mode with the baselines.</p>
                  )}
                </article>

                <article className="panel panel--challenge">
                  <div className="panel-head">
                    <h3>Scientific Readout</h3>
                    <span>{challengeRun ? challengeRun.id : "no run yet"}</span>
                  </div>
                  <div className="stack">
                    <div className="camera-health panel--comparison">
                      <strong>Challenge metric trend</strong>
                      <p>Continuity should survive stable periods, surprise should rise when a violation is introduced, and persistence should recover the same track after occlusion.</p>
                      <ChallengeMetricChart history={challengeHistory} />
                    </div>
                    {challengeRun ? (
                      <div className="camera-health panel--challenge">
                        <strong>{challengeRun.summary}</strong>
                        <div className="chips chips--challenge">
                          {Object.entries(challengeRun.success_criteria).map(([key, value]) => (
                            <span key={key}>{value ? "pass" : "watch"} {key.replace(/_/g, " ")}</span>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <p className="muted">Scientific scoring will appear here after the first live or stored challenge evaluation.</p>
                    )}
                  </div>
                </article>
              </div>
            )}
          </section>
          </>
        )}

        {activeTab === "Memory Search" && (
          <section className="panel-grid memory-grid">
            <article className="panel">
              <div className="panel-head">
                <h3>Semantic Search</h3>
                <span>Search across stored observations</span>
              </div>
              <label className="field">
                <span>Search prompt</span>
                <input
                  placeholder="desk mug, hallway motion, screen state..."
                  value={searchText}
                  onChange={(event) => setSearchText(event.target.value)}
                />
              </label>
              {queryResult?.answer ? (
                <div className="answer-box">
                  <p className="eyebrow">Answer</p>
                  <p>{queryResult.answer.text}</p>
                </div>
              ) : null}
              {renderReasoningTrace(queryResult?.reasoning_trace || [])}
            </article>
            <article className="panel wide panel-scroll">
              <div className="panel-head">
                <h3>Results</h3>
                <span>{queryResult?.hits.length || 0} matches</span>
              </div>
              <div className="stack scroll-stack">
                {(queryResult?.hits || []).map((hit) => (
                  <div key={hit.observation_id} className="list-row">
                    <ObservationThumbnail src={assetUrl(hit.thumbnail_path)} alt={hit.observation_id} />
                    <div>
                      <strong>{hit.summary || hit.observation_id}</strong>
                      <p>{new Date(hit.created_at).toLocaleString()}</p>
                    </div>
                    <span>{hit.score.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </article>
          </section>
        )}

        {activeTab === "Session Replay" && (
          <section className="panel panel-scroll">
            <div className="panel-head">
              <h3>Observation Timeline</h3>
              <span>{observations.length} frames captured</span>
            </div>
            <div className="timeline">
              {observations.map((observation) => (
                <article key={observation.id} className="timeline-card">
                  <ObservationThumbnail src={assetUrl(observation.thumbnail_path)} alt={observation.id} />
                  <div>
                    <p className="muted">{new Date(observation.created_at).toLocaleString()}</p>
                    <h4>{observation.summary || observation.id}</h4>
                    <p>{observation.source_query || "Auto-captured lens observation"}</p>
                    <div className="chips">
                      <span>{observation.providers.join(" + ")}</span>
                      <span>novelty {observation.novelty.toFixed(2)}</span>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          </section>
        )}

        {activeTab === "Integrations" && (
          <section className="panel-grid">
            <article className="panel">
              <div className="panel-head">
                <h3>Runtime Endpoints</h3>
                <span>Loopback by default</span>
              </div>
              <pre className="code-block">{`POST http://127.0.0.1:7777/v1/analyze
POST http://127.0.0.1:7777/v1/living-lens/tick
POST http://127.0.0.1:7777/v1/query
POST http://127.0.0.1:7777/v1/challenges/evaluate
GET  http://127.0.0.1:7777/v1/world-state
GET  http://127.0.0.1:7777/v1/settings
PUT  http://127.0.0.1:7777/v1/settings
GET  http://127.0.0.1:7777/v1/providers/health
WS   ws://127.0.0.1:7777/v1/events`}</pre>
            </article>
            <article className="panel">
              <div className="panel-head">
                <h3>Plugin Pattern</h3>
                <span>Any host app</span>
              </div>
              <pre className="code-block">{`const hit = await toori.query({
  query: "Where did I last see the blue notebook?",
  session_id: "desktop-live",
  top_k: 5
});`}</pre>
              <p className="muted">
                Generated SDKs live under `sdk/` and target TypeScript, Python, Swift, and Kotlin.
              </p>
            </article>
            <article className="panel">
              <div className="panel-head">
                <h3>Provider Health</h3>
                <span>Fallback order visibility</span>
              </div>
              <div className="stack">
                {health.map((item) => (
                  <div key={item.name} className="list-row compact">
                    <div>
                      <strong>{item.name}</strong>
                      <p>{item.message}</p>
                    </div>
                    <span>{item.healthy ? "ready" : "fallback"}</span>
                  </div>
                ))}
              </div>
              {renderReasoningTrace(analysis?.reasoning_trace || queryResult?.reasoning_trace || [])}
            </article>
          </section>
        )}

        {activeTab === "Settings" && settings && (
          <form className="panel-grid" onSubmit={saveSettings}>
            <article className="panel">
              <div className="panel-head">
                <h3>Runtime</h3>
                <span>Session and capture policy</span>
              </div>
              <label className="field">
                <span>Runtime profile</span>
                <input
                  value={settings.runtime_profile}
                  onChange={(event) => mutateSetting(["runtime_profile"], event.target.value)}
                />
              </label>
              <label className="field">
                <span>Camera device</span>
                <input
                  value={settings.camera_device}
                  onChange={(event) => mutateSetting(["camera_device"], event.target.value)}
                />
              </label>
              <label className="field">
                <span>Theme</span>
                <select
                  value={settings.theme_preference || "system"}
                  onChange={(event) => mutateSetting(["theme_preference"], event.target.value)}
                >
                  <option value="system">system</option>
                  <option value="dark">dark</option>
                  <option value="light">light</option>
                </select>
              </label>
              <label className="field">
                <span>Sampling FPS</span>
                <input
                  type="number"
                  min="0.2"
                  max="6"
                  step="0.1"
                  value={settings.sampling_fps}
                  onChange={(event) => mutateSetting(["sampling_fps"], Number(event.target.value))}
                />
              </label>
              <label className="field">
                <span>Top K</span>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={settings.top_k}
                  onChange={(event) => mutateSetting(["top_k"], Number(event.target.value))}
                />
              </label>
              <label className="field">
                <span>Retention days</span>
                <input
                  type="number"
                  min="1"
                  max="365"
                  value={settings.retention_days}
                  onChange={(event) => mutateSetting(["retention_days"], Number(event.target.value))}
                />
              </label>
            </article>

            <article className="panel">
              <div className="panel-head">
                <h3>Providers</h3>
                <span>Perception and reasoning routing</span>
              </div>
              <label className="field">
                <span>Primary perception</span>
                <select
                  value={settings.primary_perception_provider}
                  onChange={(event) => mutateSetting(["primary_perception_provider"], event.target.value)}
                >
                  <option value="onnx">onnx</option>
                  <option value="basic">basic fallback</option>
                </select>
              </label>
              <label className="field">
                <span>Reasoning backend</span>
                <select
                  value={settings.reasoning_backend}
                  onChange={(event) => mutateSetting(["reasoning_backend"], event.target.value)}
                >
                  <option value="cloud">cloud only</option>
                  <option value="ollama">ollama then cloud</option>
                  <option value="mlx">mlx then cloud</option>
                  <option value="auto">auto local fallback</option>
                  <option value="disabled">disabled</option>
                </select>
              </label>
              <label className="field checkbox">
                <input
                  type="checkbox"
                  checked={settings.local_reasoning_disabled}
                  onChange={(event) => mutateSetting(["local_reasoning_disabled"], event.target.checked)}
                />
                <span>Disable local reasoning by default</span>
              </label>
              <label className="field">
                <span>ONNX model path</span>
                <input
                  value={settings.providers.onnx.model_path || ""}
                  onChange={(event) => mutateSetting(["providers", "onnx", "model_path"], event.target.value)}
                />
              </label>
              <label className="field checkbox">
                <input
                  type="checkbox"
                  checked={settings.providers.onnx.enabled}
                  onChange={(event) => mutateProviderEnabled("onnx", event.target.checked)}
                />
                <span>Enable ONNX perception</span>
              </label>
              <label className="field">
                <span>Ollama host</span>
                <input
                  value={settings.providers.ollama.base_url || ""}
                  onChange={(event) => mutateSetting(["providers", "ollama", "base_url"], event.target.value)}
                />
              </label>
              <label className="field">
                <span>Ollama model</span>
                <input
                  value={settings.providers.ollama.model || ""}
                  onChange={(event) => mutateSetting(["providers", "ollama", "model"], event.target.value)}
                />
              </label>
              <label className="field">
                <span>Ollama timeout (s)</span>
                <input
                  type="number"
                  min="10"
                  max="600"
                  value={settings.providers.ollama.timeout_s}
                  onChange={(event) => mutateSetting(["providers", "ollama", "timeout_s"], Number(event.target.value))}
                />
              </label>
              <label className="field checkbox">
                <input
                  type="checkbox"
                  checked={settings.providers.ollama.enabled}
                  onChange={(event) => mutateProviderEnabled("ollama", event.target.checked)}
                />
                <span>Enable Ollama local reasoning</span>
              </label>
            </article>

            <article className="panel">
              <div className="panel-head">
                <h3>Cloud and MLX</h3>
                <span>Fallback targets</span>
              </div>
              <label className="field">
                <span>Cloud base URL</span>
                <input
                  value={settings.providers.cloud.base_url || ""}
                  onChange={(event) => mutateSetting(["providers", "cloud", "base_url"], event.target.value)}
                />
              </label>
              <label className="field">
                <span>Cloud model</span>
                <input
                  value={settings.providers.cloud.model || ""}
                  onChange={(event) => mutateSetting(["providers", "cloud", "model"], event.target.value)}
                />
              </label>
              <label className="field">
                <span>Cloud API key</span>
                <input
                  type="password"
                  autoComplete="off"
                  value={settings.providers.cloud.api_key || ""}
                  onChange={(event) => mutateSetting(["providers", "cloud", "api_key"], event.target.value)}
                />
                <small className="field-hint">Stored in local Toori runtime settings on this machine.</small>
              </label>
              <p className="field-hint">
                Each cloud reasoning call sends one image, your prompt, and up to five recent memory summaries.
                Keep cloud fallback disabled to spend zero cloud tokens during local capture.
              </p>
              <label className="field checkbox">
                <input
                  type="checkbox"
                  checked={settings.providers.cloud.enabled}
                  onChange={(event) => mutateProviderEnabled("cloud", event.target.checked)}
                />
                <span>Enable cloud fallback</span>
              </label>
              <label className="field">
                <span>MLX model path</span>
                <input
                  value={settings.providers.mlx.model_path || ""}
                  onChange={(event) => mutateSetting(["providers", "mlx", "model_path"], event.target.value)}
                />
              </label>
              <label className="field">
                <span>MLX command</span>
                <input
                  value={settings.providers.mlx.metadata.command || ""}
                  onChange={(event) => mutateSetting(["providers", "mlx", "metadata", "command"], event.target.value)}
                />
              </label>
              <label className="field">
                <span>MLX timeout (s)</span>
                <input
                  type="number"
                  min="10"
                  max="600"
                  value={settings.providers.mlx.timeout_s}
                  onChange={(event) => mutateSetting(["providers", "mlx", "timeout_s"], Number(event.target.value))}
                />
              </label>
              <label className="field checkbox">
                <input
                  type="checkbox"
                  checked={settings.providers.mlx.enabled}
                  onChange={(event) => mutateProviderEnabled("mlx", event.target.checked)}
                />
                <span>Enable MLX local reasoning</span>
              </label>
              <button className="primary" type="submit" disabled={savingSettings}>
                {savingSettings ? "Saving" : "Save Settings"}
              </button>
            </article>
          </form>
        )}
      </main>
    </div>
  );
}
