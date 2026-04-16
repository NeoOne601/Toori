import {
  createContext,
  startTransition,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { DEFAULT_CHALLENGE_STEPS, type AppTab, type LivingSection } from "../constants";
import { useCameraStream } from "../hooks/useCameraStream";
import { useLivingLens } from "../hooks/useLivingLens";
import { useRuntimeBridge } from "../hooks/useRuntimeBridge";
import { useWorldState } from "../hooks/useWorldState";
import {
  consumerMessage,
  explainSummary,
  explainWorldModel,
  boxesFromAnchorMatches,
  formatEntityBaseLabel,
  humanizeLabel,
  isPlaceholderVisionLabel,
  normalizeBoxes,
  toEnergyAnchors,
  toForecastValue,
  toGhostBoxes,
} from "../lib/formatting";
import type {
  AnalyzeResponse,
  BoundingBox,
  ChallengeRun,
  ConsumerGraphNode,
  EntityTrack,
  LivingLensTickResponse,
  Observation,
  SceneState,
} from "../types";

function isGenericEntityLabel(label: string | null | undefined) {
  return isPlaceholderVisionLabel(label);
}

function resolveDepthStratum(value?: string | null) {
  const normalized = humanizeLabel(value || "").toLowerCase().trim();
  if (normalized === "foreground" || normalized === "midground" || normalized === "background") {
    return normalized;
  }
  return "unresolved";
}

function sceneLabelCandidates(sceneState?: SceneState | null): string[] {
  const metadata = (sceneState?.metadata || {}) as Record<string, unknown>;
  const candidates = metadata.summary_candidates;
  if (!Array.isArray(candidates)) {
    return [];
  }
  return candidates
    .map((item) => humanizeLabel(String(item || "")))
    .filter((item) => item && !isGenericEntityLabel(item));
}

function createDesktopSessionId() {
  const storageKey = "toori_live_session_id";
  try {
    const existing = window.sessionStorage.getItem(storageKey);
    if (existing) {
      return existing;
    }
    const created = `desktop-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    window.sessionStorage.setItem(storageKey, created);
    return created;
  } catch {
    return `desktop-${Date.now().toString(36)}`;
  }
}

function normalizedTrackCenter(track: EntityTrack, observation?: Observation | null) {
  const metadata = (track.metadata || {}) as Record<string, unknown>;
  const bbox = (metadata.bbox_pixels || metadata.ghost_bbox_pixels || metadata.bbox) as
    | Record<string, unknown>
    | undefined;
  if (!bbox) {
    return null;
  }
  const x = Number(bbox.x);
  const y = Number(bbox.y);
  const width = Number(bbox.width || 0);
  const height = Number(bbox.height || 0);
  if (![x, y, width, height].every(Number.isFinite)) {
    return null;
  }
  const observationWidth = Math.max(observation?.width || 0, 1);
  const observationHeight = Math.max(observation?.height || 0, 1);
  const normalizedX = width > 1 || height > 1 ? (x + width / 2) / observationWidth : x + width / 2;
  const normalizedY = width > 1 || height > 1 ? (y + height / 2) / observationHeight : y + height / 2;
  return {
    x: Math.max(0, Math.min(normalizedX, 1)),
    y: Math.max(0, Math.min(normalizedY, 1)),
  };
}

function proposalLabelForTrack(
  track: EntityTrack,
  proposalBoxes: BoundingBox[],
  observation?: Observation | null,
) {
  const center = normalizedTrackCenter(track, observation);
  if (!center) {
    return "";
  }
  let bestLabel = "";
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const box of proposalBoxes) {
    const label = humanizeLabel(box.label);
    if (!label || isGenericEntityLabel(label)) {
      continue;
    }
    const boxCenterX = box.x + box.width / 2;
    const boxCenterY = box.y + box.height / 2;
    const dx = center.x - boxCenterX;
    const dy = center.y - boxCenterY;
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestLabel = label;
    }
  }
  return bestDistance <= 0.22 ? bestLabel : "";
}

function boxesFromTracks(
  tracks: EntityTrack[],
  proposalBoxes: BoundingBox[],
  observation?: Observation | null,
): BoundingBox[] {
  const width = Math.max(observation?.width || 1, 1);
  const height = Math.max(observation?.height || 1, 1);
  return tracks.flatMap((track) => {
    const metadata = (track.metadata || {}) as Record<string, unknown>;
    const bbox = (metadata.bbox_pixels || metadata.ghost_bbox_pixels || metadata.bbox) as
      | Record<string, unknown>
      | undefined;
    if (!bbox) {
      return [];
    }
    const x = Number(bbox.x);
    const y = Number(bbox.y);
    const boxWidth = Number(bbox.width || 0);
    const boxHeight = Number(bbox.height || 0);
    if (![x, y, boxWidth, boxHeight].every(Number.isFinite)) {
      return [];
    }
    const normalized =
      boxWidth > 1 || boxHeight > 1
        ? {
            x: x / width,
            y: y / height,
            width: boxWidth / width,
            height: boxHeight / height,
          }
        : {
            x,
            y,
            width: boxWidth,
            height: boxHeight,
          };
    const label = proposalLabelForTrack(track, proposalBoxes, observation) || formatEntityBaseLabel(track);
    return [
      {
        x: Math.max(0, Math.min(normalized.x, 0.96)),
        y: Math.max(0, Math.min(normalized.y, 0.96)),
        width: Math.max(0.04, Math.min(normalized.width, 1 - normalized.x)),
        height: Math.max(0.04, Math.min(normalized.height, 1 - normalized.y)),
        label,
        score: track.last_similarity || track.persistence_confidence || null,
        metadata: {
          ...metadata,
          track_id: track.id,
          track_status: track.status,
        },
      },
    ];
  });
}

function mergeConsumerNodes(nodes: ConsumerGraphNode[]) {
  const merged = new Map<string, ConsumerGraphNode & { count: number }>();
  for (const node of nodes) {
    const key = humanizeLabel(node.label).toLowerCase() || node.id;
    const current = merged.get(key);
    if (!current) {
      merged.set(key, { ...node, count: 1 });
      continue;
    }
    const count = current.count + 1;
    merged.set(key, {
      ...current,
      x: ((current.x * current.count) + node.x) / count,
      y: ((current.y * current.count) + node.y) / count,
      radius: Math.max(current.radius || 0, node.radius || 0),
      tone: current.tone === "memory" || node.tone === "memory" ? "memory" : current.tone,
      count,
    });
  }
  return Array.from(merged.values()).map(({ count: _count, ...node }) => node);
}

function stableNodeId(prefix: string, x: number, y: number, fallback: string) {
  return `${prefix}-${Math.round(x * 10)}-${Math.round(y * 10)}-${fallback}`;
}

function hasVisibleHeatmapEnergy(energyMap?: number[][] | null) {
  if (!energyMap?.length) {
    return false;
  }
  const values = energyMap.flat().map((value) => Number(value) || 0);
  if (!values.length) {
    return false;
  }
  const maxEnergy = Math.max(...values, 0);
  if (maxEnergy <= 0) {
    return false;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const p55 = sorted[Math.floor(sorted.length * 0.55)] ?? 0;
  return values.some((value) => value >= Math.max(p55, maxEnergy * 0.18));
}

function resolveConsumerTrackLabel(
  track: EntityTrack,
  index: number,
  sceneState?: SceneState | null,
  atlasNode?: Record<string, unknown> | null,
  proposalLabel?: string,
) {
  const metadata = (track.metadata || {}) as Record<string, unknown>;
  const sceneCandidates = sceneLabelCandidates(sceneState);
  
  // SEMANTIC PRIORITIZATION:
  // 1. Backend-assigned Primary Label (TVLC result)
  // 2. Atlas/Graph Node Link
  // 3. Scene Context/Gemma Alert
  // 4. Trace Metadata
  const candidates = [
    humanizeLabel(String(metadata.primary_object_label || "")),
    humanizeLabel(String(atlasNode?.label || "")),
    sceneCandidates[index] || "",
    humanizeLabel(String(metadata.top_label || "")),
    humanizeLabel(String(metadata.caption || "")),
    humanizeLabel(String((sceneState?.metadata as Record<string, unknown> | undefined)?.primary_object_label || "")),
    humanizeLabel(proposalLabel || ""),
  ].filter((item) => item && !isGenericEntityLabel(item));

  const similarity = (metadata.label_evidence as any)?.conf ?? (track.last_similarity ?? 0);
  const fallbackLabel = formatEntityBaseLabel(track);
  const picked = candidates[0] || fallbackLabel;
  
  if (!picked || isGenericEntityLabel(picked)) {
      // If we are unresolved, show a numbered entity to allow referencing in prompts
      return `Entity ${index + 1}`;
  }
  
  // Show confidence score only for meaningful semantic labels
  const confidence = (similarity > 0 && !isPlaceholderVisionLabel(picked)) 
    ? ` ${Math.round(similarity * 100)}%` 
    : "";
    
  return `${picked}${confidence}`;
}

function useDesktopAppValue() {
  const [activeTab, setActiveTab] = useState<AppTab>("Living Lens");
  const [livingSection, setLivingSection] = useState<LivingSection>("overview");
  const [prompt, setPrompt] = useState("");
  const [searchText, setSearchText] = useState("");
  const [showEnergyMap, setShowEnergyMap] = useState(true);
  const [showEntities, setShowEntities] = useState(true);
  const [showAllTracks, setShowAllTracks] = useState(false);
  const [challengeGuideActive, setChallengeGuideActive] = useState(false);
  const [challengeStepIndex, setChallengeStepIndex] = useState(0);
  const [sessionId] = useState(() => createDesktopSessionId());
  const [uiMode, setUiMode] = useState<"consumer" | "science">(() => {
    const stored = window.localStorage.getItem("toori_mode");
    if (stored === "science" || stored === "consumer") {
      return stored;
    }
    window.localStorage.setItem("toori_mode", "consumer");
    return "consumer";
  });
  const overlaySourceRef = useRef<{ kind: "anchor" | "track" | "proposal"; at: number }>({
    kind: "proposal",
    at: 0,
  });
  const lastAnchorBoxesRef = useRef<BoundingBox[]>([]);
  const lastTrackBoxesRef = useRef<BoundingBox[]>([]);
  const { assetUrl, pickImagePayload, runtimeRequest } = useRuntimeBridge();
  const world = useWorldState({
    sessionId,
    searchText,
    runtimeRequest,
  });
  const camera = useCameraStream({
    activeTab,
    settings: world.settings,
    runtimeRequest,
    onStatusChange: world.setStatus,
    onSettingsChange: world.setSettings,
  });
  const livingLens = useLivingLens({
    activeTab,
    cameraStreamLive: camera.cameraStreamLive,
    cameraReady: camera.cameraReady,
    sessionId,
    topK: world.settings?.top_k || 6,
    currentFrameBase64: camera.currentFrameBase64,
    runtimeRequest,
    onHealthChange: world.setHealth,
    onResult: (result) => {
      world.setAnalysis(result);
      world.setHealth(result.provider_health);
      world.setLatestJepaTick(result.jepa_tick || null);
      world.setWorldState((current) => {
        const nextHistory = [
          result.scene_state,
          ...(current?.history || []).filter((state) => state.id !== result.scene_state.id),
        ];
        return {
          session_id: current?.session_id || sessionId,
          current: result.scene_state,
          history: nextHistory,
          entity_tracks: result.entity_tracks,
          challenges: current?.challenges || [],
          benchmarks: current?.benchmarks || [],
          atlas: current?.atlas || null,
        };
      });
    },
    onError: (error) => {
      world.setLatestJepaTick(null);
      world.setStatus(`Living Lens tick failed: ${error.message}`);
    },
    onRefresh: world.refreshHot,
  });

  useEffect(() => {
    window.localStorage.setItem("toori_mode", uiMode);
  }, [uiMode]);

  useEffect(() => {
    const liveFeatures = world.settings?.live_features;
    if (!liveFeatures) {
      return;
    }
    setShowEnergyMap(Boolean(liveFeatures.energy_heatmap_enabled ?? true));
    setShowEntities(Boolean(liveFeatures.entity_overlay_enabled ?? true));
  }, [
    world.settings?.live_features?.energy_heatmap_enabled,
    world.settings?.live_features?.entity_overlay_enabled,
  ]);

  useEffect(() => {
    if (activeTab !== "Living Lens" || livingLens.livingLensEnabled) {
      return;
    }
    world.setLatestJepaTick(null);
  }, [activeTab, livingLens.livingLensEnabled, world.setLatestJepaTick]);

  async function runLiveLensTickForPayload(
    body: Record<string, unknown>,
  ): Promise<LivingLensTickResponse> {
    const result = await runtimeRequest<LivingLensTickResponse>("/v1/living-lens/tick", "POST", {
      session_id: sessionId,
      top_k: world.settings?.top_k || 6,
      proof_mode: "both",
      ...body,
    });
    world.setAnalysis(result);
    world.setHealth(result.provider_health);
    world.setLatestJepaTick(result.jepa_tick || null);
    livingLens.setLivingLensResult(result);
    livingLens.setLivingLensLastSuccessAt(Date.now());
    livingLens.setLivingLensLastError(null);
    world.setWorldState((current) => {
      const nextHistory = [
        result.scene_state,
        ...(current?.history || []).filter((state) => state.id !== result.scene_state.id),
      ];
      return {
        session_id: current?.session_id || sessionId,
        current: result.scene_state,
        history: nextHistory,
        entity_tracks: result.entity_tracks,
        challenges: current?.challenges || [],
        benchmarks: current?.benchmarks || [],
        atlas: current?.atlas || null,
      };
    });
    startTransition(() => {
      world.refreshHot().catch(() => undefined);
    });
    return result;
  }

  async function analyzeCurrentFrame(options: {
    query?: string;
    decodeMode: "auto" | "force";
    topK?: number;
  }): Promise<AnalyzeResponse | null> {
    const imageBase64 = camera.currentFrameBase64("live");
    if (!imageBase64) {
      return null;
    }
    const result = await runtimeRequest<AnalyzeResponse>("/v1/analyze", "POST", {
      image_base64: imageBase64,
      session_id: sessionId,
      query: options.query || undefined,
      decode_mode: options.decodeMode,
      top_k: options.topK || world.settings?.top_k || 6,
    });
    world.setHealth(result.provider_health);
    startTransition(() => {
      world.refreshHot().catch(() => undefined);
    });
    return result;
  }

  async function captureFrame() {
    if (!camera.cameraReady) {
      world.setStatus("Video stream is not ready for capture");
      return;
    }
    world.setStatus("Analyzing current frame");
    try {
      const useJepaTick = world.settings?.live_features?.live_lens_use_jepa_tick !== false;
      const result = useJepaTick
        ? await runLiveLensTickForPayload({
            image_base64: camera.currentFrameBase64("live"),
            query: prompt || undefined,
            decode_mode: prompt.trim() ? "force" : "auto",
          })
        : await analyzeCurrentFrame({
            query: prompt || undefined,
            decodeMode: "auto",
            topK: world.settings?.top_k || 6,
          });
      if (!result) {
        world.setStatus("Video stream is not ready for capture");
        return;
      }
      world.setAnalysis(result);
      world.setStatus(`Captured ${result.observation.id}`);
    } catch (error) {
      world.setStatus((error as Error).message);
    }
  }

  async function analyzeFile() {
    const payload = await pickImagePayload();
    if (!payload) {
      return;
    }
    world.setStatus(payload.filePath ? `Analyzing ${payload.filePath}` : "Analyzing selected image");
    try {
      const useJepaTick = world.settings?.live_features?.live_lens_use_jepa_tick !== false;
      const result = useJepaTick
        ? await runLiveLensTickForPayload({
            file_path: payload.filePath,
            image_base64: payload.imageBase64,
            query: prompt || undefined,
            decode_mode: "force",
          })
        : await runtimeRequest<AnalyzeResponse>("/v1/analyze", "POST", {
            file_path: payload.filePath,
            image_base64: payload.imageBase64,
            session_id: sessionId,
            query: prompt || undefined,
            decode_mode: "force",
          });
      world.setAnalysis(result);
      world.setHealth(result.provider_health);
      startTransition(() => {
        world.refreshHot().catch(() => undefined);
      });
    } catch (error) {
      world.setStatus((error as Error).message);
    }
  }

  function startGuidedChallenge() {
    setChallengeGuideActive(true);
    setChallengeStepIndex(0);
    setUiMode("science");
    setLivingSection("challenge");
    livingLens.setLivingLensEnabled(true);
    if (!camera.cameraStreamLive) {
      camera.retryCamera(false).catch(() => undefined);
    }
    setActiveTab("Living Lens");
    livingLens.setLivingLensStatus("Guided challenge started. Follow the first instruction below.");
  }

  function advanceGuidedChallenge() {
    const steps = world.challengeRun?.guide_steps?.length
      ? world.challengeRun.guide_steps
      : DEFAULT_CHALLENGE_STEPS;
    setChallengeStepIndex((current) => {
      const next = Math.min(current + 1, steps.length - 1);
      if (next === steps.length - 1) {
        livingLens.setLivingLensStatus("Final guided step reached. Run scoring when you are ready.");
      }
      return next;
    });
  }

  function resetGuidedChallenge() {
    setChallengeGuideActive(false);
    setChallengeStepIndex(0);
    livingLens.setLivingLensStatus("Continuous monitoring is ready");
  }

  const livingResultFreshnessMs = Math.max(4_000, Math.max(2, livingLens.livingLensIntervalS) * 2_500);
  const isLivingLensTab = activeTab === "Living Lens";
  const liveAnalysisPaused = isLivingLensTab && !livingLens.livingLensEnabled;
  const freshLivingLensResult =
    livingLens.livingLensResult &&
    livingLens.livingLensLastSuccessAt != null &&
    Date.now() - livingLens.livingLensLastSuccessAt <= livingResultFreshnessMs
      ? livingLens.livingLensResult
      : null;
  const freshEventJepaTick =
    world.latestJepaTick &&
    Date.now() - Number(world.latestJepaTick.timestamp_ms || 0) <= livingResultFreshnessMs
      ? world.latestJepaTick
      : null;
  const prefersFreshLiveTick = isLivingLensTab && livingLens.livingLensEnabled;
  const liveTickStale =
    prefersFreshLiveTick &&
    livingLens.livingLensLastSuccessAt != null &&
    Date.now() - livingLens.livingLensLastSuccessAt > livingResultFreshnessMs;
  const runtimeAvailable = world.runtimeAvailable;
  const latestObservation =
    runtimeAvailable
      ? freshLivingLensResult?.observation || world.analysis?.observation || world.observations[0]
      : null;
  const latestObservationSummary = latestObservation
    ? explainSummary(
        latestObservation,
        world.analysis?.observation?.id === latestObservation.id
          ? world.analysis?.answer?.text
          : undefined,
      )
    : null;
  const readyProviders = runtimeAvailable ? world.health.filter((item) => item.healthy).length : 0;
  const degradedProviders = runtimeAvailable ? Math.max(world.health.length - readyProviders, 0) : 0;
  const topFeatureHealth = runtimeAvailable ? world.featureHealth.slice(0, 5) : [];
  const runtimeUnavailableLabel =
    world.eventsState === "connecting"
      ? "Connecting to runtime..."
      : world.eventsState === "reconnecting"
        ? "Runtime unreachable. Reconnecting..."
        : "Runtime unreachable";
  const runtimeConnectionLabel = runtimeAvailable ? "Runtime ready" : runtimeUnavailableLabel;
  const fallbackAnalysisJepaTick =
    !isLivingLensTab && runtimeAvailable
      ? (((world.analysis as AnalyzeResponse & { jepa_tick?: LivingLensTickResponse["jepa_tick"] })?.jepa_tick as
          | LivingLensTickResponse["jepa_tick"]
          | null
          | undefined) ?? null)
      : null;
  const currentJepaTick = runtimeAvailable
    ? freshLivingLensResult?.jepa_tick || freshEventJepaTick || fallbackAnalysisJepaTick
    : null;
  const hasLiveJepaEvidence = Boolean(currentJepaTick);
  const currentSceneState = runtimeAvailable
    ? freshLivingLensResult?.scene_state ||
      (isLivingLensTab
        ? hasLiveJepaEvidence
          ? world.worldState?.current || null
          : null
        : world.worldState?.current || null)
    : null;
  const livingTracks = runtimeAvailable
    ? freshLivingLensResult?.entity_tracks ||
      (isLivingLensTab
        ? hasLiveJepaEvidence
          ? world.worldState?.entity_tracks || []
          : []
        : world.worldState?.entity_tracks || [])
    : [];
  const livingObservation = runtimeAvailable
    ? freshLivingLensResult?.observation ||
      (isLivingLensTab
        ? hasLiveJepaEvidence
          ? latestObservation
          : null
        : latestObservation)
    : null;
  const livingAnswer = !runtimeAvailable
    ? runtimeUnavailableLabel
    : freshLivingLensResult
      ? explainSummary(
          livingObservation,
          freshLivingLensResult.answer?.text || currentSceneState?.observed_state_summary,
        )
      : liveAnalysisPaused
        ? "Auto analyze paused. Capture a frame to refresh the live scene."
        : prefersFreshLiveTick
          ? livingLens.livingLensLastError ||
            (liveTickStale ? "Live JEPA tick stalled" : livingLens.livingLensStatus)
          : explainSummary(
              livingObservation,
              currentSceneState?.observed_state_summary,
            );
  const livingNearest = runtimeAvailable ? freshLivingLensResult?.hits?.[0] || null : null;
  const livingMatches = runtimeAvailable ? freshLivingLensResult?.hits || [] : [];
  const livingBaseline = runtimeAvailable
    ? freshLivingLensResult?.baseline_comparison ||
      world.challengeRun?.baseline_comparison ||
      world.worldState?.challenges?.[0]?.baseline_comparison ||
      null
    : null;
  const worldHistory = runtimeAvailable ? world.worldState?.history || [] : [];
  const continuitySignal = runtimeAvailable ? currentSceneState?.metrics.continuity_signal : null;
  const persistenceSignal = runtimeAvailable ? currentSceneState?.metrics.persistence_signal : null;
  const liveObservationMetadata = runtimeAvailable
    ? ((world.analysis?.observation?.metadata as Record<string, unknown> | undefined) ||
        (latestObservation?.metadata as Record<string, unknown> | undefined))
    : undefined;
  const liveAnchorBoxes = runtimeAvailable
    ? boxesFromAnchorMatches(
        (currentJepaTick?.anchor_matches as Array<Record<string, unknown>> | null | undefined) || null,
      )
    : [];
  const useLiveAnchorBoxes =
    runtimeAvailable &&
    world.settings?.live_features?.live_lens_use_jepa_tick !== false &&
    Array.isArray(currentJepaTick?.anchor_matches) &&
    liveAnchorBoxes.length > 0;
  const proposalBoxes = normalizeBoxes(
    liveObservationMetadata?.object_proposals ||
      liveObservationMetadata?.proposal_boxes ||
      liveObservationMetadata?.bounding_boxes,
    world.analysis?.observation || latestObservation,
  );
  const hasLiveAnchorSource = runtimeAvailable && Array.isArray(currentJepaTick?.anchor_matches);
  const livingTrackBoxes = runtimeAvailable ? boxesFromTracks(livingTracks, proposalBoxes, livingObservation) : [];
  if (liveAnchorBoxes.length > 0) {
    lastAnchorBoxesRef.current = liveAnchorBoxes;
  }
  if (livingTrackBoxes.length > 0) {
    lastTrackBoxesRef.current = livingTrackBoxes;
  }
  const nowMs = Date.now();
  const stickyAnchorBoxes =
    overlaySourceRef.current.kind === "anchor" &&
    nowMs - overlaySourceRef.current.at < 1500 &&
    lastAnchorBoxesRef.current.length > 0
      ? lastAnchorBoxesRef.current
      : [];
  const stickyTrackBoxes =
    overlaySourceRef.current.kind === "track" &&
    nowMs - overlaySourceRef.current.at < 1200 &&
    lastTrackBoxesRef.current.length > 0
      ? lastTrackBoxesRef.current
      : [];
  const activeAnchorBoxes =
    useLiveAnchorBoxes && liveAnchorBoxes.length > 0
      ? liveAnchorBoxes
      : !useLiveAnchorBoxes && stickyAnchorBoxes.length > 0
        ? stickyAnchorBoxes
        : [];
  const activeTrackBoxes =
    activeAnchorBoxes.length > 0
      ? []
      : livingTrackBoxes.length > 0
        ? livingTrackBoxes
        : stickyTrackBoxes;
  if (activeAnchorBoxes.length > 0) {
    overlaySourceRef.current = { kind: "anchor", at: nowMs };
  } else if (activeTrackBoxes.length > 0) {
    overlaySourceRef.current = { kind: "track", at: nowMs };
  } else {
    overlaySourceRef.current = { kind: "proposal", at: nowMs };
  }
  const liveBoxes = activeAnchorBoxes.length > 0 ? activeAnchorBoxes : proposalBoxes;
  const livingBoxes =
    activeAnchorBoxes.length > 0
      ? activeAnchorBoxes
      : activeTrackBoxes.length > 0
        ? activeTrackBoxes
        : proposalBoxes;
  const challengeSteps = world.challengeRun?.guide_steps?.length
    ? world.challengeRun.guide_steps
    : Array.from(DEFAULT_CHALLENGE_STEPS);
  const currentChallengeStep = challengeGuideActive ? challengeSteps[challengeStepIndex] : null;
  const worldModelSummary = runtimeAvailable
    ? explainWorldModel(currentSceneState)
    : runtimeUnavailableLabel;
  const cameraHasActualFault =
    camera.cameraDiagnostics.blackFrameDetected ||
    camera.cameraDiagnostics.phase === "video stalled" ||
    camera.cameraDiagnostics.phase === "camera error";
  const cameraWarmingUp = camera.cameraStreamLive && !camera.cameraReady && !cameraHasActualFault;
  const cameraStatusLabel = camera.cameraReady
    ? "camera ready"
    : cameraHasActualFault
      ? "camera degraded"
      : cameraWarmingUp
        ? "camera warming up"
        : camera.cameraStreamLive
          ? "camera connected"
          : "camera idle";
  const cameraConnectionState =
    !camera.cameraAccess.granted && camera.cameraAccess.status !== "unknown"
      ? "blocked"
      : camera.cameraBusy
        ? "reconnecting"
        : cameraHasActualFault
          ? "degraded"
        : camera.cameraReady
          ? "live"
          : camera.cameraStreamLive
            ? "live"
            : "offline";
  const ghostBoxes = toGhostBoxes(livingTracks, livingObservation);
  const energyAnchors = runtimeAvailable ? toEnergyAnchors(currentJepaTick?.energy_map || null) : [];
  const missingTickStatus = liveAnalysisPaused
    ? "auto analyze paused"
    : !runtimeAvailable
      ? runtimeUnavailableLabel
    : prefersFreshLiveTick
    ? livingLens.livingLensBusy
      ? "awaiting live JEPA tick"
      : livingLens.livingLensLastError ||
        (liveTickStale ? "live JEPA tick stalled" : livingLens.livingLensStatus) ||
        "awaiting first JEPA tick"
    : "awaiting first JEPA tick";
  const energyHeatmapStatus = !showEnergyMap
    ? "heatmap disabled"
    : !runtimeAvailable
      ? runtimeUnavailableLabel
    : !currentJepaTick
      ? missingTickStatus
      : currentJepaTick.warmup
        ? "warmup active"
        : hasVisibleHeatmapEnergy(currentJepaTick.energy_map)
          ? null
          : "no high-energy patches this tick";
  const fe1 = toForecastValue(currentJepaTick, 1);
  const fe2 = toForecastValue(currentJepaTick, 2);
  const fe5 = toForecastValue(currentJepaTick, 5);
  const forecastMonotonic =
    [fe1, fe2, fe5].every((value) => value != null)
      ? (fe1 as number) < (fe2 as number) && (fe2 as number) < (fe5 as number)
      : true;
  const occlusionTracks = livingTracks.map((track) => ({
    id: track.id,
    label: track.label,
    status:
      track.status === "re-identified"
        ? "recovered"
        : track.status === "violated prediction"
          ? "violated"
          : track.status,
    confidence: track.persistence_confidence,
    note: `${String((track.metadata as Record<string, any> | undefined)?.duration_ms || 0)} ms`,
  }));
  const consumerSceneState = currentSceneState;
  const consumerTrackSourceBase =
    runtimeAvailable && livingTracks.length > 0
      ? livingTracks
      : runtimeAvailable && !isLivingLensTab
        ? world.worldState?.entity_tracks || []
        : [];
  const consumerTrackSource = consumerTrackSourceBase.filter(
    (track) => track.status !== "disappeared" && track.status !== "violated prediction",
  );
  const consumerGroundedNodes = (consumerSceneState?.grounded_entities || []).slice(0, 8).map((entity, index) => {
    const properties = (entity.properties || {}) as Record<string, any>;
    const bbox = properties.bbox as
      | { x?: number; y?: number; width?: number; height?: number }
      | undefined;
    const center = bbox
      ? {
          x: ((Number(bbox.x || 0) + Number(bbox.width || 0) / 2) * 100),
          y: ((Number(bbox.y || 0) + Number(bbox.height || 0) / 2) * 100),
        }
      : {
          x: 18 + ((index * 15) % 62),
          y: 20 + ((index * 12) % 58),
        };
    const groundedLabel = formatEntityBaseLabel({ label: entity.label, metadata: entity.properties });
    return {
      id: entity.id,
      label: !isGenericEntityLabel(groundedLabel)
        ? groundedLabel
        : (!isGenericEntityLabel(entity.kind) ? humanizeLabel(entity.kind || "") : "") || `Unresolved ${index + 1}`,
      x: Math.max(16, Math.min(center.x, 84)),
      y: Math.max(16, Math.min(center.y, 84)),
      radius: 11 + Math.min(Math.round((entity.confidence || 0.5) * 8), 8),
      tone:
        entity.state_domain === "memory"
          ? ("memory" as const)
          : entity.status === "occluded"
            ? ("memory" as const)
            : entity.status === "visible"
              ? ("live" as const)
              : ("stable" as const),
      depthStratum: "unresolved",
    };
  });
  const sceneGraphNodes = (world.sceneGraph?.nodes || []).slice(0, 16).map((node, index) => {
    const bbox = node.bbox;
    const center = bbox
      ? {
          x: ((Number(bbox.x || 0) + Number(bbox.width || 0) / 2) * 100),
          y: ((Number(bbox.y || 0) + Number(bbox.height || 0) / 2) * 100),
        }
      : {
          x: 18 + ((index * 13) % 64),
          y: 20 + ((index * 11) % 58),
        };
    const label = !isGenericEntityLabel(node.label)
      ? humanizeLabel(node.label || "")
      : `Unresolved ${index + 1}`;
    return {
      id: node.id,
      label,
      x: Math.max(16, Math.min(center.x, 84)),
      y: Math.max(16, Math.min(center.y, 84)),
      radius: 10 + Math.min(Math.round((node.confidence || 0.4) * 10), 10),
      tone:
        node.source === "grounded_entity"
          ? ("accent" as const)
          : node.status === "occluded"
            ? ("memory" as const)
            : node.source === "track"
              ? ("live" as const)
              : ("stable" as const),
      depthStratum: resolveDepthStratum(node.depth_stratum),
      source: node.source,
      confidence: node.confidence,
      status: node.status,
    } as ConsumerGraphNode;
  }).filter((node) => !isGenericEntityLabel(node.label));
  const consumerAtlasNodes = Array.isArray((world.worldState?.atlas as any)?.nodes)
    ? (world.worldState?.atlas as any).nodes
    : [];
  const consumerNodesFromAnchors = activeAnchorBoxes.slice(0, 8).map((box, index) => {
    const anchorLabel = formatEntityBaseLabel({ label: box.label, metadata: box.metadata });
    return {
      id:
        String((box.metadata as Record<string, unknown> | undefined)?.template_name || "")
          || stableNodeId("anchor", (box.x + box.width / 2) * 100, (box.y + box.height / 2) * 100, String(index)),
      label: !isGenericEntityLabel(anchorLabel) ? anchorLabel : `Anchor ${index + 1}`,
      x: Math.max(18, Math.min((box.x + box.width / 2) * 100, 82)),
      y: Math.max(18, Math.min((box.y + box.height / 2) * 100, 82)),
      radius: 11 + Math.min(Math.round((box.score || 0.55) * 10), 10),
      tone: "live" as const,
      depthStratum: "unresolved",
    };
  });
  const consumerNodesFromTracks = consumerTrackSource.slice(0, 8).map((track, index) => {
    const metadata = (track.metadata || {}) as Record<string, any>;
    const bbox = metadata.bbox_pixels || metadata.ghost_bbox_pixels || metadata.bbox;
    const width = Math.max(livingObservation?.width || 1, 1);
    const height = Math.max(livingObservation?.height || 1, 1);
    const atlasNode = consumerAtlasNodes.find(
      (node: any) => String(node.entity_id || node.id || "") === String(track.id),
    );
    const center =
      bbox && typeof bbox === "object" && Number.isFinite(Number(bbox.x)) && Number.isFinite(Number(bbox.y))
        ? {
            x:
              Number(bbox.width) > 1
                ? ((Number(bbox.x) + Number(bbox.width) / 2) / width) * 100
                : (Number(bbox.x) + Number(bbox.width || 0) / 2) * 100,
            y:
              Number(bbox.height) > 1
                ? ((Number(bbox.y) + Number(bbox.height) / 2) / height) * 100
                : (Number(bbox.y) + Number(bbox.height || 0) / 2) * 100,
          }
        : {
            x: 20 + ((index * 17) % 60),
            y: 25 + ((index * 13) % 50),
          };
    const proposalLabel = proposalLabelForTrack(track, livingBoxes, livingObservation);
    return {
      id: String(track.id),
      label: resolveConsumerTrackLabel(track, index, consumerSceneState, atlasNode, proposalLabel),
      x: Math.max(18, Math.min(center.x, 82)),
      y: Math.max(18, Math.min(center.y, 82)),
      radius: 10 + Math.min(track.visibility_streak || 1, 10),
      tone: track.status === "occluded" ? "memory" : track.status === "visible" ? "live" : "stable",
      depthStratum: "unresolved",
    };
  });
  const consumerBoxFallback = livingBoxes.slice(0, 6).map((box, index) => {
    const boxLabel = formatEntityBaseLabel({ label: box.label, metadata: box.metadata });
    return {
      id: stableNodeId("box", (box.x + box.width / 2) * 100, (box.y + box.height / 2) * 100, String(index)),
      label: !isGenericEntityLabel(boxLabel) ? boxLabel : `Unresolved ${index + 1}`,
      x: Math.max(18, Math.min((box.x + box.width / 2) * 100, 82)),
      y: Math.max(18, Math.min((box.y + box.height / 2) * 100, 82)),
      radius: 12,
      tone: "live" as const,
      depthStratum: "unresolved",
    };
  });
  const atlasNodeData = consumerAtlasNodes.map((node: any, index: number) => ({
    id: String(node.entity_id || node.id || `node-${index}`),
    label: resolveConsumerTrackLabel(
      {
        id: String(node.entity_id || node.id || `node-${index}`),
        session_id: sessionId,
        label: String(node.label || node.entity_id || ""),
        status: String(node.status || "visible"),
        first_seen_at: "",
        last_seen_at: "",
        first_observation_id: "",
        last_observation_id: "",
        observations: [],
        visibility_streak: Number(node.track_length || 1),
        occlusion_count: 0,
        reidentification_count: 0,
        persistence_confidence: Number(node.confidence || 0),
        continuity_score: 0,
        last_similarity: 0,
        status_history: [],
        metadata: { label: node.label, track_length: node.track_length },
      },
      index,
      consumerSceneState,
      node,
    ),
    x: Number.isFinite(Number(node.centroid?.[0]))
      ? Math.max(18, Math.min(Number(node.centroid[0]) * 100, 82))
      : 20 + ((index * 17) % 60),
    y: Number.isFinite(Number(node.centroid?.[1]))
      ? Math.max(18, Math.min(Number(node.centroid[1]) * 100, 82))
      : 25 + ((index * 13) % 50),
    radius: 10 + Math.min(Number(node.track_length || 1), 10),
    tone: String(node.status || "visible") === "occluded" ? "memory" : "live",
    depthStratum: "unresolved",
  }));
  const consumerNodes = mergeConsumerNodes(
    (sceneGraphNodes.length > 0
      ? sceneGraphNodes
      : consumerNodesFromAnchors.length > 0
      ? consumerNodesFromAnchors
      : consumerGroundedNodes.length > 0
      ? consumerGroundedNodes
      : consumerNodesFromTracks.length > 0
      ? consumerNodesFromTracks
      : atlasNodeData.length > 0
        ? atlasNodeData
        : consumerBoxFallback
    ).filter((node: ConsumerGraphNode) => {
        // High-fidelity filter for Consumer View: 
        // 1. Must have a real label (not Unresolved)
        // 2. OR Must have confidence > 0.6 and not be a tiny box
        const isUnresolved = String(node.label || "").startsWith("Unresolved");
        const hasSemanticName = !isUnresolved;
        const confidence = (node as any).confidence ?? 0.7;
        const isHighConfidence = confidence > 0.62;
        
        return hasSemanticName || isHighConfidence;
    })
  );
  const consumerLeadLabel = consumerNodes[0]?.label || sceneLabelCandidates(consumerSceneState)[0] || "it";
  const consumerTrackMessageSource = consumerNodes.length
    ? [
        {
          ...(consumerTrackSource[0] || livingTracks[0] || {
            id: "lead",
            session_id: sessionId,
            label: consumerLeadLabel,
            status: "visible",
            first_seen_at: "",
            last_seen_at: "",
            first_observation_id: "",
            last_observation_id: "",
            observations: [],
            visibility_streak: 1,
            occlusion_count: 0,
            reidentification_count: 0,
            persistence_confidence: 0,
            continuity_score: 0,
            last_similarity: 0,
            status_history: [],
            metadata: {},
          }),
          label: consumerLeadLabel,
        },
        ...consumerTrackSource.slice(1),
      ]
    : consumerTrackSource;
  const consumerText = !runtimeAvailable
    ? runtimeUnavailableLabel
    : currentJepaTick
      ? consumerMessage(currentJepaTick, consumerTrackMessageSource)
      : liveAnalysisPaused
        ? "Auto analyze paused"
        : livingLens.livingLensStatus === "Continuous monitoring is ready"
          ? "Waiting for first JEPA tick"
          : livingLens.livingLensStatus;
  const consumerLinks =
    (world.sceneGraph?.edges || []).length > 0
      ? (world.sceneGraph?.edges || [])
          .filter(
            (edge) =>
              consumerNodes.some((node) => node.id === edge.source) &&
              consumerNodes.some((node) => node.id === edge.target),
          )
          .slice(0, 24)
          .map((edge) => ({
            source: edge.source,
            target: edge.target,
            strength: edge.weight,
          }))
      : consumerNodes.length > 1
      ? consumerNodes.slice(1).map((node: ConsumerGraphNode) => ({
          source: consumerNodes[0]?.id || node.id,
          target: node.id,
          strength: 0.45,
        }))
      : [];
  const livingHistory = runtimeAvailable
    ? [livingLens.livingLensResult?.scene_state, ...worldHistory]
        .filter((state): state is SceneState => Boolean(state))
        .filter((state, index, list) => list.findIndex((candidate) => candidate.id === state.id) === index)
    : [];
  const baselineHistory = livingHistory.slice(0, 24).map((state, index) => ({
    id: state.id,
    label: `Tick ${index + 1}`,
    winner: livingBaseline?.winner || "jepa_hybrid",
    composite: Math.max(0, 1 - state.metrics.surprise_score),
    continuity: state.metrics.temporal_continuity_score,
    persistence: state.metrics.persistence_confidence,
    surpriseSeparation: state.metrics.surprise_score,
    summary: state.observed_state_summary || state.predicted_state_summary,
  }));
  const challengeHistory = world.challengeRun?.world_state_ids?.length
    ? world.challengeRun.world_state_ids
        .map((worldStateId) => livingHistory.find((state) => state.id === worldStateId))
        .filter((state): state is SceneState => Boolean(state))
    : livingHistory.slice(0, 8);
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
  const topHealth = runtimeAvailable ? world.health.slice(0, 4) : [];
  const setEnergyHeatmapEnabled = async (enabled: boolean) => {
    setShowEnergyMap(enabled);
    await world.persistLiveFeatureSetting("energy_heatmap_enabled", enabled);
  };
  const setEntityOverlayEnabled = async (enabled: boolean) => {
    setShowEntities(enabled);
    await world.persistLiveFeatureSetting("entity_overlay_enabled", enabled);
  };
  const currentRollout =
    currentSceneState?.conditioned_rollouts ||
    world.latestRollout ||
    world.worldState?.current?.conditioned_rollouts ||
    null;
  const currentBenchmark = world.latestBenchmark || world.worldState?.benchmarks?.[0] || null;

  return {
    activeTab,
    setActiveTab,
    livingSection,
    setLivingSection,
    prompt,
    setPrompt,
    searchText,
    setSearchText,
    showEnergyMap,
    setShowEnergyMap: setEnergyHeatmapEnabled,
    showEntities,
    setShowEntities: setEntityOverlayEnabled,
    showAllTracks,
    setShowAllTracks,
    challengeGuideActive,
    challengeStepIndex,
    uiMode,
    setUiMode,
    sessionId,
    assetUrl,
    runtimeRequest,
    camera,
    livingLens,
    world,
    captureFrame,
    analyzeFile,
    startGuidedChallenge,
    advanceGuidedChallenge,
    resetGuidedChallenge,
    latestObservation,
    latestObservationSummary,
    readyProviders,
    degradedProviders,
    topFeatureHealth,
    currentSceneState,
    livingTracks,
    livingObservation,
    currentJepaTick,
    livingAnswer,
    livingNearest,
    livingMatches,
    livingBaseline,
    worldHistory,
    continuitySignal,
    persistenceSignal,
    liveBoxes,
    livingBoxes,
    challengeSteps,
    currentChallengeStep,
    worldModelSummary,
    cameraStatusLabel,
    cameraConnectionState,
    ghostBoxes,
    energyAnchors,
    energyHeatmapStatus,
    fe1,
    fe2,
    fe5,
    forecastMonotonic,
    occlusionTracks,
    consumerText,
    consumerNodes,
    consumerLinks,
    livingHistory,
    baselineHistory,
    challengeHistory,
    featuredTracks,
    displayedTracks,
    topHealth,
    currentRollout,
    currentBenchmark,
    runtimeConnectionLabel,
  };
}

type DesktopAppContextValue = ReturnType<typeof useDesktopAppValue>;

const DesktopAppContext = createContext<DesktopAppContextValue | null>(null);

export function DesktopAppProvider({ children }: { children: ReactNode }) {
  const value = useDesktopAppValue();
  return <DesktopAppContext.Provider value={value}>{children}</DesktopAppContext.Provider>;
}

export function useDesktopApp() {
  const context = useContext(DesktopAppContext);
  if (!context) {
    throw new Error("useDesktopApp must be used within DesktopAppProvider");
  }
  return context;
}
