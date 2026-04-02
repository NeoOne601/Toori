import {
  createContext,
  startTransition,
  useContext,
  useEffect,
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
  formatEntityBaseLabel,
  humanizeLabel,
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
  Observation,
  SceneState,
} from "../types";

function isGenericEntityLabel(label: string | null | undefined) {
  const normalized = humanizeLabel(String(label || ""));
  return !normalized || normalized === "tracked region" || /^entity[-\s]?\d+$/i.test(normalized);
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

function resolveConsumerTrackLabel(
  track: EntityTrack,
  index: number,
  sceneState?: SceneState | null,
  atlasNode?: Record<string, unknown> | null,
  proposalLabel?: string,
) {
  const metadata = (track.metadata || {}) as Record<string, unknown>;
  const sceneCandidates = sceneLabelCandidates(sceneState);
  const candidates = [
    humanizeLabel(proposalLabel || ""),
    humanizeLabel(String(metadata.primary_object_label || "")),
    humanizeLabel(String(metadata.caption || "")),
    humanizeLabel(String(metadata.top_label || "")),
    humanizeLabel(String(atlasNode?.label || "")),
    sceneCandidates[index] || "",
    humanizeLabel(String((sceneState?.metadata as Record<string, unknown> | undefined)?.primary_object_label || "")),
    formatEntityBaseLabel(track),
    `Entity ${index + 1}`,
  ].filter((item) => item && !isGenericEntityLabel(item));
  return candidates[0] || formatEntityBaseLabel(track) || `Entity ${index + 1}`;
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
    sessionId,
    topK: world.settings?.top_k || 6,
    currentFrameBase64: camera.currentFrameBase64,
    runtimeRequest,
    onHealthChange: world.setHealth,
    onRefresh: world.refreshAll,
  });

  useEffect(() => {
    window.localStorage.setItem("toori_mode", uiMode);
  }, [uiMode]);

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
      world.refreshAll().catch(() => undefined);
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
      const result = await analyzeCurrentFrame({
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
      const result = await runtimeRequest<AnalyzeResponse>("/v1/analyze", "POST", {
        file_path: payload.filePath,
        image_base64: payload.imageBase64,
        session_id: sessionId,
        query: prompt || undefined,
        decode_mode: "force",
      });
      world.setAnalysis(result);
      world.setHealth(result.provider_health);
      startTransition(() => {
        world.refreshAll().catch(() => undefined);
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

  const latestObservation = world.analysis?.observation || world.observations[0];
  const latestObservationSummary = latestObservation
    ? explainSummary(
        latestObservation,
        world.analysis?.observation?.id === latestObservation.id
          ? world.analysis?.answer?.text
          : undefined,
      )
    : null;
  const readyProviders = world.health.filter((item) => item.healthy).length;
  const degradedProviders = Math.max(world.health.length - readyProviders, 0);
  const currentSceneState = world.worldState?.current || livingLens.livingLensResult?.scene_state || null;
  const livingTracks = livingLens.livingLensResult?.entity_tracks || world.worldState?.entity_tracks || [];
  const livingObservation = livingLens.livingLensResult?.observation || latestObservation;
  const currentJepaTick = livingLens.livingLensResult?.jepa_tick || null;
  const livingAnswer = explainSummary(
    livingObservation,
    livingLens.livingLensResult?.answer?.text || currentSceneState?.observed_state_summary,
  );
  const livingNearest = livingLens.livingLensResult?.hits?.[0] || null;
  const livingMatches = livingLens.livingLensResult?.hits || [];
  const livingBaseline =
    livingLens.livingLensResult?.baseline_comparison ||
    world.challengeRun?.baseline_comparison ||
    world.worldState?.challenges?.[0]?.baseline_comparison ||
    null;
  const worldHistory = world.worldState?.history || [];
  const continuitySignal = currentSceneState?.metrics.continuity_signal;
  const persistenceSignal = currentSceneState?.metrics.persistence_signal;
  const liveObservationMetadata =
    (world.analysis?.observation?.metadata as Record<string, unknown> | undefined) ||
    (latestObservation?.metadata as Record<string, unknown> | undefined);
  const liveBoxes = normalizeBoxes(
    liveObservationMetadata?.object_proposals ||
      liveObservationMetadata?.proposal_boxes ||
      liveObservationMetadata?.bounding_boxes,
    world.analysis?.observation || latestObservation,
  );
  const livingBoxes = normalizeBoxes(
    (livingObservation?.metadata as Record<string, unknown> | undefined)?.bounding_boxes ||
      (livingObservation?.metadata as Record<string, unknown> | undefined)?.object_proposals ||
      (livingObservation?.metadata as Record<string, unknown> | undefined)?.proposal_boxes ||
      (currentSceneState?.metadata as Record<string, unknown> | undefined)?.bounding_boxes ||
      (currentSceneState?.metadata as Record<string, unknown> | undefined)?.object_proposals ||
      (currentSceneState?.metadata as Record<string, unknown> | undefined)?.proposal_boxes,
    livingObservation,
  );
  const challengeSteps = world.challengeRun?.guide_steps?.length
    ? world.challengeRun.guide_steps
    : Array.from(DEFAULT_CHALLENGE_STEPS);
  const currentChallengeStep = challengeGuideActive ? challengeSteps[challengeStepIndex] : null;
  const worldModelSummary = explainWorldModel(currentSceneState);
  const cameraStatusLabel = camera.cameraReady
    ? "camera ready"
    : camera.cameraStreamLive
      ? "camera connected"
      : "camera idle";
  const cameraConnectionState =
    !camera.cameraAccess.granted && camera.cameraAccess.status !== "unknown"
      ? "blocked"
      : camera.cameraBusy
        ? "reconnecting"
        : camera.cameraReady
          ? "live"
          : camera.cameraStreamLive
            ? "degraded"
            : "offline";
  const ghostBoxes = toGhostBoxes(livingTracks, livingObservation);
  const energyAnchors = toEnergyAnchors(currentJepaTick?.energy_map || null);
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
  const consumerSceneState = world.worldState?.current || currentSceneState;
  const consumerTrackSourceBase = world.worldState?.entity_tracks?.length ? world.worldState.entity_tracks : livingTracks;
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
    return {
      id: entity.id,
      label: humanizeLabel(entity.label || `${entity.kind} ${index + 1}`),
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
    };
  });
  const consumerAtlasNodes = Array.isArray((world.worldState?.atlas as any)?.nodes)
    ? (world.worldState?.atlas as any).nodes
    : [];
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
    };
  });
  const consumerBoxFallback = livingBoxes.slice(0, 6).map((box, index) => ({
    id: `box-${index}`,
    label: formatEntityBaseLabel({ label: box.label, metadata: box.metadata }),
    x: Math.max(18, Math.min((box.x + box.width / 2) * 100, 82)),
    y: Math.max(18, Math.min((box.y + box.height / 2) * 100, 82)),
    radius: 12,
    tone: "live" as const,
  }));
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
  }));
  const consumerNodes = mergeConsumerNodes(
    consumerGroundedNodes.length > 0
      ? consumerGroundedNodes
      : consumerNodesFromTracks.length > 0
      ? consumerNodesFromTracks
      : atlasNodeData.length > 0
        ? atlasNodeData
        : consumerBoxFallback,
  );
  const consumerLeadLabel = consumerNodes[0]?.label || sceneLabelCandidates(consumerSceneState)[0] || "it";
  const consumerText = consumerMessage(
    currentJepaTick,
    consumerNodes.length
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
      : consumerTrackSource,
  );
  const consumerLinks =
    consumerNodes.length > 1
      ? consumerNodes.slice(1).map((node: ConsumerGraphNode) => ({
          source: consumerNodes[0]?.id || node.id,
          target: node.id,
          strength: 0.45,
        }))
      : [];
  const livingHistory = [livingLens.livingLensResult?.scene_state, ...worldHistory]
    .filter((state): state is SceneState => Boolean(state))
    .filter((state, index, list) => list.findIndex((candidate) => candidate.id === state.id) === index);
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
  const topHealth = world.health.slice(0, 4);
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
    setShowEnergyMap,
    showEntities,
    setShowEntities,
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
