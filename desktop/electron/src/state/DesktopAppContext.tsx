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
  normalizeBoxes,
  toEnergyAnchors,
  toForecastValue,
  toGhostBoxes,
} from "../lib/formatting";
import type {
  AnalyzeResponse,
  ChallengeRun,
  ConsumerGraphNode,
  Observation,
  SceneState,
} from "../types";

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
  const [sessionId] = useState("desktop-live");
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
    setLivingSection("challenge");
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
  const currentSceneState = livingLens.livingLensResult?.scene_state || world.worldState?.current || null;
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
  const consumerText = consumerMessage(currentJepaTick, livingTracks);
  const atlasNodes = Array.isArray((world.worldState?.atlas as any)?.nodes)
    ? (world.worldState?.atlas as any).nodes
    : [];
  const atlasEdges = Array.isArray((world.worldState?.atlas as any)?.edges)
    ? (world.worldState?.atlas as any).edges
    : [];
  const trackNodeFallback = livingTracks.slice(0, 6).map((track, index) => {
    const metadata = (track.metadata || {}) as Record<string, any>;
    const bbox = metadata.bbox_pixels || metadata.ghost_bbox_pixels || metadata.bbox;
    const width = Math.max(livingObservation?.width || 1, 1);
    const height = Math.max(livingObservation?.height || 1, 1);
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
    return {
      id: String(track.id),
      label: formatEntityBaseLabel(track),
      x: Math.max(10, Math.min(center.x, 90)),
      y: Math.max(10, Math.min(center.y, 90)),
      radius: 10 + Math.min(track.visibility_streak || 1, 10),
      tone: track.status === "occluded" ? "memory" : track.status === "visible" ? "live" : "stable",
    };
  });
  const boxNodeFallback = livingBoxes.slice(0, 6).map((box, index) => ({
    id: `box-${index}`,
    label: formatEntityBaseLabel({ label: box.label, metadata: box.metadata }),
    x: Math.max(10, Math.min((box.x + box.width / 2) * 100, 90)),
    y: Math.max(10, Math.min((box.y + box.height / 2) * 100, 90)),
    radius: 12,
    tone: "live" as const,
  }));
  const atlasNodeData = atlasNodes.map((node: any, index: number) => ({
    id: String(node.entity_id || node.id || `node-${index}`),
    label: String(node.label || node.entity_id || `Entity ${index + 1}`),
    x: Number.isFinite(Number(node.centroid?.[0]))
      ? Number(node.centroid[0]) * 100
      : 20 + ((index * 17) % 60),
    y: Number.isFinite(Number(node.centroid?.[1]))
      ? Number(node.centroid[1]) * 100
      : 25 + ((index * 13) % 50),
    radius: 10 + Math.min(Number(node.track_length || 1), 10),
    tone: String(node.status || "visible") === "occluded" ? "memory" : "live",
  }));
  const consumerNodes =
    trackNodeFallback.length > 0
      ? trackNodeFallback
      : atlasNodeData.length > 0
        ? atlasNodeData
        : boxNodeFallback;
  const consumerLinks =
    trackNodeFallback.length > 0 || boxNodeFallback.length > 0
      ? consumerNodes.slice(1).map((node: ConsumerGraphNode) => ({
          source: consumerNodes[0]?.id || node.id,
          target: node.id,
          strength: 0.45,
        }))
      : atlasEdges.map((edge: any) => ({
          source: String(edge.source_id),
          target: String(edge.target_id),
          strength: Number(edge.spatial_proximity || 0.5),
        }));
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
  const topHealth = world.health.slice(0, 4);

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
