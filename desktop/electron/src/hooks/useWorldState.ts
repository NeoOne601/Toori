import { startTransition, useCallback, useEffect, useRef, useState } from "react";
import { BROWSER_RUNTIME_URL, copyTextToClipboard, getDesktopBridge } from "./useRuntimeBridge";
import type {
  AnalyzeResponse,
  ChallengeRun,
  JEPATickPayload,
  ObservationSharePayload,
  PlanningRolloutResponse,
  ProviderHealth,
  ProviderHealthResponse,
  QueryResponse,
  RecoveryBenchmarkRun,
  RuntimeSnapshotResponse,
  RuntimeFeatureStatus,
  SceneGraphPayload,
  Settings,
  ToolStateObserveResponse,
  WorldModelStatus,
  WorldStateResponse,
  Observation,
} from "../types";

type UseWorldStateOptions = {
  sessionId: string;
  searchText: string;
  runtimeRequest: <T>(path: string, method?: string, body?: unknown) => Promise<T>;
};

type LiveFeatureSettingKey = keyof NonNullable<Settings["live_features"]>;

function cloneSettings(settings: Settings | null) {
  return settings ? structuredClone(settings) : null;
}

function settingsEqual(left: Settings | null, right: Settings | null) {
  return JSON.stringify(left || {}) === JSON.stringify(right || {});
}

function resolveThemePreference(preference: string | undefined) {
  if (!preference || preference === "system") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  return preference;
}

function resolveColorScheme(theme: string) {
  if (theme === "light" || theme === "sepia" || theme === "high_contrast_light") {
    return "light";
  }
  return "dark";
}

export function useWorldState({
  sessionId,
  searchText,
  runtimeRequest,
}: UseWorldStateOptions) {
  const runtimeSnapshotFailureCountRef = useRef(0);
  const [savedSettings, setSavedSettings] = useState<Settings | null>(null);
  const [settingsDraft, setSettingsDraft] = useState<Settings | null>(null);
  const [settingsDirty, setSettingsDirty] = useState(false);
  const [health, setHealth] = useState<ProviderHealth[]>([]);
  const [featureHealth, setFeatureHealth] = useState<RuntimeFeatureStatus[]>([]);
  const [observations, setObservations] = useState<Observation[]>([]);
  const [worldState, setWorldState] = useState<WorldStateResponse | null>(null);
  const [worldModelStatus, setWorldModelStatus] = useState<WorldModelStatus | null>(null);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [latestJepaTick, setLatestJepaTick] = useState<JEPATickPayload | null>(null);
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [status, setStatus] = useState("Connecting to runtime");
  const [eventsState, setEventsState] = useState("connecting");
  const [savingSettings, setSavingSettings] = useState(false);
  const [challengeRun, setChallengeRun] = useState<ChallengeRun | null>(null);
  const [sceneGraph, setSceneGraph] = useState<SceneGraphPayload>({ nodes: [], edges: [] });
  const [challengeBusy, setChallengeBusy] = useState(false);
  const [llmLatencyMs, setLlmLatencyMs] = useState<number | null>(null);
  const [exportingProof, setExportingProof] = useState(false);
  const [latestRollout, setLatestRollout] = useState<PlanningRolloutResponse["comparison"] | null>(null);
  const [latestBenchmark, setLatestBenchmark] = useState<RecoveryBenchmarkRun | null>(null);
  const [runtimeAvailable, setRuntimeAvailable] = useState(false);
  const lastRefreshAtRef = useRef(0);
  const refreshTimerRef = useRef<number | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectUiTimerRef = useRef<number | null>(null);
  const reconnectAttemptRef = useRef(0);
  const runtimeGenerationRef = useRef(0);

  const settings = settingsDraft ?? savedSettings;

  function clearRuntimeDerivedState() {
    setHealth([]);
    setFeatureHealth([]);
    setObservations([]);
    setWorldState(null);
    setWorldModelStatus(null);
    setAnalysis(null);
    setLatestJepaTick(null);
    setQueryResult(null);
    setChallengeRun(null);
    setSceneGraph({ nodes: [], edges: [] });
    setLatestRollout(null);
    setLatestBenchmark(null);
    setLlmLatencyMs(null);
  }

  function clearReconnectUiTimer() {
    if (reconnectUiTimerRef.current == null) {
      return;
    }
    window.clearTimeout(reconnectUiTimerRef.current);
    reconnectUiTimerRef.current = null;
  }

  function recordRuntimeSnapshotSuccess() {
    runtimeSnapshotFailureCountRef.current = 0;
    clearReconnectUiTimer();
    setRuntimeAvailable(true);
    setStatus("Runtime ready");
  }

  function scheduleReconnectUi() {
    if (reconnectUiTimerRef.current != null) {
      return;
    }
    reconnectUiTimerRef.current = window.setTimeout(() => {
      reconnectUiTimerRef.current = null;
      setEventsState("reconnecting");
      setStatus("Runtime stream reconnecting");
    }, 240);
  }

  function isTransportFailure(error: unknown) {
    const message = (error as Error)?.message || "";
    return /runtime unreachable|failed to fetch|fetch failed|networkerror/i.test(message);
  }

  const applySettingsSnapshot = useCallback((next: Settings) => {
    setSavedSettings(next);
    setSettingsDraft((current) => (settingsDirty && current ? current : next));
  }, [settingsDirty]);

  const applySnapshot = useCallback((snapshot: RuntimeSnapshotResponse) => {
    setObservations(snapshot.observations || []);
    setWorldModelStatus(snapshot.world_model_status);
    setLatestJepaTick(snapshot.latest_jepa_tick || null);
    setSceneGraph(snapshot.scene_graph || { nodes: [], edges: [] });
    setWorldState((current) => {
      const nextCurrent = snapshot.current ?? null;
      const priorHistory = current?.history || [];
      const history = nextCurrent
        ? [nextCurrent, ...priorHistory.filter((state) => state.id !== nextCurrent.id)]
        : priorHistory;
      return {
        session_id: snapshot.session_id,
        current: nextCurrent,
        history,
        entity_tracks: snapshot.entity_tracks || [],
        challenges: current?.challenges || [],
        benchmarks: current?.benchmarks || [],
        atlas: current?.atlas || null,
      };
    });
    setLatestRollout(snapshot.current?.conditioned_rollouts || null);
    recordRuntimeSnapshotSuccess();
  }, []);

  const refreshHot = useCallback(async () => {
    const generation = runtimeGenerationRef.current;
    try {
      const snapshot = await runtimeRequest<RuntimeSnapshotResponse>(
        `/v1/runtime/snapshot?session_id=${encodeURIComponent(sessionId)}&observation_limit=24`,
      );
      if (generation !== runtimeGenerationRef.current) {
        return;
      }
      applySnapshot(snapshot);
    } catch (error) {
      if (!isTransportFailure(error)) {
        throw error;
      }
      if (generation !== runtimeGenerationRef.current) {
        return;
      }
      runtimeGenerationRef.current += 1;
      runtimeSnapshotFailureCountRef.current += 1;
      if (runtimeSnapshotFailureCountRef.current >= 3) {
        clearReconnectUiTimer();
        setRuntimeAvailable(false);
        setStatus("Runtime unavailable. Reconnecting...");
      }
    }
  }, [applySnapshot, runtimeRequest, sessionId]);

  const refreshCold = useCallback(async () => {
    const generation = runtimeGenerationRef.current;
    try {
      const [
        settingsResponse,
        healthResponse,
        observationsResponse,
        worldStateResponse,
      ] = await Promise.all([
        runtimeRequest<Settings>("/v1/settings"),
        runtimeRequest<ProviderHealthResponse>("/v1/providers/health"),
        runtimeRequest<{ observations: Observation[] }>(
          `/v1/observations?session_id=${encodeURIComponent(sessionId)}&limit=48&summary_only=true`,
        ),
        runtimeRequest<WorldStateResponse>(`/v1/world-state?session_id=${encodeURIComponent(sessionId)}`),
      ]);
      if (generation !== runtimeGenerationRef.current) {
        return;
      }
      applySettingsSnapshot(settingsResponse);
      setHealth(healthResponse.providers);
      setFeatureHealth(healthResponse.features || []);
      setObservations(observationsResponse.observations);
      setWorldState(worldStateResponse);
      setChallengeRun(worldStateResponse.challenges?.[0] || null);
      setLatestRollout(worldStateResponse.current?.conditioned_rollouts || null);
      setLatestBenchmark(worldStateResponse.benchmarks?.[0] || null);
      recordRuntimeSnapshotSuccess();
    } catch (error) {
      if (!isTransportFailure(error)) {
        throw error;
      }
      if (generation !== runtimeGenerationRef.current) {
        return;
      }
      runtimeGenerationRef.current += 1;
      runtimeSnapshotFailureCountRef.current += 1;
      if (runtimeSnapshotFailureCountRef.current >= 3) {
        clearReconnectUiTimer();
        setRuntimeAvailable(false);
        setStatus("Runtime unavailable. Reconnecting...");
      }
    }
  }, [applySettingsSnapshot, runtimeRequest, sessionId]);

  const refreshAll = useCallback(async () => {
    await refreshHot();
    void refreshCold();
  }, [refreshCold, refreshHot]);

  const setSettings = useCallback((next: Settings) => {
    setSavedSettings(next);
    setSettingsDraft(next);
    setSettingsDirty(false);
  }, []);

  async function saveSettings() {
    if (!settingsDraft) {
      return;
    }
    setSavingSettings(true);
    try {
      const updated = await runtimeRequest<Settings>("/v1/settings", "PUT", settingsDraft);
      setSavedSettings(updated);
      setSettingsDraft(updated);
      setSettingsDirty(false);
      setStatus("Settings saved");
      await refreshCold();
    } catch (error) {
      setStatus((error as Error).message);
    } finally {
      setSavingSettings(false);
    }
  }

  async function setThemePreference(nextTheme: string) {
    const persistedBase = cloneSettings(savedSettings) ?? cloneSettings(settingsDraft);
    const nextDraft = cloneSettings(settingsDraft) ?? cloneSettings(savedSettings);
    if (!persistedBase || !nextDraft) {
      return;
    }
    persistedBase.theme_preference = nextTheme;
    nextDraft.theme_preference = nextTheme;
    setSettingsDraft(nextDraft);
    setSavedSettings((current) => (current ? { ...current, theme_preference: nextTheme } : current));
    setSavingSettings(true);
    try {
      const updated = await runtimeRequest<Settings>("/v1/settings", "PUT", persistedBase);
      setSavedSettings(updated);
      setSettingsDraft((current) => {
        if (!current) {
          return updated;
        }
        const merged = structuredClone(current);
        merged.theme_preference = updated.theme_preference;
        return merged;
      });
      setSettingsDirty((current) => {
        const draftWithTheme = cloneSettings(nextDraft);
        if (!draftWithTheme) {
          return current;
        }
        return !settingsEqual(
          { ...draftWithTheme, theme_preference: updated.theme_preference },
          updated,
        );
      });
      setStatus(`Theme set to ${nextTheme.replace(/_/g, " ")}`);
      await refreshCold();
    } catch (error) {
      setStatus((error as Error).message);
      setSavedSettings(savedSettings);
      setSettingsDraft(settingsDraft);
    } finally {
      setSavingSettings(false);
    }
  }

  function mutateSetting(path: string[], value: string | number | boolean) {
    setSettingsDraft((current) => {
      const base = structuredClone(current ?? savedSettings);
      if (!base) {
        return base;
      }
      let pointer = base;
      for (const key of path.slice(0, -1)) {
        pointer = pointer[key];
      }
      pointer[path[path.length - 1]] = value;
      return base;
    });
    setSettingsDirty(true);
  }

  function mutateProviderEnabled(name: string, enabled: boolean) {
    mutateSetting(["providers", name, "enabled"], enabled);
  }

  async function persistLiveFeatureSetting(key: LiveFeatureSettingKey, value: boolean) {
    const base = cloneSettings(savedSettings) ?? cloneSettings(settingsDraft);
    if (!base) {
      return;
    }
    const next = structuredClone(base);
    next.live_features = {
      ...(next.live_features || {}),
      [key]: value,
    };
    try {
      const updated = await runtimeRequest<Settings>("/v1/settings", "PUT", next);
      setSavedSettings(updated);
      setSettingsDraft((current) => {
        if (!current) {
          return updated;
        }
        const merged = structuredClone(current);
        merged.live_features = {
          ...(merged.live_features || {}),
          [key]: updated.live_features?.[key],
        };
        return merged;
      });
      setStatus("Live feature settings saved");
      await refreshCold();
      await refreshHot();
    } catch (error) {
      setStatus((error as Error).message);
    }
  }

  async function runChallengeEvaluation(challengeSet: "live" | "curated" | "both" = "both") {
    setChallengeBusy(true);
    setStatus("Evaluating JEPA challenge sequence");
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
        refreshCold().catch(() => undefined);
        refreshHot().catch(() => undefined);
      });
    } catch (error) {
      setStatus((error as Error).message);
    } finally {
      setChallengeBusy(false);
    }
  }

  async function retryNativeJEPA() {
    setStatus("Retrying native JEPA");
    try {
      const nextStatus = await runtimeRequest<WorldModelStatus>("/v1/world-model/retry-native", "POST", {
        session_id: sessionId,
      });
      setWorldModelStatus(nextStatus);
      setStatus("Native JEPA retry requested");
      await refreshHot();
      void refreshCold();
      return nextStatus;
    } catch (error) {
      setStatus((error as Error).message);
      throw error;
    }
  }

  async function exportProofReport() {
    setExportingProof(true);
    setStatus("Generating proof report");
    try {
      const result = await runtimeRequest<{ path?: string }>("/v1/proof-report/generate", "POST", {
        session_id: sessionId,
      });
      const desktopBridge = getDesktopBridge();
      if (desktopBridge.openPath && result.path) {
        const openError = await desktopBridge.openPath(result.path);
        if (openError) {
          const response = await fetch(`${BROWSER_RUNTIME_URL}/v1/proof-report/latest`);
          if (!response.ok) {
            throw new Error(openError);
          }
          const blob = await response.blob();
          const objectUrl = URL.createObjectURL(blob);
          window.open(objectUrl, "_blank", "noopener,noreferrer");
          window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);
        }
      } else {
        const response = await fetch(`${BROWSER_RUNTIME_URL}/v1/proof-report/latest`);
        if (!response.ok) {
          throw new Error(`Proof report unavailable (${response.status})`);
        }
        const blob = await response.blob();
        const objectUrl = URL.createObjectURL(blob);
        window.open(objectUrl, "_blank", "noopener,noreferrer");
        window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);
      }
      setStatus("Proof report generated");
    } catch (error) {
      setStatus((error as Error).message);
    } finally {
      setExportingProof(false);
    }
  }

  async function copyObservationShare(observation: Observation) {
    setStatus("Preparing share text");
    try {
      const payload = await runtimeRequest<ObservationSharePayload>("/v1/share/observation", "POST", {
        session_id: observation.session_id,
        observation_id: observation.id,
      });
      await copyTextToClipboard(payload.share_text);
      await runtimeRequest<{ recorded: boolean }>("/v1/share/observation/event", "POST", {
        session_id: observation.session_id,
        observation_id: observation.id,
        event_type: "share_copied",
      });
      const message =
        payload.tracked_entities > 0
          ? `Copied share text for ${payload.tracked_entities} tracked ${payload.tracked_entities === 1 ? "entity" : "entities"}.`
          : "Copied share text.";
      setStatus(message);
      return message;
    } catch (error) {
      const message = (error as Error).message;
      setStatus(message);
      throw error;
    }
  }

  async function runPlanningRollout(body?: Record<string, unknown>) {
    setStatus("Planning rollout");
    try {
      const result = await runtimeRequest<PlanningRolloutResponse>("/v1/planning/rollout", "POST", {
        session_id: sessionId,
        horizon: 2,
        ...body,
      });
      setLatestRollout(result.comparison);
      setWorldState((current) =>
        current
          ? {
              ...current,
              current: result.scene_state,
              history: [
                result.scene_state,
                ...current.history.filter((state) => state.id !== result.scene_state.id),
              ],
            }
          : current,
      );
      
      if (result.comparison) {
        runtimeRequest<{ summary: string }>("/v1/planning/rollout/narrate", "POST", result)
          .then((narrated) => {
            if (narrated?.summary?.trim()) {
              setLatestRollout((current) => 
                current ? { ...current, summary: narrated.summary } : current
              );
              setWorldState((current) => {
                if (!current?.current?.conditioned_rollouts) return current;
                return {
                  ...current,
                  current: {
                    ...current.current,
                    conditioned_rollouts: {
                      ...current.current.conditioned_rollouts,
                      summary: narrated.summary
                    }
                  }
                };
              });
            }
          })
          .catch((e) => console.warn("Rollout narration skipped:", e));
      }

      setStatus(result.comparison.summary || "Rollout updated");
      return result;
    } catch (error) {
      setStatus((error as Error).message);
      throw error;
    }
  }

  async function runRecoveryBenchmark(body?: Record<string, unknown>) {
    setStatus("Running recovery benchmark");
    try {
      const result = await runtimeRequest<RecoveryBenchmarkRun>("/v1/benchmarks/recovery/run", "POST", {
        session_id: sessionId,
        ...body,
      });
      setLatestBenchmark(result);
      setWorldState((current) =>
        current
          ? {
              ...current,
              benchmarks: [
                result,
                ...current.benchmarks.filter((benchmark) => benchmark.id !== result.id),
              ],
            }
          : current,
      );
      setStatus(result.summary);
      return result;
    } catch (error) {
      setStatus((error as Error).message);
      throw error;
    }
  }

  async function observeToolState(body: Record<string, unknown>) {
    setStatus("Observing tool state");
    try {
      const result = await runtimeRequest<ToolStateObserveResponse>("/v1/tool-state/observe", "POST", {
        session_id: sessionId,
        ...body,
      });
      setAnalysis({
        observation: result.observation,
        hits: result.hits,
        answer: result.answer,
        provider_health: result.provider_health,
        reasoning_trace: result.reasoning_trace,
      });
      startTransition(() => {
        refreshHot().catch(() => undefined);
      });
      setStatus("Tool state observed");
      return result;
    } catch (error) {
      setStatus((error as Error).message);
      throw error;
    }
  }

  const scheduleRefresh = useCallback(() => {
    const now = Date.now();
    const elapsed = now - lastRefreshAtRef.current;
    const run = () => {
      lastRefreshAtRef.current = Date.now();
      startTransition(() => {
        refreshHot().catch(() => undefined);
      });
    };
    if (elapsed >= 750) {
      if (refreshTimerRef.current != null) {
        window.clearTimeout(refreshTimerRef.current);
        refreshTimerRef.current = null;
      }
      run();
      return;
    }
    if (refreshTimerRef.current != null) {
      return;
    }
    refreshTimerRef.current = window.setTimeout(() => {
      refreshTimerRef.current = null;
      run();
    }, 750 - elapsed);
  }, [refreshHot]);

  useEffect(() => {
    refreshHot().catch((error) => setStatus((error as Error).message));
    refreshCold().catch((error) => setStatus((error as Error).message));
  }, [refreshCold, refreshHot]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      refreshCold().catch(() => undefined);
    }, 30_000);
    return () => window.clearInterval(interval);
  }, [refreshCold]);

  useEffect(() => {
    let cancelled = false;

    const clearReconnectTimer = () => {
      if (reconnectTimerRef.current == null) {
        return;
      }
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    };

    const connect = () => {
      if (cancelled) {
        return;
      }
      const existing = socketRef.current;
      if (
        existing &&
        (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)
      ) {
        return;
      }
      clearReconnectTimer();
      setEventsState((current) => (current === "connected" ? "reconnecting" : "connecting"));
      const socket = new WebSocket("ws://127.0.0.1:7777/v1/events");
      socketRef.current = socket;
      socket.onopen = () => {
        if (cancelled) {
          return;
        }
        reconnectAttemptRef.current = 0;
        clearReconnectTimer();
        clearReconnectUiTimer();
        runtimeSnapshotFailureCountRef.current = 0;
        setEventsState("connected");
        void refreshHot();
        void refreshCold();
      };
      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          const type = String(message?.type || "");
          if (type === "llm_latency") {
            const ms = Number(message?.payload?.ms);
            if (Number.isFinite(ms)) {
              setLlmLatencyMs(ms);
            }
            return;
          }
          if (type === "jepa_tick" || type === "jepa_tick.enriched") {
            const eventSessionId = String(message?.payload?.session_id || "");
            const tickPayload = message?.payload?.payload;
            if (
              eventSessionId === sessionId &&
              tickPayload &&
              Array.isArray(tickPayload.energy_map)
            ) {
              setLatestJepaTick(tickPayload as JEPATickPayload);
              setWorldModelStatus((current) =>
                current
                  ? {
                      ...current,
                      last_tick_encoder_type: String(tickPayload.last_tick_encoder_type || current.last_tick_encoder_type),
                      active_backend: String(tickPayload.last_tick_encoder_type || current.active_backend),
                      degraded: Boolean(tickPayload.degraded ?? current.degraded),
                      degrade_reason:
                        tickPayload.degrade_reason == null ? current.degrade_reason : String(tickPayload.degrade_reason),
                      degrade_stage:
                        tickPayload.degrade_stage == null ? current.degrade_stage : String(tickPayload.degrade_stage),
                    }
                  : current,
              );
            }
            return;
          }
          if (
            type === "world_state.updated" ||
            type === "observation.created" ||
            type === "tool_state.observed" ||
            type === "planning.rollout" ||
            type === "recovery.benchmark" ||
            type === "world_model.degraded"
          ) {
            scheduleRefresh();
            return;
          }
        } catch {
          scheduleRefresh();
          return;
        }
      };
      socket.onerror = () => {
        if (cancelled) {
          return;
        }
        runtimeGenerationRef.current += 1;
        scheduleReconnectUi();
        if (reconnectTimerRef.current == null) {
          const delayMs = Math.min(8000, 1000 * (2 ** reconnectAttemptRef.current || 1));
          reconnectAttemptRef.current += 1;
          reconnectTimerRef.current = window.setTimeout(() => {
            reconnectTimerRef.current = null;
            connect();
          }, delayMs);
        }
      };
      socket.onclose = () => {
        if (socketRef.current === socket) {
          socketRef.current = null;
        }
        if (cancelled) {
          return;
        }
        runtimeGenerationRef.current += 1;
        scheduleReconnectUi();
        if (reconnectTimerRef.current == null) {
          const delayMs = Math.min(8000, 1000 * (2 ** reconnectAttemptRef.current || 1));
          reconnectAttemptRef.current += 1;
          reconnectTimerRef.current = window.setTimeout(() => {
            reconnectTimerRef.current = null;
            connect();
          }, delayMs);
        }
      };
    };

    connect();
    return () => {
      cancelled = true;
      if (refreshTimerRef.current != null) {
        window.clearTimeout(refreshTimerRef.current);
        refreshTimerRef.current = null;
      }
      clearReconnectTimer();
      clearReconnectUiTimer();
      const socket = socketRef.current;
      socketRef.current = null;
      if (socket) {
        socket.onopen = null;
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        socket.close();
      }
    };
  }, [refreshCold, refreshHot, scheduleRefresh, sessionId]);

  useEffect(() => {
    const root = document.documentElement;
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const applyTheme = () => {
      const resolved = resolveThemePreference(settings?.theme_preference);
      root.dataset.theme = resolved;
      root.style.colorScheme = resolveColorScheme(resolved);
    };
    applyTheme();
    if ((settings?.theme_preference || "system") !== "system") {
      return;
    }
    const handleChange = () => applyTheme();
    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, [settings?.theme_preference]);

  useEffect(() => {
    if (!searchText.trim()) {
      setQueryResult(null);
      return;
    }
    const timeout = window.setTimeout(() => {
      runtimeRequest<QueryResponse>("/v1/query", "POST", {
        query: searchText.trim(),
        session_id: sessionId,
        top_k: settings?.top_k || 6,
      })
        .then(setQueryResult)
        .catch((error) => setStatus((error as Error).message));
    }, 250);
    return () => window.clearTimeout(timeout);
  }, [runtimeRequest, searchText, sessionId, settings?.top_k]);

  return {
    settings,
    savedSettings,
    settingsDirty,
    setSettings,
    setSettingsDraft,
    setThemePreference,
    health,
    setHealth,
    featureHealth,
    setFeatureHealth,
    observations,
    setObservations,
    worldState,
    setWorldState,
    worldModelStatus,
    setWorldModelStatus,
    analysis,
    setAnalysis,
    latestJepaTick,
    setLatestJepaTick,
    queryResult,
    setQueryResult,
    status,
    setStatus,
    eventsState,
    savingSettings,
    challengeRun,
    setChallengeRun,
    sceneGraph,
    challengeBusy,
    llmLatencyMs,
    exportingProof,
    latestRollout,
    latestBenchmark,
    runtimeAvailable,
    refreshHot,
    refreshCold,
    refreshAll,
    saveSettings,
    mutateSetting,
    mutateProviderEnabled,
    persistLiveFeatureSetting,
    retryNativeJEPA,
    runChallengeEvaluation,
    exportProofReport,
    copyObservationShare,
    observeToolState,
    runPlanningRollout,
    runRecoveryBenchmark,
  };
}
