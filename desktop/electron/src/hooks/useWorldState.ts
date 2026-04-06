import { startTransition, useCallback, useEffect, useRef, useState } from "react";
import { BROWSER_RUNTIME_URL, copyTextToClipboard, getDesktopBridge } from "./useRuntimeBridge";
import type {
  AnalyzeResponse,
  ChallengeRun,
  ObservationSharePayload,
  PlanningRolloutResponse,
  ProviderHealth,
  QueryResponse,
  RecoveryBenchmarkRun,
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
  const [savedSettings, setSavedSettings] = useState<Settings | null>(null);
  const [settingsDraft, setSettingsDraft] = useState<Settings | null>(null);
  const [settingsDirty, setSettingsDirty] = useState(false);
  const [health, setHealth] = useState<ProviderHealth[]>([]);
  const [observations, setObservations] = useState<Observation[]>([]);
  const [worldState, setWorldState] = useState<WorldStateResponse | null>(null);
  const [worldModelStatus, setWorldModelStatus] = useState<WorldModelStatus | null>(null);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [status, setStatus] = useState("Connecting to runtime");
  const [eventsState, setEventsState] = useState("disconnected");
  const [savingSettings, setSavingSettings] = useState(false);
  const [challengeRun, setChallengeRun] = useState<ChallengeRun | null>(null);
  const [challengeBusy, setChallengeBusy] = useState(false);
  const [llmLatencyMs, setLlmLatencyMs] = useState<number | null>(null);
  const [exportingProof, setExportingProof] = useState(false);
  const [latestRollout, setLatestRollout] = useState<PlanningRolloutResponse["comparison"] | null>(null);
  const [latestBenchmark, setLatestBenchmark] = useState<RecoveryBenchmarkRun | null>(null);
  const lastRefreshAtRef = useRef(0);
  const refreshTimerRef = useRef<number | null>(null);

  const settings = settingsDraft ?? savedSettings;

  const applySettingsSnapshot = useCallback((next: Settings) => {
    setSavedSettings(next);
    setSettingsDraft((current) => (settingsDirty && current ? current : next));
  }, [settingsDirty]);

  const refreshAll = useCallback(async () => {
    const [
      settingsResponse,
      healthResponse,
      observationsResponse,
      worldStateResponse,
      worldModelStatusResponse,
    ] = await Promise.all([
      runtimeRequest<Settings>("/v1/settings"),
      runtimeRequest<{ providers: ProviderHealth[] }>("/v1/providers/health"),
      runtimeRequest<{ observations: Observation[] }>(
        `/v1/observations?session_id=${encodeURIComponent(sessionId)}&limit=48`,
      ),
      runtimeRequest<WorldStateResponse>(`/v1/world-state?session_id=${encodeURIComponent(sessionId)}`),
      runtimeRequest<WorldModelStatus>("/v1/world-model/status"),
    ]);
    applySettingsSnapshot(settingsResponse);
    setHealth(healthResponse.providers);
    setObservations(observationsResponse.observations);
    setWorldState(worldStateResponse);
    setWorldModelStatus(worldModelStatusResponse);
    setChallengeRun(worldStateResponse.challenges?.[0] || null);
    setLatestRollout(worldStateResponse.current?.conditioned_rollouts || null);
    setLatestBenchmark(worldStateResponse.benchmarks?.[0] || null);
    setStatus("Runtime ready");
  }, [applySettingsSnapshot, runtimeRequest, sessionId]);

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
        refreshAll().catch(() => undefined);
      });
    } catch (error) {
      setStatus((error as Error).message);
    } finally {
      setChallengeBusy(false);
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
            if (narrated?.summary && !narrated.summary.startsWith("Gemma 4 narrator is visually unavailable") && !narrated.summary.startsWith("Rollout plan calculated safely")) {
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
        refreshAll().catch(() => undefined);
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
        refreshAll().catch(() => undefined);
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
  }, [refreshAll]);

  useEffect(() => {
    refreshAll().catch((error) => setStatus((error as Error).message));
  }, [refreshAll]);

  useEffect(() => {
    const socket = new WebSocket("ws://127.0.0.1:7777/v1/events");
    socket.onopen = () => setEventsState("connected");
    socket.onclose = () => setEventsState("disconnected");
    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message?.type === "llm_latency") {
          const ms = Number(message?.payload?.ms);
          if (Number.isFinite(ms)) {
            setLlmLatencyMs(ms);
          }
          return;
        }
      } catch {
        // Ignore parse failures and still refresh.
      }
      scheduleRefresh();
    };
    return () => {
      if (refreshTimerRef.current != null) {
        window.clearTimeout(refreshTimerRef.current);
        refreshTimerRef.current = null;
      }
      if (socket.readyState === WebSocket.CONNECTING) {
        socket.onopen = () => socket.close();
        socket.onerror = () => {};
      } else {
        socket.close();
      }
    };
  }, [scheduleRefresh]);

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
    observations,
    setObservations,
    worldState,
    setWorldState,
    worldModelStatus,
    setWorldModelStatus,
    analysis,
    setAnalysis,
    queryResult,
    setQueryResult,
    status,
    setStatus,
    eventsState,
    savingSettings,
    challengeRun,
    setChallengeRun,
    challengeBusy,
    llmLatencyMs,
    exportingProof,
    latestRollout,
    latestBenchmark,
    refreshAll,
    saveSettings,
    mutateSetting,
    mutateProviderEnabled,
    runChallengeEvaluation,
    exportProofReport,
    copyObservationShare,
    observeToolState,
    runPlanningRollout,
    runRecoveryBenchmark,
  };
}
