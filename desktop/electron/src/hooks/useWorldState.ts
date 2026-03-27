import { startTransition, useCallback, useEffect, useState } from "react";
import { BROWSER_RUNTIME_URL } from "./useRuntimeBridge";
import type {
  AnalyzeResponse,
  ChallengeRun,
  ProviderHealth,
  QueryResponse,
  Settings,
  WorldStateResponse,
  Observation,
} from "../types";

type UseWorldStateOptions = {
  sessionId: string;
  searchText: string;
  runtimeRequest: <T>(path: string, method?: string, body?: unknown) => Promise<T>;
};

export function useWorldState({
  sessionId,
  searchText,
  runtimeRequest,
}: UseWorldStateOptions) {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [health, setHealth] = useState<ProviderHealth[]>([]);
  const [observations, setObservations] = useState<Observation[]>([]);
  const [worldState, setWorldState] = useState<WorldStateResponse | null>(null);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [status, setStatus] = useState("Connecting to runtime");
  const [eventsState, setEventsState] = useState("disconnected");
  const [savingSettings, setSavingSettings] = useState(false);
  const [challengeRun, setChallengeRun] = useState<ChallengeRun | null>(null);
  const [challengeBusy, setChallengeBusy] = useState(false);
  const [llmLatencyMs, setLlmLatencyMs] = useState<number | null>(null);
  const [exportingProof, setExportingProof] = useState(false);

  const refreshAll = useCallback(async () => {
    const [settingsResponse, healthResponse, observationsResponse, worldStateResponse] =
      await Promise.all([
        runtimeRequest<Settings>("/v1/settings"),
        runtimeRequest<{ providers: ProviderHealth[] }>("/v1/providers/health"),
        runtimeRequest<{ observations: Observation[] }>(
          `/v1/observations?session_id=${encodeURIComponent(sessionId)}&limit=48`,
        ),
        runtimeRequest<WorldStateResponse>(`/v1/world-state?session_id=${encodeURIComponent(sessionId)}`),
      ]);
    setSettings(settingsResponse);
    setHealth(healthResponse.providers);
    setObservations(observationsResponse.observations);
    setWorldState(worldStateResponse);
    setChallengeRun(worldStateResponse.challenges?.[0] || null);
    setStatus("Runtime ready");
  }, [runtimeRequest, sessionId]);

  async function saveSettings() {
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
      await runtimeRequest<{ path: string }>("/v1/proof-report/generate", "POST", {
        session_id: sessionId,
      });
      window.open(`${BROWSER_RUNTIME_URL}/v1/proof-report/latest`, "_blank", "noopener,noreferrer");
      setStatus("Proof report generated");
    } catch (error) {
      setStatus((error as Error).message);
    } finally {
      setExportingProof(false);
    }
  }

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
        }
      } catch {
        // Ignore parse failures and still refresh.
      }
      startTransition(() => {
        refreshAll().catch(() => undefined);
      });
    };
    return () => {
      if (socket.readyState === WebSocket.CONNECTING) {
        socket.onopen = () => socket.close();
        socket.onerror = () => {};
      } else {
        socket.close();
      }
    };
  }, [refreshAll]);

  useEffect(() => {
    const root = document.documentElement;
    const preference = settings?.theme_preference || "system";
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const applyTheme = () => {
      const resolved = preference === "system" ? (mediaQuery.matches ? "dark" : "light") : preference;
      root.dataset.theme = resolved;
      root.style.colorScheme = resolved;
    };
    applyTheme();
    if (preference !== "system") {
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
    setSettings,
    health,
    setHealth,
    observations,
    setObservations,
    worldState,
    setWorldState,
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
    refreshAll,
    saveSettings,
    mutateSetting,
    mutateProviderEnabled,
    runChallengeEvaluation,
    exportProofReport,
  };
}
