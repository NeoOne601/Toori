import { useEffect, useRef, useState } from "react";
import type { AppTab } from "../constants";
import type { LivingLensTickResponse, ProviderHealth } from "../types";

type UseLivingLensOptions = {
  activeTab: AppTab;
  cameraStreamLive: boolean;
  cameraReady: boolean;
  sessionId: string;
  topK: number;
  currentFrameBase64: (mode: "live" | "living") => string | null;
  runtimeRequest: <T>(path: string, method?: string, body?: unknown) => Promise<T>;
  onHealthChange?: (providers: ProviderHealth[]) => void;
  onResult?: (result: LivingLensTickResponse) => void;
  onError?: (error: Error) => void;
  onRefresh?: () => Promise<unknown>;
};

export function useLivingLens({
  activeTab,
  cameraStreamLive,
  cameraReady,
  sessionId,
  topK,
  currentFrameBase64,
  runtimeRequest,
  onHealthChange,
  onResult,
  onError,
  onRefresh,
}: UseLivingLensOptions) {
  const [livingLensEnabled, setLivingLensEnabled] = useState(true);
  const [livingLensIntervalS, setLivingLensIntervalS] = useState(6);
  const [livingLensPrompt, setLivingLensPrompt] = useState("");
  const [livingLensBusy, setLivingLensBusy] = useState(false);
  const [livingLensStatus, setLivingLensStatus] = useState("Continuous monitoring is ready");
  const [livingLensResult, setLivingLensResult] = useState<LivingLensTickResponse | null>(null);
  const [livingLensLastSuccessAt, setLivingLensLastSuccessAt] = useState<number | null>(null);
  const [livingLensLastError, setLivingLensLastError] = useState<string | null>(null);
  const livingLensInFlightRef = useRef(false);
  const livingLensLastTickRef = useRef(0);
  const livingLensLastReconcileRef = useRef(0);

  function isTransportFailure(error: unknown) {
    const message = (error as Error)?.message || "";
    return /runtime unreachable|failed to fetch|fetch failed/i.test(message);
  }

  async function sleep(ms: number) {
    await new Promise((resolve) => window.setTimeout(resolve, ms));
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
    const requestBody = {
      image_base64: imageBase64,
      session_id: sessionId,
      query: options.query || undefined,
      decode_mode: options.decodeMode,
      top_k: options.topK || topK,
      proof_mode: "both",
    };
    let result: LivingLensTickResponse;
    try {
      result = await runtimeRequest<LivingLensTickResponse>("/v1/living-lens/tick", "POST", requestBody);
    } catch (error) {
      if (!isTransportFailure(error)) {
        throw error;
      }
      await sleep(300);
      result = await runtimeRequest<LivingLensTickResponse>("/v1/living-lens/tick", "POST", requestBody);
    }
    return result;
  }

  useEffect(() => {
    if (activeTab !== "Living Lens") {
      return;
    }
    if (livingLensEnabled) {
      if (livingLensStatus === "Auto analyze paused") {
        setLivingLensStatus("Continuous monitoring is ready");
      }
      return;
    }
    livingLensInFlightRef.current = false;
    livingLensLastTickRef.current = 0;
    setLivingLensBusy(false);
    setLivingLensResult(null);
    setLivingLensLastSuccessAt(null);
    setLivingLensLastError(null);
    setLivingLensStatus("Auto analyze paused");
  }, [activeTab, livingLensEnabled, livingLensStatus]);

  useEffect(() => {
    if (!livingLensEnabled || activeTab !== "Living Lens" || !cameraStreamLive || !cameraReady) {
      return;
    }
    let cancelled = false;
    const tick = () => {
      if (livingLensInFlightRef.current) {
        return;
      }
      const intervalMs = Math.max(2, livingLensIntervalS) * 1000;
      if (Date.now() - livingLensLastTickRef.current < intervalMs) {
        return;
      }
      livingLensInFlightRef.current = true;
      setLivingLensBusy(true);
      setLivingLensStatus("Analyzing live scene");
      runLivingLensTick({
        query: livingLensPrompt.trim() || undefined,
        decodeMode: livingLensPrompt.trim() ? "force" : "auto",
        topK,
      })
        .then((result) => {
          if (cancelled) {
            return;
          }
          if (!result) {
            setLivingLensLastError("Waiting for a usable frame");
            setLivingLensStatus("Waiting for a usable frame");
            return;
          }
          onHealthChange?.(result.provider_health);
          onResult?.(result);
          const now = Date.now();
          if (onRefresh && now - livingLensLastReconcileRef.current >= 4000) {
            livingLensLastReconcileRef.current = now;
            onRefresh().catch(() => undefined);
          }
          livingLensLastTickRef.current = now;
          setLivingLensResult(result);
          setLivingLensLastSuccessAt(now);
          setLivingLensLastError(null);
          setLivingLensStatus(`Updated ${new Date().toLocaleTimeString()}`);
        })
        .catch((error) => {
          if (cancelled) {
            return;
          }
          const nextError = error as Error;
          const retryDelayMs = isTransportFailure(nextError) ? 1500 : 2000;
          livingLensLastTickRef.current = Date.now() - Math.max(0, intervalMs - retryDelayMs);
          setLivingLensResult(null);
          setLivingLensLastSuccessAt(null);
          setLivingLensLastError(nextError.message);
          setLivingLensStatus(nextError.message);
          onError?.(nextError);
        })
        .finally(() => {
          if (cancelled) {
            return;
          }
          livingLensInFlightRef.current = false;
          setLivingLensBusy(false);
        });
    };

    tick();
    const interval = window.setInterval(tick, 1200);
    return () => {
      cancelled = true;
      livingLensInFlightRef.current = false;
      setLivingLensBusy(false);
      window.clearInterval(interval);
    };
  }, [activeTab, cameraReady, cameraStreamLive, livingLensEnabled, livingLensIntervalS, livingLensPrompt, sessionId, topK]);

  return {
    livingLensEnabled,
    setLivingLensEnabled,
    livingLensIntervalS,
    setLivingLensIntervalS,
    livingLensPrompt,
    setLivingLensPrompt,
    livingLensBusy,
    livingLensStatus,
    setLivingLensStatus,
    livingLensResult,
    setLivingLensResult,
    livingLensLastSuccessAt,
    setLivingLensLastSuccessAt,
    livingLensLastError,
    setLivingLensLastError,
    runLivingLensTick,
  };
}
