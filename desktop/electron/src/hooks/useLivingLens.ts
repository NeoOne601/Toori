import { useEffect, useRef, useState } from "react";
import type { AppTab } from "../constants";
import type { LivingLensTickResponse, ProviderHealth } from "../types";

type UseLivingLensOptions = {
  activeTab: AppTab;
  cameraStreamLive: boolean;
  sessionId: string;
  topK: number;
  currentFrameBase64: (mode: "live" | "living") => string | null;
  runtimeRequest: <T>(path: string, method?: string, body?: unknown) => Promise<T>;
  onHealthChange?: (providers: ProviderHealth[]) => void;
  onRefresh?: () => Promise<unknown>;
};

export function useLivingLens({
  activeTab,
  cameraStreamLive,
  sessionId,
  topK,
  currentFrameBase64,
  runtimeRequest,
  onHealthChange,
  onRefresh,
}: UseLivingLensOptions) {
  const [livingLensEnabled, setLivingLensEnabled] = useState(true);
  const [livingLensIntervalS, setLivingLensIntervalS] = useState(6);
  const [livingLensPrompt, setLivingLensPrompt] = useState("");
  const [livingLensBusy, setLivingLensBusy] = useState(false);
  const [livingLensStatus, setLivingLensStatus] = useState("Continuous monitoring is ready");
  const [livingLensResult, setLivingLensResult] = useState<LivingLensTickResponse | null>(null);
  const livingLensInFlightRef = useRef(false);
  const livingLensLastTickRef = useRef(0);

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
      top_k: options.topK || topK,
      proof_mode: "both",
    });
    onHealthChange?.(result.provider_health);
    onRefresh?.().catch(() => undefined);
    return result;
  }

  useEffect(() => {
    if (!livingLensEnabled || activeTab !== "Living Lens" || !cameraStreamLive) {
      return;
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
        topK,
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
  }, [activeTab, cameraStreamLive, livingLensEnabled, livingLensIntervalS, livingLensPrompt, sessionId, topK]);

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
    runLivingLensTick,
  };
}
