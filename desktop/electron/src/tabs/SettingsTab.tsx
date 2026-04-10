import { useEffect, useState, type FormEvent } from "react";
import SmritiStorageSettings from "../components/smriti/SmritiStorageSettings";
import { pickFolderPath, runtimeRequest } from "../hooks/useRuntimeBridge";
import { useDesktopApp } from "../state/DesktopAppContext";
import type { WorldModelConfig, WorldModelStatus } from "../types";

function featureStateLabel(item: { enabled: boolean; healthy: boolean; message?: string | null }) {
  if (!item.enabled) {
    return "off";
  }
  if (item.healthy) {
    return "active";
  }
  return /awaiting|warmup|warming up|connecting/i.test(item.message || "") ? "waiting" : "degraded";
}

function HelpMark({ text }: { text: string }) {
  return (
    <span title={text} style={{ cursor: "help", marginLeft: "0.35rem" }}>
      ⓘ
    </span>
  );
}

function FieldTitle({ label, help }: { label: string; help: string }) {
  return (
    <span>
      {label}
      <HelpMark text={help} />
    </span>
  );
}

export default function SettingsTab() {
  const app = useDesktopApp();
  const settings = app.world.settings;
  const runtimeAvailable = app.world.runtimeAvailable;
  const [wmStatus, setWmStatus] = useState<WorldModelStatus | null>(app.world.worldModelStatus);
  const [wmConfig, setWmConfig] = useState<WorldModelConfig | null>(null);
  const [wmLoading, setWmLoading] = useState(true);
  const [wmSaving, setWmSaving] = useState(false);

  useEffect(() => {
    let active = true;

    async function loadWorldModelConfig() {
      if (!app.world.runtimeAvailable) {
        if (active) {
          setWmLoading(false);
        }
        return;
      }
      if (active) {
        setWmLoading(true);
      }
      try {
        const [status, config] = await Promise.all([
          runtimeRequest<WorldModelStatus>("/v1/world-model/status"),
          runtimeRequest<WorldModelConfig>("/v1/world-model/config"),
        ]);
        if (!active) {
          return;
        }
        setWmStatus(status);
        setWmConfig(config);
      } catch (error) {
        if (active) {
          app.world.setStatus((error as Error).message);
        }
      } finally {
        if (active) {
          setWmLoading(false);
        }
      }
    }

    void loadWorldModelConfig();
    return () => {
      active = false;
    };
  }, [app.world.runtimeAvailable, app.world.setStatus]);

  useEffect(() => {
    setWmStatus(app.world.worldModelStatus);
  }, [app.world.worldModelStatus]);

  if (!settings) {
    return null;
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await app.world.saveSettings();
  }

  async function saveWorldModelConfig() {
    if (!wmConfig) {
      return;
    }
    setWmSaving(true);
    try {
      const nextConfig = await runtimeRequest<WorldModelConfig>("/v1/world-model/config", "PUT", {
        model_path: wmConfig.model_path,
        cache_dir: wmConfig.cache_dir,
        n_frames: wmConfig.n_frames,
      });
      setWmConfig(nextConfig);
      const nextStatus = await runtimeRequest<WorldModelStatus>("/v1/world-model/status");
      setWmStatus(nextStatus);
      app.world.setStatus("World model settings saved");
    } catch (error) {
      app.world.setStatus((error as Error).message);
    } finally {
      setWmSaving(false);
    }
  }

  async function chooseWorldModelPath() {
    const nextPath = await pickFolderPath();
    if (!nextPath) {
      return;
    }
    setWmConfig((current) => (current ? { ...current, model_path: nextPath } : current));
  }

  async function chooseWorldModelCacheDir() {
    const nextPath = await pickFolderPath();
    if (!nextPath) {
      return;
    }
    setWmConfig((current) => (current ? { ...current, cache_dir: nextPath } : current));
  }

  return (
    <form className="panel-grid" onSubmit={handleSubmit}>
      <article className="panel">
        <div className="panel-head">
          <h3>Runtime</h3>
          <span>Session and capture policy</span>
        </div>
        <p className="field-hint" style={{ marginBottom: "1rem" }}>
          Runtime posture is managed automatically. Camera switching and live-analysis pacing are the controls that affect the visible desktop experience here.
        </p>
        <label className="field">
          <FieldTitle
            label="Camera routing"
            help="This is the last camera device identifier used by the runtime. Switch cameras from Live Lens or Living Lens instead of editing this manually."
          />
          <input value={settings.camera_device || "default"} readOnly />
          <small className="field-hint">Use the live camera controls to switch devices. This field is kept for diagnostics only.</small>
        </label>
        <label className="field">
          <FieldTitle
            label="Theme"
            help="Changes the desktop visual theme immediately. It does not affect capture, JEPA, or provider selection."
          />
          <select
            value={settings.theme_preference || "system"}
            onChange={(event) => void app.world.setThemePreference(event.target.value)}
          >
            <option value="system">system</option>
            <option value="dark">dark</option>
            <option value="light">light</option>
            <option value="graphite">graphite</option>
            <option value="sepia">sepia</option>
            <option value="high_contrast_dark">high contrast dark</option>
            <option value="high_contrast_light">high contrast light</option>
          </select>
          <small className="field-hint">Theme saves immediately. Other settings stay in draft until you save.</small>
        </label>
        <label className="field">
          <FieldTitle
            label="Sampling FPS"
            help="Controls how often the runtime samples frames during continuous capture. Higher values feel more responsive but increase CPU and storage pressure."
          />
          <input
            type="number"
            min="0.2"
            max="6"
            step="0.1"
            value={settings.sampling_fps}
            onChange={(event) => app.world.mutateSetting(["sampling_fps"], Number(event.target.value))}
          />
          <small className="field-hint">Raise this if scene changes are missed. Lower it if the machine is under load.</small>
        </label>
        <label className="field">
          <FieldTitle
            label="Top K"
            help="Sets how many nearest memories or retrieval candidates the runtime keeps around for ranking and recap generation."
          />
          <input
            type="number"
            min="1"
            max="20"
            value={settings.top_k}
            onChange={(event) => app.world.mutateSetting(["top_k"], Number(event.target.value))}
          />
          <small className="field-hint">Higher values widen memory search but can make side panels busier.</small>
        </label>
        <label className="field">
          <FieldTitle
            label="Retention days"
            help="Controls how long stored observations and scene history stay in the local runtime before pruning."
          />
          <input
            type="number"
            min="1"
            max="365"
            value={settings.retention_days}
            onChange={(event) => app.world.mutateSetting(["retention_days"], Number(event.target.value))}
          />
          <small className="field-hint">Longer retention improves memory recall but increases local storage usage.</small>
        </label>
      </article>

      <article className="panel">
        <div className="panel-head">
          <h3>World Model</h3>
          <span>
            {runtimeAvailable
              ? wmLoading
                ? "Checking V-JEPA 2 status"
                : wmStatus?.active_backend || wmStatus?.configured_encoder || "unavailable"
              : app.runtimeConnectionLabel}
          </span>
        </div>
        <div className="status-grid" style={{ gridTemplateColumns: "repeat(4, minmax(0, 1fr))", marginBottom: "1rem" }}>
          <div className="status-metric">
            <span>Configured</span>
            <strong>{runtimeAvailable ? wmStatus?.configured_encoder || "checking..." : app.runtimeConnectionLabel}</strong>
          </div>
          <div className="status-metric">
            <span>Active</span>
            <strong>{runtimeAvailable ? wmStatus?.active_backend || "checking..." : app.runtimeConnectionLabel}</strong>
          </div>
          <div className="status-metric">
            <span>Device</span>
            <strong>{runtimeAvailable ? wmStatus?.device || "checking..." : app.runtimeConnectionLabel}</strong>
          </div>
          <div className="status-metric">
            <span>Preflight</span>
            <strong>
              {runtimeAvailable
                ? wmStatus
                  ? wmStatus.preflight_status
                  : "checking..."
                : app.runtimeConnectionLabel}
            </strong>
          </div>
        </div>
        {!runtimeAvailable ? (
          <p className="field-hint" style={{ color: "var(--warning)", marginBottom: "1rem" }}>
            {app.runtimeConnectionLabel} Cached settings remain editable, and this panel will refresh after reconnect.
          </p>
        ) : wmStatus ? (
          <p className="field-hint">
            Loaded: {wmStatus.model_loaded ? "yes" : "no"} · Native ready: {wmStatus.native_ready ? "yes" : "no"} · Frames: {wmStatus.n_frames} · Ticks:{" "}
            {wmStatus.total_ticks} · Telescope: {wmStatus.telescope_test}
          </p>
        ) : null}
        {wmStatus?.degraded ? (
          <p className="field-hint" style={{ color: "var(--danger)", marginBottom: "1rem" }}>
            Runtime degraded at {wmStatus.degrade_stage || "runtime"}: {wmStatus.degrade_reason || "degraded continuity active"}
          </p>
        ) : null}
        {wmStatus?.crash_fingerprint ? (
          <p className="field-hint" style={{ color: "var(--warning)", marginBottom: "1rem" }}>
            Crash fingerprint: {wmStatus.crash_fingerprint}
            {wmStatus.last_failure_at ? ` · Last failure: ${new Date(wmStatus.last_failure_at).toLocaleString()}` : ""}
          </p>
        ) : null}
        {runtimeAvailable && wmStatus?.retryable_native_failure ? (
          <div style={{ display: "flex", gap: "0.75rem", marginBottom: "1rem", alignItems: "center" }}>
            <button
              type="button"
              onClick={() => {
                void app.world.retryNativeJEPA().then(setWmStatus).catch(() => undefined);
              }}
            >
              Retry Native JEPA
            </button>
            <span className="field-hint">
              Process state: {wmStatus.native_process_state || "unknown"}
              {wmStatus.last_native_exit_code != null ? ` · exit ${wmStatus.last_native_exit_code}` : ""}
              {wmStatus.last_native_signal != null ? ` · signal ${wmStatus.last_native_signal}` : ""}
            </span>
          </div>
        ) : null}
        <label className="field">
          <FieldTitle
            label="V-JEPA 2 model path"
            help="Leave empty to use the HuggingFace Hub cache. Point this at a local model folder only if you already downloaded the weights."
          />
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <input
              value={wmConfig?.model_path || ""}
              onChange={(event) =>
                setWmConfig((current) => (current ? { ...current, model_path: event.target.value } : current))
              }
              placeholder="Leave empty to use the HuggingFace cache"
            />
            <button type="button" onClick={() => void chooseWorldModelPath()}>
              Browse
            </button>
          </div>
        </label>
        <label className="field">
          <FieldTitle
            label="HuggingFace cache directory"
            help="This folder is where local V-JEPA downloads are reused. Set it only if you keep model weights outside the default HuggingFace cache."
          />
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <input
              value={wmConfig?.cache_dir || ""}
              onChange={(event) =>
                setWmConfig((current) => (current ? { ...current, cache_dir: event.target.value } : current))
              }
              placeholder="~/.cache/huggingface/hub"
            />
            <button type="button" onClick={() => void chooseWorldModelCacheDir()}>
              Browse
            </button>
          </div>
        </label>
        <label className="field">
          <FieldTitle
            label="Frames per clip"
            help="Auto lets the runtime pick the safest value. Four frames is lighter on memory; eight frames uses the full native clip size when the machine is stable."
          />
          <select
            value={wmConfig?.n_frames ?? 0}
            onChange={(event) =>
              setWmConfig((current) => (current ? { ...current, n_frames: Number(event.target.value) } : current))
            }
          >
            <option value={0}>Auto</option>
            <option value={4}>4</option>
            <option value={8}>8</option>
          </select>
        </label>
        {wmConfig ? (
          <div className="field-hint" style={{ display: "grid", gap: "0.35rem", marginBottom: "1rem" }}>
            <span>Effective model: {wmConfig.effective_model}</span>
            <span>Cache location: {wmConfig.cache_dir}</span>
            <span>Download URL: {wmConfig.download_url}</span>
          </div>
        ) : null}
        <button
          type="button"
          className="primary"
          onClick={() => void saveWorldModelConfig()}
          disabled={wmSaving || !wmConfig}
        >
          {wmSaving ? "Saving" : "Save World Model Settings"}
        </button>
      </article>

      <article className="panel">
        <div className="panel-head">
          <h3>Providers</h3>
          <span>Legacy perception + language routing</span>
        </div>
        <p className="field-hint" style={{ marginBottom: "1rem" }}>
          These controls do not disable V-JEPA 2. ONNX/basic still drive proposal boxes and observation summaries,
          while the reasoning backend only affects language answers.
        </p>
        <label className="field">
          <FieldTitle
            label="Proposal + observation provider"
            help="Controls the non-JEPA visual provider used for proposal boxes and compatibility summaries. It does not replace the native JEPA world model."
          />
          <select
            value={settings.primary_perception_provider}
            onChange={(event) =>
              app.world.mutateSetting(["primary_perception_provider"], event.target.value)
            }
          >
            <option value="dinov2">dinov2</option>
            <option value="onnx">onnx</option>
            <option value="basic">basic compatibility</option>
          </select>
          <small className="field-hint">Use `dinov2` for the strongest local proposal support. `basic` is compatibility-only and should not become the semantic truth source.</small>
        </label>
        <label className="field">
          <FieldTitle
            label="Language reasoning order"
            help="Controls which language backend writes recaps, challenge narration, and open-vocabulary relabeling. It does not change the JEPA tick path."
          />
          <select
            value={settings.reasoning_backend}
            onChange={(event) => app.world.mutateSetting(["reasoning_backend"], event.target.value)}
          >
            <option value="cloud">cloud only</option>
            <option value="ollama">ollama then cloud</option>
            <option value="mlx">mlx then cloud</option>
            <option value="auto">auto local</option>
            <option value="disabled">disabled</option>
          </select>
          <small className="field-hint">`mlx then cloud` keeps local reasoning first. `cloud only` skips local models. `disabled` removes narration-style reasoning altogether.</small>
        </label>
        <label className="field checkbox">
          <input
            type="checkbox"
            checked={settings.local_reasoning_disabled}
            onChange={(event) =>
              app.world.mutateSetting(["local_reasoning_disabled"], event.target.checked)
            }
          />
          <span title="When enabled, MLX and Ollama stay off unless a flow explicitly opts into them. Expect weaker local narration and semantic relabeling.">
            Disable local reasoning by default
          </span>
        </label>
        <label className="field checkbox">
          <input
            type="checkbox"
            checked={settings.providers.onnx.enabled}
            onChange={(event) => app.world.mutateProviderEnabled("onnx", event.target.checked)}
          />
          <span title="Keeps ONNX available for proposal boxes and compatibility perception. This is proposal support, not the active JEPA world model.">
            Enable ONNX proposal support
          </span>
        </label>
        {settings.providers.onnx.enabled ? (
          <label className="field">
            <FieldTitle
              label="ONNX model path"
              help="Local ONNX file used for proposal support when ONNX is enabled."
            />
            <input
              value={settings.providers.onnx.model_path || ""}
              onChange={(event) =>
                app.world.mutateSetting(["providers", "onnx", "model_path"], event.target.value)
              }
            />
          </label>
        ) : null}
        <label className="field checkbox">
          <input
            type="checkbox"
            checked={settings.providers.ollama.enabled}
            onChange={(event) => app.world.mutateProviderEnabled("ollama", event.target.checked)}
          />
          <span title="Allows Ollama to answer local reasoning requests when the selected reasoning order includes it.">
            Enable Ollama local reasoning
          </span>
        </label>
        {settings.providers.ollama.enabled ? (
          <>
            <label className="field">
              <FieldTitle
                label="Ollama host"
                help="Local Ollama server used for optional desktop-only reasoning when enabled."
              />
              <input
                value={settings.providers.ollama.base_url || ""}
                onChange={(event) =>
                  app.world.mutateSetting(["providers", "ollama", "base_url"], event.target.value)
                }
              />
            </label>
            <label className="field">
              <FieldTitle
                label="Ollama model"
                help="Model name served by Ollama for local reasoning attempts."
              />
              <input
                value={settings.providers.ollama.model || ""}
                onChange={(event) =>
                  app.world.mutateSetting(["providers", "ollama", "model"], event.target.value)
                }
              />
            </label>
            <label className="field">
              <FieldTitle
                label="Ollama timeout (s)"
                help="Maximum wait before Ollama is treated as unavailable for the current request."
              />
              <input
                type="number"
                min="10"
                max="600"
                value={settings.providers.ollama.timeout_s}
                onChange={(event) =>
                  app.world.mutateSetting(["providers", "ollama", "timeout_s"], Number(event.target.value))
                }
              />
            </label>
          </>
        ) : null}
      </article>
      {app.world.settingsDirty ? (
        <p className="field-hint" style={{ gridColumn: "1 / -1", margin: 0 }}>
          Draft settings are pending. Save when you want non-theme changes to persist.
        </p>
      ) : null}

      <article className="panel">
        <div className="panel-head">
          <h3>Cloud and MLX</h3>
          <span>Optional reasoning targets</span>
        </div>
        <label className="field">
          <FieldTitle
            label="Cloud base URL"
            help="HTTP endpoint for optional cloud reasoning. Leave cloud disabled if you want a fully local run."
          />
          <input
            value={settings.providers.cloud.base_url || ""}
            onChange={(event) =>
              app.world.mutateSetting(["providers", "cloud", "base_url"], event.target.value)
            }
          />
        </label>
        <label className="field">
          <FieldTitle
            label="Cloud model"
            help="Remote model name used only when cloud reasoning is enabled and selected by the reasoning order."
          />
          <input
            value={settings.providers.cloud.model || ""}
            onChange={(event) =>
              app.world.mutateSetting(["providers", "cloud", "model"], event.target.value)
            }
          />
        </label>
        <label className="field">
          <FieldTitle
            label="Cloud API key"
            help="Stored locally on this machine and used only for optional cloud reasoning calls."
          />
          <input
            type="password"
            autoComplete="off"
            value={settings.providers.cloud.api_key || ""}
            onChange={(event) =>
              app.world.mutateSetting(["providers", "cloud", "api_key"], event.target.value)
            }
          />
          <small className="field-hint">Stored in local Toori runtime settings on this machine.</small>
        </label>
        <p className="field-hint">
          Each cloud reasoning call sends one image, your prompt, and up to five recent memory summaries.
          Keep cloud reasoning disabled to spend zero cloud tokens during local capture.
        </p>
        <label className="field checkbox">
          <input
            type="checkbox"
            checked={settings.providers.cloud.enabled}
            onChange={(event) => app.world.mutateProviderEnabled("cloud", event.target.checked)}
          />
          <span title="Lets Toori call the configured cloud model for optional narration or fallback reasoning.">
            Enable cloud reasoning
          </span>
        </label>
        <label className="field checkbox">
          <input
            type="checkbox"
            checked={settings.providers.mlx.enabled}
            onChange={(event) => app.world.mutateProviderEnabled("mlx", event.target.checked)}
          />
          <span title="Turns on the local Gemma / MLX reasoning path. Expect stronger local narration, but also more memory pressure when the model is first loaded.">
            Enable MLX local reasoning
          </span>
        </label>
        {settings.providers.mlx.enabled ? (
          <>
            <label className="field">
              <FieldTitle
                label="MLX model path"
                help="Local Gemma / MLX weights used for on-device reasoning and semantic relabeling."
              />
              <input
                value={settings.providers.mlx.model_path || ""}
                onChange={(event) =>
                  app.world.mutateSetting(["providers", "mlx", "model_path"], event.target.value)
                }
              />
            </label>
            <label className="field">
              <FieldTitle
                label="MLX command"
                help="Command used to launch the MLX reasoning helper. Change this only if you know the local runtime wiring."
              />
              <input
                value={settings.providers.mlx.metadata.command || ""}
                onChange={(event) =>
                  app.world.mutateSetting(["providers", "mlx", "metadata", "command"], event.target.value)
                }
              />
            </label>
            <label className="field">
              <FieldTitle
                label="MLX timeout (s)"
                help="Maximum wait before MLX is treated as unavailable for the current reasoning request."
              />
              <input
                type="number"
                min="10"
                max="600"
                value={settings.providers.mlx.timeout_s}
                onChange={(event) =>
                  app.world.mutateSetting(["providers", "mlx", "timeout_s"], Number(event.target.value))
                }
              />
            </label>
          </>
        ) : null}
        <button className="primary" type="submit" disabled={app.world.savingSettings}>
          {app.world.savingSettings ? "Saving" : "Save Settings"}
        </button>
      </article>

      <article className="panel">
        <div className="panel-head">
          <h3>Live Features</h3>
          <span>JEPA overlays and narration</span>
        </div>
        <p className="field-hint" style={{ marginBottom: "1rem" }}>
          Dependency order: JEPA tick path → heatmap / entity overlays → open-vocabulary labels → TVLC semantic context.
          If native JEPA is quarantined, overlays can still fall back to degraded continuity while proposal support remains visible.
        </p>
        <label className="field checkbox">
          <input
            type="checkbox"
            checked={settings.live_features?.live_lens_use_jepa_tick !== false}
            onChange={(event) =>
              app.world.mutateSetting(["live_features", "live_lens_use_jepa_tick"], event.target.checked)
            }
          />
          <span title="Turns on JEPA-driven temporal scene analysis for Live Lens and Living Lens. If native JEPA is quarantined, the app falls back to degraded continuity mode.">
            Use JEPA-backed tick path for Live Lens capture
          </span>
        </label>
        <label className="field checkbox">
          <input
            type="checkbox"
            checked={settings.live_features?.energy_heatmap_enabled !== false}
            onChange={(event) =>
              app.world.mutateSetting(["live_features", "energy_heatmap_enabled"], event.target.checked)
            }
          />
          <span title="Shows patch-level change energy. Violet and blue mean lower change; green is stable; yellow, orange, and red show stronger prediction mismatch.">
            Enable energy heatmap overlay
          </span>
        </label>
        <label className="field checkbox">
          <input
            type="checkbox"
            checked={settings.live_features?.entity_overlay_enabled !== false}
            onChange={(event) =>
              app.world.mutateSetting(["live_features", "entity_overlay_enabled"], event.target.checked)
            }
          />
          <span title="Shows tracked boxes and anchors when entity evidence is available. Boxes may still come from proposal support even if JEPA is degraded.">
            Enable entity overlay
          </span>
        </label>
        <label className="field checkbox">
          <input
            type="checkbox"
            checked={settings.live_features?.open_vocab_labels_enabled !== false}
            onChange={(event) =>
              app.world.mutateSetting(["live_features", "open_vocab_labels_enabled"], event.target.checked)
            }
          />
          <span title="Allows semantic relabeling of entities. Best results require healthy local reasoning and trained TVLC context.">
            Enable open-vocabulary labels
          </span>
        </label>
        <label className="field checkbox">
          <input
            type="checkbox"
            checked={settings.live_features?.tvlc_enabled !== false}
            onChange={(event) =>
              app.world.mutateSetting(["live_features", "tvlc_enabled"], event.target.checked)
            }
          />
          <span title="Adds multimodal semantic context to help open-vocabulary labels. If the connector is present but untrained, the impact will be limited.">
            Enable TVLC connector context
          </span>
        </label>
        <div className="stack" style={{ marginTop: "1rem" }}>
          {app.world.featureHealth.map((item) => (
            <div key={item.name} className="list-row compact">
              <div>
                <strong>{item.name}</strong>
                <p>{item.message}</p>
              </div>
              <span>{featureStateLabel(item)}</span>
            </div>
          ))}
        </div>
      </article>

      <article className="panel">
        <div className="panel-head">
          <h3>Smriti Storage</h3>
          <span>Media indexing, watch folders, and disk usage</span>
        </div>
        <SmritiStorageSettings onStatusChange={(msg) => app.world.setStatus(msg)} />
      </article>
    </form>
  );
}
