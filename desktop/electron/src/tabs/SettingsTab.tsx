import { useEffect, useState, type FormEvent } from "react";
import SmritiStorageSettings from "../components/smriti/SmritiStorageSettings";
import { pickFolderPath, runtimeRequest } from "../hooks/useRuntimeBridge";
import { useDesktopApp } from "../state/DesktopAppContext";
import type { WorldModelConfig, WorldModelStatus } from "../types";

export default function SettingsTab() {
  const app = useDesktopApp();
  const settings = app.world.settings;
  const [wmStatus, setWmStatus] = useState<WorldModelStatus | null>(app.world.worldModelStatus);
  const [wmConfig, setWmConfig] = useState<WorldModelConfig | null>(null);
  const [wmLoading, setWmLoading] = useState(true);
  const [wmSaving, setWmSaving] = useState(false);

  useEffect(() => {
    let active = true;

    async function loadWorldModelConfig() {
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
  }, [app.world.setStatus]);

  useEffect(() => {
    if (app.world.worldModelStatus) {
      setWmStatus(app.world.worldModelStatus);
    }
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

  return (
    <form className="panel-grid" onSubmit={handleSubmit}>
      <article className="panel">
        <div className="panel-head">
          <h3>Runtime</h3>
          <span>Session and capture policy</span>
        </div>
        <label className="field">
          <span>Runtime profile</span>
          <input
            value={settings.runtime_profile}
            onChange={(event) => app.world.mutateSetting(["runtime_profile"], event.target.value)}
          />
        </label>
        <label className="field">
          <span>Camera device</span>
          <input
            value={settings.camera_device}
            onChange={(event) => app.world.mutateSetting(["camera_device"], event.target.value)}
          />
        </label>
        <label className="field">
          <span>Theme</span>
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
          <span>Sampling FPS</span>
          <input
            type="number"
            min="0.2"
            max="6"
            step="0.1"
            value={settings.sampling_fps}
            onChange={(event) => app.world.mutateSetting(["sampling_fps"], Number(event.target.value))}
          />
        </label>
        <label className="field">
          <span>Top K</span>
          <input
            type="number"
            min="1"
            max="20"
            value={settings.top_k}
            onChange={(event) => app.world.mutateSetting(["top_k"], Number(event.target.value))}
          />
        </label>
        <label className="field">
          <span>Retention days</span>
          <input
            type="number"
            min="1"
            max="365"
            value={settings.retention_days}
            onChange={(event) => app.world.mutateSetting(["retention_days"], Number(event.target.value))}
          />
        </label>
      </article>

      <article className="panel">
        <div className="panel-head">
          <h3>World Model</h3>
          <span>{wmLoading ? "Checking V-JEPA 2 status" : wmStatus?.configured_encoder || "unavailable"}</span>
        </div>
        <div className="status-grid" style={{ gridTemplateColumns: "repeat(4, minmax(0, 1fr))", marginBottom: "1rem" }}>
          <div className="status-metric">
            <span>Configured</span>
            <strong>{wmStatus?.configured_encoder || "checking..."}</strong>
          </div>
          <div className="status-metric">
            <span>Last tick</span>
            <strong>{wmStatus?.last_tick_encoder_type || "checking..."}</strong>
          </div>
          <div className="status-metric">
            <span>Device</span>
            <strong>{wmStatus?.device || "checking..."}</strong>
          </div>
          <div className="status-metric">
            <span>Mode</span>
            <strong>{wmStatus ? (wmStatus.test_mode ? "test" : "production") : "checking..."}</strong>
          </div>
        </div>
        {wmStatus ? (
          <p className="field-hint">
            Loaded: {wmStatus.model_loaded ? "yes" : "no"} · Frames: {wmStatus.n_frames} · Ticks: {wmStatus.total_ticks} · Telescope:{" "}
            {wmStatus.telescope_test}
          </p>
        ) : null}
        {wmStatus?.degraded ? (
          <p className="field-hint" style={{ color: "var(--danger)", marginBottom: "1rem" }}>
            Runtime degraded at {wmStatus.degrade_stage || "runtime"}: {wmStatus.degrade_reason || "surrogate fallback active"}
          </p>
        ) : null}
        <label className="field">
          <span>
            V-JEPA 2 model path
            <span
              title="Leave empty to use the HuggingFace Hub cache. Pick a local folder if you already downloaded the weights and want to avoid a re-download."
              style={{ cursor: "help", marginLeft: "0.35rem" }}
            >
              ⓘ
            </span>
          </span>
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
          <span>
            Frames per clip
            <span title="0 = auto, 4 = lower memory, 8 = full quality." style={{ cursor: "help", marginLeft: "0.35rem" }}>
              ⓘ
            </span>
          </span>
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
          <span>Primary perception</span>
          <select
            value={settings.primary_perception_provider}
            onChange={(event) =>
              app.world.mutateSetting(["primary_perception_provider"], event.target.value)
            }
          >
            <option value="dinov2">dinov2</option>
            <option value="onnx">onnx</option>
            <option value="basic">basic fallback</option>
          </select>
        </label>
        <label className="field">
          <span>Reasoning backend</span>
          <select
            value={settings.reasoning_backend}
            onChange={(event) => app.world.mutateSetting(["reasoning_backend"], event.target.value)}
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
            onChange={(event) =>
              app.world.mutateSetting(["local_reasoning_disabled"], event.target.checked)
            }
          />
          <span>Disable local reasoning by default</span>
        </label>
        <label className="field">
          <span>ONNX model path</span>
          <input
            value={settings.providers.onnx.model_path || ""}
            onChange={(event) =>
              app.world.mutateSetting(["providers", "onnx", "model_path"], event.target.value)
            }
          />
        </label>
        <label className="field checkbox">
          <input
            type="checkbox"
            checked={settings.providers.onnx.enabled}
            onChange={(event) => app.world.mutateProviderEnabled("onnx", event.target.checked)}
          />
          <span>Enable ONNX perception</span>
        </label>
        <label className="field">
          <span>Ollama host</span>
          <input
            value={settings.providers.ollama.base_url || ""}
            onChange={(event) =>
              app.world.mutateSetting(["providers", "ollama", "base_url"], event.target.value)
            }
          />
        </label>
        <label className="field">
          <span>Ollama model</span>
          <input
            value={settings.providers.ollama.model || ""}
            onChange={(event) =>
              app.world.mutateSetting(["providers", "ollama", "model"], event.target.value)
            }
          />
        </label>
        <label className="field">
          <span>Ollama timeout (s)</span>
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
        <label className="field checkbox">
          <input
            type="checkbox"
            checked={settings.providers.ollama.enabled}
            onChange={(event) => app.world.mutateProviderEnabled("ollama", event.target.checked)}
          />
          <span>Enable Ollama local reasoning</span>
        </label>
      </article>
      {app.world.settingsDirty ? (
        <p className="field-hint" style={{ gridColumn: "1 / -1", margin: 0 }}>
          Draft settings are pending. Save when you want non-theme changes to persist.
        </p>
      ) : null}

      <article className="panel">
        <div className="panel-head">
          <h3>Cloud and MLX</h3>
          <span>Fallback targets</span>
        </div>
        <label className="field">
          <span>Cloud base URL</span>
          <input
            value={settings.providers.cloud.base_url || ""}
            onChange={(event) =>
              app.world.mutateSetting(["providers", "cloud", "base_url"], event.target.value)
            }
          />
        </label>
        <label className="field">
          <span>Cloud model</span>
          <input
            value={settings.providers.cloud.model || ""}
            onChange={(event) =>
              app.world.mutateSetting(["providers", "cloud", "model"], event.target.value)
            }
          />
        </label>
        <label className="field">
          <span>Cloud API key</span>
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
          Keep cloud fallback disabled to spend zero cloud tokens during local capture.
        </p>
        <label className="field checkbox">
          <input
            type="checkbox"
            checked={settings.providers.cloud.enabled}
            onChange={(event) => app.world.mutateProviderEnabled("cloud", event.target.checked)}
          />
          <span>Enable cloud fallback</span>
        </label>
        <label className="field">
          <span>MLX model path</span>
          <input
            value={settings.providers.mlx.model_path || ""}
            onChange={(event) =>
              app.world.mutateSetting(["providers", "mlx", "model_path"], event.target.value)
            }
          />
        </label>
        <label className="field">
          <span>MLX command</span>
          <input
            value={settings.providers.mlx.metadata.command || ""}
            onChange={(event) =>
              app.world.mutateSetting(["providers", "mlx", "metadata", "command"], event.target.value)
            }
          />
        </label>
        <label className="field">
          <span>MLX timeout (s)</span>
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
        <label className="field checkbox">
          <input
            type="checkbox"
            checked={settings.providers.mlx.enabled}
            onChange={(event) => app.world.mutateProviderEnabled("mlx", event.target.checked)}
          />
          <span>Enable MLX local reasoning</span>
        </label>
        <button className="primary" type="submit" disabled={app.world.savingSettings}>
          {app.world.savingSettings ? "Saving" : "Save Settings"}
        </button>
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
