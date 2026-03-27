import type { FormEvent } from "react";
import { useDesktopApp } from "../state/DesktopAppContext";

export default function SettingsTab() {
  const app = useDesktopApp();
  const settings = app.world.settings;

  if (!settings) {
    return null;
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await app.world.saveSettings();
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
            onChange={(event) => app.world.mutateSetting(["theme_preference"], event.target.value)}
          >
            <option value="system">system</option>
            <option value="dark">dark</option>
            <option value="light">light</option>
          </select>
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
          <h3>Providers</h3>
          <span>Perception and reasoning routing</span>
        </div>
        <label className="field">
          <span>Primary perception</span>
          <select
            value={settings.primary_perception_provider}
            onChange={(event) =>
              app.world.mutateSetting(["primary_perception_provider"], event.target.value)
            }
          >
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
    </form>
  );
}
