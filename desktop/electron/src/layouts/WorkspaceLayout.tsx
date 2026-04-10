import { useDesktopApp } from "../state/DesktopAppContext";
import IntegrationsTab from "../tabs/IntegrationsTab";
import LiveLensTab from "../tabs/LiveLensTab";
import LivingLensTab from "../tabs/LivingLensTab";
import MemorySearchTab from "../tabs/MemorySearchTab";
import SessionReplayTab from "../tabs/SessionReplayTab";
import SettingsTab from "../tabs/SettingsTab";
import SmritiTab from "../tabs/SmritiTab";

export function WorkspaceLayout() {
  const app = useDesktopApp();
  const isSmriti = app.activeTab === "Smriti";
  const runtimeAvailable = app.world.runtimeAvailable;
  const configuredWorldModel = app.world.worldModelStatus?.configured_encoder || "vjepa2";
  const activeWorldModel =
    app.currentJepaTick?.last_tick_encoder_type ||
    app.world.worldModelStatus?.active_backend ||
    app.world.worldModelStatus?.last_tick_encoder_type ||
    "awaiting first tick";
  const worldModelChip = !runtimeAvailable
    ? app.runtimeConnectionLabel
    : app.world.worldModelStatus?.native_ready
      ? `${configuredWorldModel} ready`
      : app.world.worldModelStatus?.preflight_status === "failed" || app.world.worldModelStatus?.preflight_status === "quarantined"
        ? `${configuredWorldModel} quarantined`
        : `${configuredWorldModel} configured`;
  const worldModelRuntimeChip = !runtimeAvailable
    ? app.runtimeConnectionLabel
    : app.currentJepaTick?.degraded || app.world.worldModelStatus?.degraded
      ? `${activeWorldModel} degraded`
      : activeWorldModel === "surrogate"
        ? "surrogate fallback"
      : activeWorldModel === "vjepa2"
        ? "vjepa2 active"
        : activeWorldModel === "awaiting first tick" || activeWorldModel === "not_loaded"
            ? "waiting for first JEPA tick"
            : activeWorldModel;

  return (
    <>
      <section className="hero">
        <div>
          <p className="eyebrow">
            {app.activeTab === "Living Lens"
              ? "Continuous scene mode"
              : isSmriti
                ? "Semantic memory system"
                : "Live runtime"}
          </p>
          <h2>{app.activeTab}</h2>
          <div className="hero-meta">
            <span>{app.world.settings?.primary_perception_provider || "onnx"} proposals</span>
            <span>{worldModelChip}</span>
            <span>{worldModelRuntimeChip}</span>
            <span>{app.world.settings?.reasoning_backend || "cloud"} reasoning</span>
            <span>{isSmriti ? "recall + ingestion active" : app.cameraStatusLabel}</span>
            {isSmriti ? <span>{app.sessionId}</span> : null}
          </div>
        </div>
        <div className="hero-actions">
          {isSmriti ? (
            <>
              <button className="primary" onClick={() => app.setActiveTab("Living Lens")}>
                Open Live Lens
              </button>
              <button onClick={() => app.setActiveTab("Integrations")}>Provider Health</button>
            </>
          ) : (
            <>
              <button className="primary" onClick={app.captureFrame} disabled={!app.camera.cameraReady || app.camera.cameraBusy}>
                Capture Frame
              </button>
              <button onClick={app.analyzeFile}>Analyze File</button>
            </>
          )}
        </div>
      </section>

      {app.activeTab === "Live Lens" ? <LiveLensTab /> : null}
      {app.activeTab === "Living Lens" ? <LivingLensTab /> : null}
      {app.activeTab === "Memory Search" ? <MemorySearchTab /> : null}
      {app.activeTab === "Smriti" ? <SmritiTab /> : null}
      {app.activeTab === "Session Replay" ? <SessionReplayTab /> : null}
      {app.activeTab === "Integrations" ? <IntegrationsTab /> : null}
      {app.activeTab === "Settings" ? <SettingsTab /> : null}
    </>
  );
}
