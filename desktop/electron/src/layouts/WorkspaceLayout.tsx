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
            <span>{app.world.settings?.primary_perception_provider || "onnx"} perception</span>
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
