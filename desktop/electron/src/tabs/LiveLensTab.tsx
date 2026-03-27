import { useDesktopApp } from "../state/DesktopAppContext";
import LiveUnderstanding from "../panels/LiveUnderstanding";
import MemoryRelinking from "../panels/MemoryRelinking";
import SceneMonitor from "../panels/SceneMonitor";

export default function LiveLensTab() {
  const app = useDesktopApp();
  const { camera, world } = app;

  return (
    <section className="panel-grid lens-grid">
      <SceneMonitor
        title="Camera"
        subtitle={`${world.settings?.primary_perception_provider || "onnx"} -> ${world.settings?.reasoning_backend || "cloud"}`}
        videoRef={camera.liveVideoRef}
        captureCanvasRef={camera.liveCaptureCanvasRef}
        diagnosticsCanvasRef={camera.liveDiagnosticsCanvasRef}
        boxes={app.liveBoxes}
        showEntities={app.showEntities}
        showEnergyMap={app.showEnergyMap}
        energyMap={app.currentJepaTick?.energy_map || []}
        ghosts={app.ghostBoxes}
        anchors={app.energyAnchors}
        uiMode={app.uiMode}
        controls={
          <>
            <div className="camera-controls">
              <label className="field">
                <span>Camera Device</span>
                <select
                  value={camera.selectedCameraId}
                  onChange={(event) => {
                    camera.setSelectedCameraId(event.target.value);
                    camera
                      .startCamera({ preferredDeviceId: event.target.value, phase: "camera retrying" })
                      .catch(() => undefined);
                  }}
                >
                  {camera.cameraDevices.length ? (
                    camera.cameraDevices.map((device) => (
                      <option key={device.deviceId} value={device.deviceId}>
                        {device.label}
                      </option>
                    ))
                  ) : (
                    <option value="default">Auto camera</option>
                  )}
                </select>
              </label>
              <div className="camera-actions">
                <button
                  onClick={() => camera.retryCamera(false)}
                  disabled={
                    camera.cameraBusy ||
                    (camera.desktopBridgeAvailable &&
                      !camera.cameraAccess.granted &&
                      camera.cameraAccess.status !== "unknown")
                  }
                >
                  Retry Camera
                </button>
                <button
                  onClick={camera.switchCamera}
                  disabled={
                    camera.cameraBusy ||
                    camera.cameraDevices.length < 2 ||
                    (camera.desktopBridgeAvailable &&
                      !camera.cameraAccess.granted &&
                      camera.cameraAccess.status !== "unknown")
                  }
                >
                  Switch Camera
                </button>
                <button
                  onClick={() => camera.retryCamera(true)}
                  disabled={
                    camera.cameraBusy ||
                    (camera.desktopBridgeAvailable &&
                      !camera.cameraAccess.granted &&
                      camera.cameraAccess.status !== "unknown")
                  }
                >
                  Auto Pick
                </button>
              </div>
            </div>
            <label className="field">
              <span>Prompt</span>
              <textarea
                rows={3}
                placeholder="Ask a question or leave blank to let auto-decoding describe the scene."
                value={app.prompt}
                onChange={(event) => app.setPrompt(event.target.value)}
              />
            </label>
          </>
        }
        footer={
          <>
            <div className="camera-diagnostics">
              <div className="diagnostic-card">
                <span>Phase</span>
                <strong>{camera.cameraDiagnostics.phase}</strong>
              </div>
              <div className="diagnostic-card">
                <span>Camera</span>
                <strong>{camera.cameraDiagnostics.selectedLabel}</strong>
              </div>
              <div className="diagnostic-card">
                <span>Permission</span>
                <strong>{camera.cameraDiagnostics.permissionStatus}</strong>
              </div>
              <div className="diagnostic-card">
                <span>Resolution</span>
                <strong>{camera.cameraDiagnostics.resolution}</strong>
              </div>
              <div className="diagnostic-card">
                <span>Track</span>
                <strong>{camera.cameraDiagnostics.trackState}</strong>
              </div>
              <div className="diagnostic-card">
                <span>Ready State</span>
                <strong>{camera.cameraDiagnostics.readyState}</strong>
              </div>
              <div className="diagnostic-card">
                <span>Last Frame</span>
                <strong>{camera.cameraDiagnostics.lastFrameAt || "n/a"}</strong>
              </div>
            </div>
            <div className={camera.cameraDiagnostics.blackFrameDetected ? "camera-health warning" : "camera-health"}>
              <strong>{camera.cameraDiagnostics.message}</strong>
              <p>
                track enabled: {String(camera.cameraDiagnostics.trackEnabled)} | muted:{" "}
                {String(camera.cameraDiagnostics.trackMuted)}
                {camera.cameraDiagnostics.frameLuma != null
                  ? ` | luma ${camera.cameraDiagnostics.frameLuma.toFixed(1)}`
                  : ""}
              </p>
            </div>
            {camera.cameraAccess.status !== "unknown" && !camera.cameraAccess.granted ? (
              <div className="camera-health warning">
                <strong>
                  {camera.desktopBridgeAvailable
                    ? "Electron does not currently have camera permission."
                    : "Camera permission is not currently granted."}
                </strong>
                <p>
                  Status: {camera.cameraAccess.status}. Use the buttons below to request access again
                  {camera.desktopBridgeAvailable
                    ? ", open the macOS camera privacy settings, or switch to the browser runtime while the packaged macOS app identity is being fixed."
                    : "."}
                </p>
                <div className="camera-actions">
                  <button onClick={camera.requestCameraPermission} disabled={camera.cameraBusy}>
                    Request Camera Access
                  </button>
                  {camera.desktopBridgeAvailable ? (
                    <button
                      onClick={() => {
                        const openCameraSettings = window.tooriDesktop?.openCameraSettings;
                        openCameraSettings?.().catch(() => undefined);
                      }}
                    >
                      Open Camera Settings
                    </button>
                  ) : null}
                  {camera.desktopBridgeAvailable ? (
                    <button onClick={() => window.open("http://127.0.0.1:4173/", "_blank", "noopener,noreferrer")}>
                      Open Browser Runtime
                    </button>
                  ) : null}
                </div>
              </div>
            ) : null}
          </>
        }
      />

      <LiveUnderstanding
        title="Latest Observation"
        observation={app.latestObservation}
        summary={app.latestObservationSummary}
        assetUrl={app.assetUrl}
        trace={world.analysis?.reasoning_trace || []}
        chips={
          app.latestObservation
            ? [
                `novelty ${app.latestObservation.novelty.toFixed(2)}`,
                `confidence ${app.latestObservation.confidence.toFixed(2)}`,
                app.latestObservation.providers.join(" + "),
              ]
            : []
        }
        emptyLabel="No observations captured yet."
      />

      <MemoryRelinking
        title="Nearest Memory"
        hits={world.analysis?.hits || []}
        assetUrl={app.assetUrl}
        emptyLabel="No related scenes yet."
        limit={6}
        toneClassName="panel--memory"
      />
    </section>
  );
}
