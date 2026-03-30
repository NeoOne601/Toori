import BaselineBattle from "../components/BaselineBattle";
import ConsumerMode from "../components/ConsumerMode";
import ForecastPanel from "../components/ForecastPanel";
import OcclusionPanel from "../components/OcclusionPanel";
import { LIVING_SECTIONS } from "../constants";
import { sigregBadge, sigregColor } from "../lib/formatting";
import ChallengeGuide from "../panels/ChallengeGuide";
import EntityTracksPanel from "../panels/EntityTracksPanel";
import LiveUnderstanding from "../panels/LiveUnderstanding";
import MemoryRelinking from "../panels/MemoryRelinking";
import SceneMonitor from "../panels/SceneMonitor";
import ScenePulse from "../panels/ScenePulse";
import ScientificReadout from "../panels/ScientificReadout";
import { useDesktopApp } from "../state/DesktopAppContext";
import BaselineScoreboard from "../widgets/BaselineScoreboard";
import KPICard from "../widgets/KPICard";

export default function LivingLensTab() {
  const app = useDesktopApp();
  const cameraStateLabel =
    app.cameraConnectionState === "live"
      ? "Live"
      : app.cameraConnectionState === "reconnecting"
        ? "Reconnecting"
        : app.cameraConnectionState === "blocked"
          ? "Blocked"
          : app.cameraConnectionState === "degraded"
            ? "Degraded"
            : "Offline";

  const scienceMonitorFooter = (
    <div className="signal-grid">
      <div className="diagnostic-card panel--live">
        <span>Status</span>
        <strong>{app.livingLens.livingLensStatus}</strong>
      </div>
      <div className="diagnostic-card panel--stable">
        <span>Camera</span>
        <strong>{app.camera.cameraReady ? "ready" : app.camera.cameraStreamLive ? "live" : "idle"}</strong>
      </div>
      <div className="diagnostic-card panel--persistence">
        <span>World State</span>
        <strong>{app.currentSceneState ? "tracking" : "warming up"}</strong>
      </div>
    </div>
  );

  return (
    <>
      <header className="global-controls-header">
        <div className="global-controls-left">
          <label className="field checkbox compact-check">
            <input
              type="checkbox"
              checked={app.livingLens.livingLensEnabled}
              onChange={(event) => app.livingLens.setLivingLensEnabled(event.target.checked)}
            />
            <span>Auto analyze</span>
          </label>
          <label className="field checkbox compact-check">
            <input
              type="checkbox"
              checked={app.showEnergyMap}
              onChange={(event) => app.setShowEnergyMap(event.target.checked)}
            />
            <span>Energy Map</span>
          </label>
          <label className="field checkbox compact-check">
            <input
              type="checkbox"
              checked={app.showEntities}
              onChange={(event) => app.setShowEntities(event.target.checked)}
            />
            <span>Entities</span>
          </label>
          <div className="interval-control">
            <span className="interval-label">Interval</span>
            <input
              type="number"
              min="2"
              max="30"
              value={app.livingLens.livingLensIntervalS}
              onChange={(event) => app.livingLens.setLivingLensIntervalS(Number(event.target.value) || 6)}
            />
            <small>s</small>
          </div>
          <div className="camera-runtime-chip">
            <span className={`camera-status-dot is-${app.cameraConnectionState}`} title={cameraStateLabel} />
            <span className="camera-runtime-label">{cameraStateLabel}</span>
            <span className="camera-device-chip">
              {app.camera.cameraDiagnostics.selectedLabel || "Auto camera"}
            </span>
          </div>
        </div>
        <div className="layout-switcher">
          <button onClick={() => app.camera.retryCamera(false)} disabled={app.camera.cameraBusy}>
            {app.camera.cameraBusy ? "Reconnecting..." : "Reconnect Camera"}
          </button>
          <div className="segmented-control">
            <button
              className={app.uiMode === "consumer" ? "tab active" : "tab"}
              onClick={() => app.setUiMode("consumer")}
            >
              Consumer
            </button>
            <button
              className={app.uiMode === "science" ? "tab active" : "tab"}
              onClick={() => app.setUiMode("science")}
            >
              Science
            </button>
          </div>
          <button onClick={app.world.exportProofReport} disabled={app.world.exportingProof}>
            {app.world.exportingProof ? "Exporting..." : "Export Proof"}
          </button>
        </div>
      </header>

      {app.uiMode === "consumer" ? (
        <section className="panel-grid lens-grid">
          <SceneMonitor
            title="Consumer Mode"
            subtitle={app.consumerText}
            videoRef={app.camera.livingVideoRef}
            captureCanvasRef={app.camera.livingCaptureCanvasRef}
            diagnosticsCanvasRef={app.camera.livingDiagnosticsCanvasRef}
            boxes={app.livingBoxes}
            showEntities={app.showEntities}
            showEnergyMap={app.showEnergyMap}
            energyMap={app.currentJepaTick?.energy_map || []}
            ghosts={app.ghostBoxes}
            anchors={app.energyAnchors}
            uiMode="consumer"
            overlay={
              <div className="living-overlay">
                <span className={`video-status-dot is-${app.cameraConnectionState}`} title={cameraStateLabel} />
              </div>
            }
            footer={
              <div className="camera-health">
                <strong>{app.consumerText}</strong>
                <p>{app.worldModelSummary}</p>
              </div>
            }
          />
          <ConsumerMode
            copy={{
              title: "What Toori Knows",
              subtitle: app.consumerText,
              actionLabel: app.world.exportingProof ? "Exporting..." : "Export Proof",
              emptyLabel: "Waiting for live world-state links",
              statusLabel: "Tracked entities",
            }}
            nodes={app.consumerNodes}
            links={app.consumerLinks}
            onAction={app.world.exportProofReport}
          />
        </section>
      ) : (
        <section className="living-shell">
          <div className="living-subnav-container">
            <div className="living-subnav" role="tablist" aria-label="Living Lens workspaces">
              {LIVING_SECTIONS.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  role="tab"
                  aria-selected={app.livingSection === item.id}
                  className={app.livingSection === item.id ? "living-subtab active" : "living-subtab"}
                  onClick={() => app.setLivingSection(item.id)}
                >
                  <strong>{item.label}</strong>
                  <span>{item.detail}</span>
                </button>
              ))}
            </div>
          </div>

          {app.livingSection === "overview" ? (
            <section className="ll-grid living-view-grid">
              <div className="ll-row ll-row--3col">
                <SceneMonitor
                  title="Scene Monitor"
                  subtitle={app.currentSceneState ? `scene ${app.currentSceneState.id.slice(0, 8)}` : "warming up"}
                  videoRef={app.camera.livingVideoRef}
                  captureCanvasRef={app.camera.livingCaptureCanvasRef}
                  diagnosticsCanvasRef={app.camera.livingDiagnosticsCanvasRef}
                  boxes={app.livingBoxes}
                  showEntities={app.showEntities}
                  showEnergyMap={app.showEnergyMap}
                  energyMap={app.currentJepaTick?.energy_map || []}
                  ghosts={app.ghostBoxes}
                  anchors={app.energyAnchors}
                  uiMode="science"
                  overlay={
                    <div className="living-overlay">
                      <span className={`video-status-dot is-${app.cameraConnectionState}`} title={cameraStateLabel} />
                    </div>
                  }
                  footer={scienceMonitorFooter}
                />
                <LiveUnderstanding
                  observation={app.livingObservation}
                  summary={app.livingAnswer}
                  assetUrl={app.assetUrl}
                  onShareObservation={app.world.copyObservationShare}
                  chips={
                    app.livingObservation
                      ? [
                          `conf ${app.livingObservation.confidence.toFixed(2)}`,
                          app.livingObservation.providers[0] || "provider",
                        ]
                      : []
                  }
                />
                <ScenePulse
                  sceneState={app.currentSceneState}
                  continuitySignal={app.continuitySignal}
                />
              </div>
              <div className="ll-row ll-row--4kpi">
                <KPICard
                  label="Prediction"
                  value={app.currentSceneState?.metrics.prediction_consistency.toFixed(2) ?? "—"}
                />
                <KPICard
                  label="Surprise"
                  value={app.currentSceneState?.metrics.surprise_score.toFixed(2) ?? "—"}
                />
                <KPICard
                  label="Persistence"
                  value={app.currentSceneState?.metrics.persistence_confidence.toFixed(2) ?? "—"}
                />
                <KPICard
                  label="SigReg Loss"
                  value={app.currentJepaTick?.sigreg_loss?.toFixed(3) ?? "—"}
                  badge={sigregBadge(app.currentJepaTick?.sigreg_loss)}
                  color={sigregColor(app.currentJepaTick?.sigreg_loss)}
                  tooltip="Non-zero value proves representations have not collapsed. If this reaches 0.000, the JEPA proof is invalid."
                  priority="high"
                />
              </div>
              <div className="ll-row ll-row--2col">
                <OcclusionPanel
                  summary={
                    app.persistenceSignal
                      ? `${app.persistenceSignal.occluded_track_ids.length} occluded, ${app.persistenceSignal.recovered_track_ids.length} recovered`
                      : "Ghost recovery and occlusion state"
                  }
                  score={
                    app.currentSceneState?.metrics.occlusion_recovery_score ??
                    app.currentSceneState?.metrics.persistence_confidence ??
                    null
                  }
                  tracks={app.occlusionTracks as any}
                />
                <ForecastPanel
                  k={5}
                  fe={app.fe5}
                  monotonic={app.forecastMonotonic}
                  llmMs={app.world.llmLatencyMs}
                  jepaMs={app.currentJepaTick?.planning_time_ms ?? null}
                  horizonLabel={
                    app.fe1 != null && app.fe2 != null && app.fe5 != null
                      ? `FE(1) ${app.fe1.toFixed(3)} • FE(2) ${app.fe2.toFixed(3)} • FE(5) ${app.fe5.toFixed(3)}`
                      : "Forecast horizon is warming up"
                  }
                />
              </div>
              <div className="ll-row ll-row--full">
                <BaselineBattle history={app.baselineHistory} />
              </div>
            </section>
          ) : null}

          {app.livingSection === "memory" ? (
            <section className="ll-grid living-view-grid">
              <div className="ll-row ll-row--2col">
                <MemoryRelinking
                  title="Continuity Memory"
                  hits={app.livingNearest ? [app.livingNearest] : []}
                  assetUrl={app.assetUrl}
                  emptyLabel="Continuity Memory will populate once the live scene has a relevant prior observation."
                  limit={1}
                  toneClassName="panel--memory"
                />
                <EntityTracksPanel
                  tracks={app.displayedTracks}
                  showAllTracks={app.showAllTracks}
                  onToggleShowAll={() => app.setShowAllTracks((value: boolean) => !value)}
                />
              </div>
              <div className="ll-row ll-row--full">
                <MemoryRelinking
                  title="Related Memory Matches"
                  hits={app.livingMatches}
                  assetUrl={app.assetUrl}
                  emptyLabel="No related memory scenes yet. Continue monitoring to build continuity anchors."
                  limit={8}
                  toneClassName="panel--memory"
                />
              </div>
            </section>
          ) : null}

          {app.livingSection === "challenge" ? (
            <section className="ll-grid living-view-grid">
              <div className="ll-row ll-row--2col">
                <ChallengeGuide
                  challengeGuideActive={app.challengeGuideActive}
                  challengeStepIndex={app.challengeStepIndex}
                  challengeSteps={app.challengeSteps}
                  currentChallengeStep={app.currentChallengeStep}
                  challengeBusy={app.world.challengeBusy}
                  onStart={app.startGuidedChallenge}
                  onAdvance={app.advanceGuidedChallenge}
                  onReset={app.resetGuidedChallenge}
                  onScoreLive={() => app.world.runChallengeEvaluation("live")}
                  onScoreStored={() => app.world.runChallengeEvaluation("curated")}
                />
                <article className="panel panel--comparison" data-panel="baseline-scoreboard">
                  <div className="panel-head">
                    <h3>Baseline Scoreboard</h3>
                    <span>{app.livingBaseline ? "scored" : "waiting"}</span>
                  </div>
                  {app.livingBaseline ? (
                    <BaselineScoreboard comparison={app.livingBaseline} />
                  ) : (
                    <p className="muted">
                      Run the guided challenge or score a stored window to compare JEPA / Hybrid mode with the baselines.
                    </p>
                  )}
                </article>
              </div>
              <div className="ll-row ll-row--full">
                <BaselineBattle history={app.baselineHistory} />
              </div>
              <div className="ll-row ll-row--full">
                <ScientificReadout challengeRun={app.world.challengeRun} history={app.challengeHistory} />
              </div>
            </section>
          ) : null}
        </section>
      )}
    </>
  );
}
