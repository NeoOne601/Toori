import { useState } from "react";
import { Gemma4AlertBanner } from "../components/Gemma4Panel";
import BaselineBattle from "../components/BaselineBattle";
import ConsumerMode from "../components/ConsumerMode";
import EnergySpectrumLegend from "../components/EnergySpectrumLegend";
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
  const [toolUrl, setToolUrl] = useState("");
  const [toolViewId, setToolViewId] = useState("");
  const [toolEntities, setToolEntities] = useState("");
  const [toolAffordances, setToolAffordances] = useState("");
  const [toolErrors, setToolErrors] = useState("");
  const [toolBusy, setToolBusy] = useState(false);
  const cameraStateLabel =
    app.cameraConnectionState === "blocked"
      ? "Permission blocked"
      : app.cameraConnectionState === "reconnecting"
        ? "Reconnecting"
        : app.cameraConnectionState === "degraded"
          ? "Camera degraded"
          : app.cameraStatusLabel === "camera warming up"
            ? "Warming up"
            : app.cameraConnectionState === "live"
              ? "Live"
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
      {app.currentJepaTick?.gemma4_alert && (
        <Gemma4AlertBanner alert={app.currentJepaTick.gemma4_alert as any} />
      )}
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
              onChange={(event) => {
                void app.setShowEnergyMap(event.target.checked);
              }}
            />
            <span>Energy Map</span>
          </label>
          <label className="field checkbox compact-check">
            <input
              type="checkbox"
              checked={app.showEntities}
              onChange={(event) => {
                void app.setShowEntities(event.target.checked);
              }}
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
      {app.showEnergyMap ? <EnergySpectrumLegend /> : null}

      {app.uiMode === "consumer" ? (
        <section className="consumer-two-pane">
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
            energyWarmup={Boolean(app.currentJepaTick?.warmup)}
            energyStatusLabel={app.energyHeatmapStatus}
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
                  energyWarmup={Boolean(app.currentJepaTick?.warmup)}
                  energyStatusLabel={app.energyHeatmapStatus}
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
                  continuitySignal={app.continuitySignal ?? undefined}
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

          {app.livingSection === "planning" ? (
            <section className="ll-grid living-view-grid">
              <div className="ll-row ll-row--2col">
                <article className="panel panel--comparison">
                  <div className="panel-head">
                    <h3>Recovery Lab</h3>
                    <span>{app.currentSceneState?.state_domain || "camera"}</span>
                  </div>
                  <div className="stack">
                    <div className="status-grid" style={{ gridTemplateColumns: "repeat(4, minmax(0, 1fr))" }}>
                      <div className="status-metric">
                        <span>Grounded entities</span>
                        <strong>{app.currentSceneState?.grounded_entities?.length || 0}</strong>
                      </div>
                      <div className="status-metric">
                        <span>Affordances</span>
                        <strong>{app.currentSceneState?.affordances?.length || 0}</strong>
                      </div>
                      <div className="status-metric">
                        <span>Plan branches</span>
                        <strong>{app.currentRollout?.ranked_branches.length || 0}</strong>
                      </div>
                      <div className="status-metric">
                        <span>Active backend</span>
                        <strong>{app.world.worldModelStatus?.active_backend || app.world.worldModelStatus?.last_tick_encoder_type || "waiting"}</strong>
                      </div>
                    </div>
                    <div className="camera-health panel--stable">
                      <strong>{app.currentRollout?.summary || "Action-conditioned rollout is ready to rank Plan A vs Plan B."}</strong>
                      <p>
                    {app.world.worldModelStatus?.degraded
                          ? `World model degraded at ${app.world.worldModelStatus.degrade_stage || "runtime"}: ${app.world.worldModelStatus.degrade_reason || "degraded continuity active"}`
                          : "Use this lab to compare the current best action path against recovery branches."}
                      </p>
                    </div>
                    <div className="button-row">
                      <button type="button" className="primary" onClick={() => void app.world.runPlanningRollout()}>
                        Refresh Rollout
                      </button>
                      <button type="button" onClick={() => void app.world.runRecoveryBenchmark()}>
                        Run Recovery Benchmark
                      </button>
                    </div>
                    <div className="chips chips--stable">
                      {(app.currentSceneState?.grounded_entities || []).slice(0, 8).map((entity) => (
                        <span key={entity.id}>{entity.label}</span>
                      ))}
                    </div>
                  </div>
                </article>
                <article className="panel panel--memory">
                  <div className="panel-head">
                    <h3>Tool-State Grounding</h3>
                    <span>Manual browser or desktop evidence</span>
                  </div>
                  <div className="stack">
                    <label className="field">
                      <span>Current URL</span>
                      <input value={toolUrl} onChange={(event) => setToolUrl(event.target.value)} placeholder="https://example.com/app" />
                    </label>
                    <label className="field">
                      <span>View or dialog id</span>
                      <input value={toolViewId} onChange={(event) => setToolViewId(event.target.value)} placeholder="checkout-modal" />
                    </label>
                    <label className="field">
                      <span>Visible entities</span>
                      <textarea
                        rows={3}
                        value={toolEntities}
                        onChange={(event) => setToolEntities(event.target.value)}
                        placeholder="button:Submit order, field:Card number, dialog:Payment failed"
                      />
                    </label>
                    <label className="field">
                      <span>Affordances</span>
                      <textarea
                        rows={3}
                        value={toolAffordances}
                        onChange={(event) => setToolAffordances(event.target.value)}
                        placeholder="click:Retry payment, click:Change card, open:Support chat"
                      />
                    </label>
                    <label className="field">
                      <span>Error banners</span>
                      <input value={toolErrors} onChange={(event) => setToolErrors(event.target.value)} placeholder="Payment failed, Button moved" />
                    </label>
                    <button
                      type="button"
                      disabled={toolBusy}
                      onClick={async () => {
                        const parseItems = (value: string) =>
                          value
                            .split(/\n|,/)
                            .map((item) => item.trim())
                            .filter(Boolean);
                        const visibleEntities = parseItems(toolEntities).map((item, index) => {
                          const [kind, ...labelParts] = item.split(":");
                          const label = (labelParts.join(":").trim() || kind.trim());
                          return {
                            id: `tool-entity-${index}-${label.toLowerCase().replace(/\s+/g, "-")}`,
                            label,
                            kind: labelParts.length ? kind.trim() : "ui_element",
                            state_domain: "browser",
                            status: "visible",
                            confidence: 0.82,
                            properties: {},
                          };
                        });
                        const affordances = parseItems(toolAffordances).map((item, index) => {
                          const [kind, ...labelParts] = item.split(":");
                          const label = (labelParts.join(":").trim() || kind.trim());
                          return {
                            id: `tool-affordance-${index}-${label.toLowerCase().replace(/\s+/g, "-")}`,
                            label,
                            kind: labelParts.length ? `browser.${kind.trim()}` : "browser.click",
                            state_domain: "browser",
                            availability: "available",
                            confidence: 0.78,
                            properties: {},
                          };
                        });
                        setToolBusy(true);
                        try {
                          await app.world.observeToolState({
                            state_domain: "browser",
                            current_url: toolUrl || undefined,
                            view_id: toolViewId || undefined,
                            visible_entities: visibleEntities,
                            affordances,
                            error_banners: parseItems(toolErrors),
                          });
                          await app.world.runPlanningRollout();
                        } finally {
                          setToolBusy(false);
                        }
                      }}
                    >
                      {toolBusy ? "Grounding..." : "Observe Tool State"}
                    </button>
                  </div>
                </article>
              </div>
              <div className="ll-row ll-row--2col">
                <article className="panel panel--comparison">
                  <div className="panel-head">
                    <h3>Ranked Rollouts</h3>
                    <span>{app.currentRollout?.chosen_branch_id || "no branch selected"}</span>
                  </div>
                  <div className="stack">
                    {(app.currentRollout?.ranked_branches || []).length ? (
                      app.currentRollout?.ranked_branches.map((branch) => (
                        <div key={branch.id} className="camera-health panel--comparison">
                          <strong>{branch.candidate_action.verb.replace(/_/g, " ")} · risk {branch.risk_score.toFixed(2)}</strong>
                          <p>{branch.predicted_next_state_summary}</p>
                          <div className="chips chips--stable">
                            {branch.failure_predicates.length
                              ? branch.failure_predicates.map((predicate) => <span key={predicate}>{predicate}</span>)
                              : <span>no blockers predicted</span>}
                          </div>
                        </div>
                      ))
                    ) : (
                      <p className="muted">Rollout branches will appear after the first camera tick or tool-state observation.</p>
                    )}
                  </div>
                </article>
                <article className="panel panel--challenge">
                  <div className="panel-head">
                    <h3>Benchmark Summary</h3>
                    <span>{app.currentBenchmark?.winner || "waiting"}</span>
                  </div>
                  <div className="stack">
                    <div className="camera-health panel--challenge">
                      <strong>{app.currentBenchmark?.summary || "Run the benchmark to validate closed-loop recovery."}</strong>
                    </div>
                    <div className="chips chips--challenge">
                      {(app.currentBenchmark?.scenarios || []).map((scenario) => (
                        <span key={scenario.id}>{scenario.passed ? "pass" : "watch"} {scenario.title}</span>
                      ))}
                    </div>
                  </div>
                </article>
              </div>
            </section>
          ) : null}

          {app.livingSection === "challenge" ? (
            <section className="ll-grid living-view-grid">
              <div className="ll-row ll-row--2col">
                <SceneMonitor
                  title="Challenge Capture"
                  subtitle={app.currentSceneState ? `scene ${app.currentSceneState.id.slice(0, 8)}` : "waiting for challenge frames"}
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
                  toneClassName="panel--live"
                  overlay={
                    <div className="living-overlay">
                      <span className={`video-status-dot is-${app.cameraConnectionState}`} title={cameraStateLabel} />
                    </div>
                  }
                  footer={
                    <div className="signal-grid">
                      <div className="diagnostic-card panel--live">
                        <span>Status</span>
                        <strong>{app.livingLens.livingLensStatus}</strong>
                      </div>
                      <div className="diagnostic-card panel--stable">
                        <span>Step</span>
                        <strong>
                          {app.challengeGuideActive
                            ? `${app.challengeStepIndex + 1}/${app.challengeSteps.length}`
                            : "ready"}
                        </strong>
                      </div>
                      <div className="diagnostic-card panel--persistence">
                        <span>World States</span>
                        <strong>{app.challengeHistory.length}</strong>
                      </div>
                    </div>
                  }
                />
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
                <ScientificReadout
                  challengeRun={app.world.challengeRun}
                  history={app.challengeHistory}
                  rollout={app.currentRollout}
                  benchmark={app.currentBenchmark}
                  cameraConnectionState={app.cameraConnectionState}
                  cameraStatusLabel={app.cameraStatusLabel}
                />
              </div>
              <div className="ll-row ll-row--full">
                <BaselineBattle history={app.baselineHistory} />
              </div>
            </section>
          ) : null}
        </section>
      )}
    </>
  );
}
