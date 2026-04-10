import type { ChallengeRun, RecoveryBenchmarkRun, RolloutComparison, SceneState } from "../types";
import ChallengeMetricChart from "../widgets/ChallengeMetricChart";

function percent(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) {
    return "—";
  }
  return `${Math.round(value * 100)}%`;
}

function winnerLabel(value?: string | null): string {
  switch (value) {
    case "jepa_hybrid":
      return "JEPA / Hybrid";
    case "frame_captioning":
      return "Frame Captioning";
    case "embedding_retrieval":
      return "Embedding Retrieval";
    default:
      return value || "—";
  }
}

function challengeSetLabel(value?: string | null): string {
  switch (value) {
    case "live":
      return "Live sequence";
    case "curated":
      return "Stored window";
    case "both":
      return "Latest session window";
    default:
      return value || "Challenge window";
  }
}

function buildChallengeBrief(challengeRun?: ChallengeRun | null): string {
  if (!challengeRun) {
    return "Run a scored challenge to generate a result brief.";
  }
  if (challengeRun.narration?.trim()) {
    return challengeRun.narration;
  }
  const winner = winnerLabel(challengeRun.baseline_comparison?.winner);
  const passedCriteria = Object.values(challengeRun.success_criteria || {}).filter(Boolean).length;
  return `${challengeSetLabel(challengeRun.challenge_set)} reviewed ${challengeRun.world_state_ids?.length || 0} world states. ${winner} currently leads, and ${passedCriteria}/${Object.keys(challengeRun.success_criteria || {}).length} JEPA checks passed.`;
}

type ScientificReadoutProps = {
  challengeRun?: ChallengeRun | null;
  history: SceneState[];
  rollout?: RolloutComparison | null;
  benchmark?: RecoveryBenchmarkRun | null;
  cameraConnectionState?: string;
  cameraStatusLabel?: string;
};

export default function ScientificReadout({
  challengeRun,
  history,
  rollout,
  benchmark,
  cameraConnectionState,
  cameraStatusLabel,
}: ScientificReadoutProps) {
  const isScoredChallenge = Boolean(challengeRun?.world_state_ids?.length);
  const passedCriteria = challengeRun
    ? Object.values(challengeRun.success_criteria).filter(Boolean).length
    : 0;
  const historySample = history.length;
  const meanContinuity = historySample
    ? history.reduce((sum, state) => sum + state.metrics.temporal_continuity_score, 0) / historySample
    : null;
  const peakSurprise = historySample
    ? Math.max(...history.map((state) => state.metrics.surprise_score))
    : null;
  const recoveredTracks = history.reduce(
    (sum, state) => sum + state.metrics.persistence_signal.recovered_track_ids.length,
    0,
  );
  const occludedTracks = history.reduce(
    (sum, state) => sum + state.metrics.persistence_signal.occluded_track_ids.length,
    0,
  );

  return (
    <article className="panel panel--challenge" data-panel="scientific-readout">
      <div className="panel-head">
        <h3>Scientific Readout</h3>
        <span>{challengeRun ? challengeRun.id : "preview mode"}</span>
      </div>
      <div className="stack">
        <div className="status-grid" style={{ gridTemplateColumns: "repeat(4, minmax(0, 1fr))" }}>
          <div className="status-metric">
            <span>Window</span>
            <strong>{isScoredChallenge ? `${challengeRun?.world_state_ids.length} world states` : `${historySample} frames tracked`}</strong>
          </div>
          <div className="status-metric">
            <span>Criteria</span>
            <strong>{isScoredChallenge ? `${passedCriteria}/${Object.keys(challengeRun?.success_criteria || {}).length} passed` : "preview"}</strong>
          </div>
          <div className="status-metric">
            <span>Winner</span>
            <strong>{isScoredChallenge ? winnerLabel(challengeRun?.baseline_comparison.winner) : "preview"}</strong>
          </div>
          <div className="status-metric">
            <span>Peak surprise</span>
            <strong>{percent(peakSurprise)}</strong>
          </div>
        </div>
        <div className="camera-health panel--comparison">
          <strong>{isScoredChallenge ? "Challenge metric trend" : "Live preview metrics"}</strong>
          <p>
            {isScoredChallenge
              ? "Continuity should survive stable periods, surprise should rise when a violation is introduced, persistence should recover the same track after occlusion, and energy shows patch excitation rather than surprise."
              : "Preview mode shows live continuity, surprise, persistence, and energy before a scored challenge run is created."}
          </p>
          <ChallengeMetricChart history={history} />
        </div>
        <div className="status-grid" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}>
          <div className="status-metric">
            <span>Mean continuity</span>
            <strong>{percent(meanContinuity)}</strong>
          </div>
          <div className="status-metric">
            <span>Occlusion events</span>
            <strong>{occludedTracks}</strong>
          </div>
          <div className="status-metric">
            <span>Recovered tracks</span>
            <strong>{recoveredTracks}</strong>
          </div>
        </div>
        <div className="status-grid" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}>
          <div className="status-metric">
            <span>Challenge window</span>
            <strong>{challengeRun ? challengeRun.window_label || challengeSetLabel(challengeRun.challenge_set) : "preview"}</strong>
          </div>
          <div className="status-metric">
            <span>Camera</span>
            <strong>{cameraConnectionState === "degraded" ? "degraded" : cameraStatusLabel || "waiting"}</strong>
          </div>
          <div className="status-metric">
            <span>Recovery passes</span>
            <strong>
              {benchmark
                ? `${benchmark.scenarios.filter((scenario) => scenario.passed).length}/${benchmark.scenarios.length}`
                : "waiting"}
            </strong>
          </div>
        </div>
        {isScoredChallenge ? (
          <div className="camera-health panel--challenge">
            <strong>{challengeRun?.summary}</strong>
            <p>
              JEPA / Hybrid continuity: {percent(challengeRun?.baseline_comparison.jepa_hybrid.continuity)} ·
              persistence: {percent(challengeRun?.baseline_comparison.jepa_hybrid.persistence)} ·
              surprise separation: {percent(challengeRun?.baseline_comparison.jepa_hybrid.surprise_separation)}
            </p>
            <div className="chips chips--challenge">
              {Object.entries(challengeRun?.success_criteria || {}).map(([key, value]) => (
                <span key={key}>{value ? "pass" : "watch"} {key.replace(/_/g, " ")}</span>
              ))}
            </div>
          </div>
        ) : (
          <p className="muted">Scientific scoring will appear here after the first live or stored challenge evaluation.</p>
        )}
        {challengeRun ? (
          <div className="camera-health panel--comparison">
            <span className="chip chip--narrator">Challenge brief</span>
            <strong style={{ display: "block", marginTop: "var(--space-xs)" }}>{buildChallengeBrief(challengeRun)}</strong>
            <p className="muted" style={{ marginTop: "var(--space-xs)" }}>
              `Score Live Sequence` grades the latest live capture window. `Score Stored Window` re-scores the most recently saved challenge window so you can compare without recapturing.
            </p>
          </div>
        ) : null}
        {rollout ? (
          <div className="camera-health panel--comparison">
            <span className="chip chip--narrator">Recovery branch brief</span>
            <strong style={{ display: "block", marginTop: "var(--space-xs)" }}>{rollout.summary}</strong>
            <p className="muted" style={{ marginTop: "var(--space-xs)" }}>
              This panel explains the current rollout branch selection. It is separate from the scored challenge result above.
            </p>
          </div>
        ) : null}
      </div>
    </article>
  );
}
