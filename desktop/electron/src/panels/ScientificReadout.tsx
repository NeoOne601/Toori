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

type ScientificReadoutProps = {
  challengeRun?: ChallengeRun | null;
  history: SceneState[];
  rollout?: RolloutComparison | null;
  benchmark?: RecoveryBenchmarkRun | null;
};

export default function ScientificReadout({
  challengeRun,
  history,
  rollout,
  benchmark,
}: ScientificReadoutProps) {
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
        <span>{challengeRun ? challengeRun.id : "no run yet"}</span>
      </div>
      <div className="stack">
        <div className="status-grid" style={{ gridTemplateColumns: "repeat(4, minmax(0, 1fr))" }}>
          <div className="status-metric">
            <span>Window</span>
            <strong>{challengeRun ? `${challengeRun.world_state_ids.length} world states` : `${historySample} frames tracked`}</strong>
          </div>
          <div className="status-metric">
            <span>Criteria</span>
            <strong>{challengeRun ? `${passedCriteria}/${Object.keys(challengeRun.success_criteria).length} passed` : "waiting"}</strong>
          </div>
          <div className="status-metric">
            <span>Winner</span>
            <strong>{challengeRun ? winnerLabel(challengeRun.baseline_comparison.winner) : "unscored"}</strong>
          </div>
          <div className="status-metric">
            <span>Peak surprise</span>
            <strong>{percent(peakSurprise)}</strong>
          </div>
        </div>
        <div className="camera-health panel--comparison">
          <strong>Challenge metric trend</strong>
          <p>
            Continuity should survive stable periods, surprise should rise when a violation is introduced, and persistence should recover the same track after occlusion.
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
            <span>Plan branches</span>
            <strong>{rollout?.ranked_branches.length ?? 0}</strong>
          </div>
          <div className="status-metric">
            <span>Plan A risk</span>
            <strong>{rollout?.ranked_branches[0] ? percent(1 - rollout.ranked_branches[0].risk_score) : "—"}</strong>
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
        {challengeRun ? (
          <div className="camera-health panel--challenge">
            <strong>{challengeRun.summary}</strong>
            <p>
              JEPA / Hybrid continuity: {percent(challengeRun.baseline_comparison.jepa_hybrid.continuity)} ·
              persistence: {percent(challengeRun.baseline_comparison.jepa_hybrid.persistence)} ·
              surprise separation: {percent(challengeRun.baseline_comparison.jepa_hybrid.surprise_separation)}
            </p>
            <div className="chips chips--challenge">
              {Object.entries(challengeRun.success_criteria).map(([key, value]) => (
                <span key={key}>{value ? "pass" : "watch"} {key.replace(/_/g, " ")}</span>
              ))}
            </div>
          </div>
        ) : (
          <p className="muted">Scientific scoring will appear here after the first live or stored challenge evaluation.</p>
        )}
        {rollout ? (
          <div className="camera-health panel--comparison">
            <span className="chip chip--narrator">Gemma 4 Narrator</span>
            <strong style={{ display: "block", marginTop: "var(--space-xs)" }}>{rollout.summary}</strong>
            <p className="muted" style={{ marginTop: "var(--space-xs)" }}>
              The world model compares the current best action branch (Plan A) against fallback recovery routes (Plan B) to ensure continuity without relying on cloud reasoning.
            </p>
          </div>
        ) : null}
      </div>
    </article>
  );
}
