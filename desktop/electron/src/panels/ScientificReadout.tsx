import type { ChallengeRun, SceneState } from "../types";
import ChallengeMetricChart from "../widgets/ChallengeMetricChart";

type ScientificReadoutProps = {
  challengeRun?: ChallengeRun | null;
  history: SceneState[];
};

export default function ScientificReadout({
  challengeRun,
  history,
}: ScientificReadoutProps) {
  return (
    <article className="panel panel--challenge" data-panel="scientific-readout">
      <div className="panel-head">
        <h3>Scientific Readout</h3>
        <span>{challengeRun ? challengeRun.id : "no run yet"}</span>
      </div>
      <div className="stack">
        <div className="camera-health panel--comparison">
          <strong>Challenge metric trend</strong>
          <p>
            Continuity should survive stable periods, surprise should rise when a violation is introduced, and persistence should recover the same track after occlusion.
          </p>
          <ChallengeMetricChart history={history} />
        </div>
        {challengeRun ? (
          <div className="camera-health panel--challenge">
            <strong>{challengeRun.summary}</strong>
            <div className="chips chips--challenge">
              {Object.entries(challengeRun.success_criteria).map(([key, value]) => (
                <span key={key}>{value ? "pass" : "watch"} {key.replace(/_/g, " ")}</span>
              ))}
            </div>
          </div>
        ) : (
          <p className="muted">Scientific scoring will appear here after the first live or stored challenge evaluation.</p>
        )}
      </div>
    </article>
  );
}
