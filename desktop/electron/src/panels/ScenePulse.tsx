import type { ContinuitySignal, SceneState } from "../types";

type ScenePulseProps = {
  sceneState?: SceneState | null;
  continuitySignal?: ContinuitySignal;
};

export default function ScenePulse({ sceneState, continuitySignal }: ScenePulseProps) {
  return (
    <article className="panel panel--change" data-panel="scene-pulse">
      <div className="panel-head">
        <h3>Scene Pulse</h3>
        <span>{sceneState ? "active" : "waiting"}</span>
      </div>
      {sceneState ? (
        <div className="stack">
          <div className="camera-health panel--stable">
            <small>Stable Anchors</small>
            <div className="chips chips--stable">
              {(continuitySignal?.stable_elements || []).slice(0, 3).map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
          </div>
          <div className="camera-health panel--change">
            <small>Changes</small>
            <div className="chips chips--changed">
              {(continuitySignal?.changed_elements || []).slice(0, 3).map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <p className="muted">Waiting for state...</p>
      )}
    </article>
  );
}
