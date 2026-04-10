import { challengeStepExpectation } from "../lib/formatting";

type ChallengeGuideProps = {
  challengeGuideActive: boolean;
  challengeStepIndex: number;
  challengeSteps: string[];
  currentChallengeStep?: string | null;
  challengeBusy: boolean;
  onStart: () => void;
  onAdvance: () => void;
  onReset: () => void;
  onScoreLive: () => void;
  onScoreStored: () => void;
};

export default function ChallengeGuide({
  challengeGuideActive,
  challengeStepIndex,
  challengeSteps,
  currentChallengeStep,
  challengeBusy,
  onStart,
  onAdvance,
  onReset,
  onScoreLive,
  onScoreStored,
}: ChallengeGuideProps) {
  return (
    <article className="panel panel--challenge" data-panel="challenge-guide">
      <div className="panel-head">
        <h3>Guided JEPA Challenge</h3>
        <span>{challengeGuideActive ? `step ${challengeStepIndex + 1} / ${challengeSteps.length}` : "ready"}</span>
      </div>
      <p className="muted">
        Run one guided occlusion-and-change sequence. The app compares JEPA / Hybrid mode against frame captioning and embedding retrieval on the same session window.
      </p>
      <div className="status-grid" style={{ gridTemplateColumns: "repeat(2, minmax(0, 1fr))", marginBottom: "1rem" }}>
        <div className="status-metric">
          <span>Score live sequence</span>
          <strong>Grade the latest live camera window</strong>
        </div>
        <div className="status-metric">
          <span>Score stored window</span>
          <strong>Re-score the most recent saved challenge run</strong>
        </div>
      </div>
      <div className="camera-actions">
        <button
          onClick={onStart}
          disabled={challengeBusy}
          title="Reset the guided flow to step 1 and start capturing a fresh challenge sequence."
        >
          {challengeGuideActive ? "Guided run active" : "Start Guided Run"}
        </button>
        <button
          onClick={onAdvance}
          disabled={!challengeGuideActive || challengeStepIndex >= challengeSteps.length - 1}
          title="Move to the next instruction after you physically perform the current step."
        >
          Next Step
        </button>
        <button onClick={onReset} disabled={!challengeGuideActive} title="Stop this run and clear the current guided progress.">
          Reset
        </button>
        <button
          onClick={onScoreLive}
          disabled={challengeBusy}
          title="Score the latest live camera sequence from this session, even if you have not saved a previous challenge window."
        >
          {challengeBusy ? "Scoring" : "Score Live Sequence"}
        </button>
        <button
          onClick={onScoreStored}
          disabled={challengeBusy}
          title="Score the most recently stored challenge window so you can rerun evaluation without capturing a new one."
        >
          Score Stored Window
        </button>
      </div>
      <div className="challenge-stepper">
        {challengeSteps.map((step, index) => (
          <div
            key={step}
            className={
              index === challengeStepIndex && challengeGuideActive
                ? "challenge-step active"
                : index < challengeStepIndex && challengeGuideActive
                  ? "challenge-step complete"
                  : "challenge-step"
            }
          >
            <span>Step {index + 1}</span>
            <strong>{step}</strong>
          </div>
        ))}
      </div>
      <div className="camera-health panel--challenge challenge-current">
        <strong>
          {challengeGuideActive
            ? `Current instruction: step ${challengeStepIndex + 1} of ${challengeSteps.length}`
            : "Current instruction"}
        </strong>
        <p>{currentChallengeStep || challengeSteps[0]}</p>
        <p className="muted">{challengeStepExpectation(challengeGuideActive ? challengeStepIndex : 0)}</p>
        <p className="muted">
          Flow: start the run, follow each physical step, click `Next Step` after performing it, then score the live or stored window.
        </p>
      </div>
    </article>
  );
}
