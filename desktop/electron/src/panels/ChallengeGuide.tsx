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
      <div className="camera-actions">
        <button onClick={onStart} disabled={challengeBusy}>
          {challengeGuideActive ? "Guided run active" : "Start Guided Run"}
        </button>
        <button onClick={onAdvance} disabled={!challengeGuideActive || challengeStepIndex >= challengeSteps.length - 1}>
          Next Step
        </button>
        <button onClick={onReset} disabled={!challengeGuideActive}>
          Reset
        </button>
        <button onClick={onScoreLive} disabled={challengeBusy}>
          {challengeBusy ? "Scoring" : "Score Live Sequence"}
        </button>
        <button onClick={onScoreStored} disabled={challengeBusy}>
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
      </div>
    </article>
  );
}
