import type { BaselineComparison } from "../types";

type BaselineScoreboardProps = {
  comparison: BaselineComparison;
};

export default function BaselineScoreboard({ comparison }: BaselineScoreboardProps) {
  const rows = [
    { label: "JEPA / Hybrid", tone: "accent", value: comparison.jepa_hybrid },
    { label: "Frame captioning", tone: "neutral", value: comparison.frame_captioning },
    { label: "Embedding retrieval", tone: "neutral", value: comparison.embedding_retrieval },
  ];
  const winner = rows.reduce((best, row) =>
    row.value.composite > best.value.composite ? row : best,
  rows[0]);

  return (
    <div className="baseline-scoreboard">
      {rows.map((row) => (
        <div
          key={row.label}
          className={row.label === winner.label ? "baseline-row is-winner" : "baseline-row"}
          data-tone={row.tone}
        >
          <div className="baseline-copy">
            <strong>{row.label}</strong>
            <p className="muted">
              {row.label === winner.label
                ? "Current best performer on this sequence."
                : "Comparison baseline for the same sequence."}
            </p>
          </div>
          <div className="baseline-metrics">
            <div className="baseline-metric">
              <span>Continuity</span>
              <div className="baseline-mini-bar continuity">
                <span style={{ width: `${Math.max(8, Math.round(row.value.continuity * 100))}%` }} />
              </div>
              <strong>{row.value.continuity.toFixed(2)}</strong>
            </div>
            <div className="baseline-metric">
              <span>Persistence</span>
              <div className="baseline-mini-bar persistence">
                <span style={{ width: `${Math.max(8, Math.round(row.value.persistence * 100))}%` }} />
              </div>
              <strong>{row.value.persistence.toFixed(2)}</strong>
            </div>
            <div className="baseline-metric">
              <span>Change separation</span>
              <div className="baseline-mini-bar surprise">
                <span style={{ width: `${Math.max(8, Math.round(row.value.surprise_separation * 100))}%` }} />
              </div>
              <strong>{row.value.surprise_separation.toFixed(2)}</strong>
            </div>
          </div>
          <div className="baseline-composite">
            <span>Overall</span>
            <strong>{row.value.composite.toFixed(2)}</strong>
          </div>
        </div>
      ))}
      <p className="muted baseline-summary">{comparison.summary}</p>
    </div>
  );
}
