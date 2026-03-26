import type { CSSProperties, HTMLAttributes } from "react";

export type BaselineBattleHistoryItem = {
  id: string;
  label: string;
  winner: string;
  composite: number;
  continuity?: number | null;
  persistence?: number | null;
  surpriseSeparation?: number | null;
  summary?: string;
};

type BaselineBattleProps = {
  history?: BaselineBattleHistoryItem[];
  className?: string;
  style?: CSSProperties;
} & HTMLAttributes<HTMLDivElement>;

function mergeClassNames(...parts: Array<string | undefined | false>) {
  return parts.filter(Boolean).join(" ");
}

function formatPct(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return "n/a";
  return `${Math.round(value * 100)}%`;
}

export default function BaselineBattle({
  history = [],
  className,
  style,
  ...rest
}: BaselineBattleProps) {
  if (!history.length) {
    return null;
  }

  return (
    <section
      className={mergeClassNames("panel baseline-battle", className)}
      style={style}
      {...rest}
    >
      <div className="panel-head">
        <div>
          <p className="eyebrow">Baseline battle</p>
          <h3>History and score trajectory</h3>
        </div>
      </div>

      <div className="baseline-battle__history">
        {history.map((entry) => (
          <article
            key={entry.id}
            className="baseline-row baseline-battle__row"
            data-winner={entry.winner}
          >
            <div className="baseline-copy">
              <strong>{entry.label}</strong>
              <span className="muted">{entry.summary ?? entry.winner}</span>
            </div>
            <div className="baseline-metrics">
              <div className="baseline-metric">
                <span>Composite</span>
                <div className="baseline-mini-bar">
                  <span style={{ width: `${Math.max(entry.composite * 100, 0)}%` }} />
                </div>
                <strong>{formatPct(entry.composite)}</strong>
              </div>
              <div className="baseline-metric">
                <span>Continuity</span>
                <div className="baseline-mini-bar continuity">
                  <span style={{ width: `${Math.max((entry.continuity ?? 0) * 100, 0)}%` }} />
                </div>
                <strong>{formatPct(entry.continuity)}</strong>
              </div>
            </div>
            <div className="baseline-composite">
              <strong>{entry.winner}</strong>
              <span>Winner</span>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
