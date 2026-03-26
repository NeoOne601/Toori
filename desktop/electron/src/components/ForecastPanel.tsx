import type { CSSProperties, HTMLAttributes } from "react";

type ForecastPanelProps = {
  k?: number;
  fe?: number | null;
  monotonic?: boolean;
  llmMs?: number | null;
  jepaMs?: number | null;
  horizonLabel?: string;
  className?: string;
  style?: CSSProperties;
} & HTMLAttributes<HTMLDivElement>;

function mergeClassNames(...parts: Array<string | undefined | false>) {
  return parts.filter(Boolean).join(" ");
}

function formatMs(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return "n/a";
  return `${Math.round(value)} ms`;
}

export default function ForecastPanel({
  k = 1,
  fe = null,
  monotonic = false,
  llmMs = null,
  jepaMs = null,
  horizonLabel = "Forecast horizon",
  className,
  style,
  ...rest
}: ForecastPanelProps) {
  return (
    <section
      className={mergeClassNames("panel forecast-panel", className)}
      style={style}
      {...rest}
    >
      <div className="panel-head">
        <div>
          <p className="eyebrow">Forecast</p>
          <h3>FE(k) and runtime timing</h3>
        </div>
        <span
          className="forecast-panel__badge"
          data-monotonic={monotonic ? "true" : "false"}
        >
          {monotonic ? "Monotonic" : "Non-monotonic"}
        </span>
      </div>

      <div className="forecast-panel__metric">
        <span>FE({k})</span>
        <strong>{fe == null ? "n/a" : fe.toFixed(3)}</strong>
      </div>

      <div className="forecast-panel__timings">
        <div className="forecast-panel__slot">
          <span>LLM</span>
          <strong>{formatMs(llmMs)}</strong>
        </div>
        <div className="forecast-panel__slot">
          <span>JEPA</span>
          <strong>{formatMs(jepaMs)}</strong>
        </div>
      </div>

      <p className="forecast-panel__horizon muted">{horizonLabel}</p>
    </section>
  );
}
