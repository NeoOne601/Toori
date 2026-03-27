import type { SceneState } from "../types";

type ChallengeMetricChartProps = {
  history: SceneState[];
};

export default function ChallengeMetricChart({ history }: ChallengeMetricChartProps) {
  if (!history.length) {
    return (
      <div className="metric-chart empty">
        <p className="muted">
          Metric trend lines will appear once Living Lens has recorded a few world-state updates.
        </p>
      </div>
    );
  }

  const points = [...history].slice(0, 8).reverse();
  return (
    <div className="metric-chart-shell">
      <div className="metric-chart-legend" aria-hidden="true">
        <span className="legend-dot continuity">Continuity</span>
        <span className="legend-dot surprise">Surprise</span>
        <span className="legend-dot persistence">Persistence</span>
      </div>
      <div className="metric-chart" aria-label="World-model metrics over time">
        {points.map((state, index) => (
          <div
            key={state.id}
            className={index === points.length - 1 ? "metric-column is-latest" : "metric-column"}
          >
            <div className="metric-bars">
              <span className="metric-lane" />
              <span className="metric-lane" />
              <span className="metric-lane" />
              <span className="metric-lane" />
              <span className="metric-lane" />
              <span
                className="metric-bar continuity"
                style={{ height: `${Math.max(8, Math.round(state.metrics.temporal_continuity_score * 100))}%` }}
                title={`Continuity ${state.metrics.temporal_continuity_score.toFixed(2)}`}
              />
              <span
                className="metric-bar surprise"
                style={{ height: `${Math.max(8, Math.round(state.metrics.surprise_score * 100))}%` }}
                title={`Surprise ${state.metrics.surprise_score.toFixed(2)}`}
              />
              <span
                className="metric-bar persistence"
                style={{ height: `${Math.max(8, Math.round(state.metrics.persistence_confidence * 100))}%` }}
                title={`Persistence ${state.metrics.persistence_confidence.toFixed(2)}`}
              />
            </div>
            <small>t{index + 1}</small>
          </div>
        ))}
      </div>
    </div>
  );
}
