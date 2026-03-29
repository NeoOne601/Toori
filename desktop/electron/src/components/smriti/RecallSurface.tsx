import type { SmritiRecallResult } from "../../types";

type RecallSurfaceProps = {
  query: string;
  onQueryChange: (value: string) => void;
  personFilter: string;
  onPersonFilterChange: (value: string) => void;
  locationFilter: string;
  onLocationFilterChange: (value: string) => void;
  minConfidence: number;
  onMinConfidenceChange: (value: number) => void;
  timeRangeDays: number;
  onTimeRangeDaysChange: (value: number) => void;
  results: SmritiRecallResult[];
  busy: boolean;
  totalSearched: number;
  onOpenMedia: (media: SmritiRecallResult) => void;
  assetUrl: (filePath: string) => string;
};

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function scoreTone(score: number) {
  if (score >= 0.8) {
    return "safe";
  }
  if (score >= 0.5) {
    return "watch";
  }
  return "risk";
}

export default function RecallSurface({
  query,
  onQueryChange,
  personFilter,
  onPersonFilterChange,
  locationFilter,
  onLocationFilterChange,
  minConfidence,
  onMinConfidenceChange,
  timeRangeDays,
  onTimeRangeDaysChange,
  results,
  busy,
  totalSearched,
  onOpenMedia,
  assetUrl,
}: RecallSurfaceProps) {
  const orderedResults = [...results].sort(
    (left, right) => new Date(left.created_at).getTime() - new Date(right.created_at).getTime(),
  );
  const earliest = orderedResults[0] ? new Date(orderedResults[0].created_at).getTime() : Date.now();
  const latest = orderedResults[orderedResults.length - 1]
    ? new Date(orderedResults[orderedResults.length - 1].created_at).getTime()
    : earliest + 1;
  const range = Math.max(latest - earliest, 1);

  return (
    <div className="smriti-stack">
      <section className="panel panel--memory">
        <div className="smriti-panel-header">
          <div>
            <p className="eyebrow">Recall Surface</p>
            <h3>Grounded memory query</h3>
          </div>
          <div className="smriti-inline-metrics">
            <span>{busy ? "Searching…" : `${results.length} results`}</span>
            <span>{totalSearched} searched</span>
          </div>
        </div>
        <div className="smriti-filter-grid">
          <label className="smriti-field">
            <span>Query</span>
            <input
              className="smriti-recall-input"
              value={query}
              onChange={(event) => onQueryChange(event.target.value)}
              placeholder="blue bottle behind person"
            />
          </label>
          <label className="smriti-field">
            <span>Person</span>
            <input value={personFilter} onChange={(event) => onPersonFilterChange(event.target.value)} placeholder="Priya" />
          </label>
          <label className="smriti-field">
            <span>Location</span>
            <input value={locationFilter} onChange={(event) => onLocationFilterChange(event.target.value)} placeholder="Office" />
          </label>
          <label className="smriti-field">
            <span>Min confidence</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={minConfidence}
              onChange={(event) => onMinConfidenceChange(Number(event.target.value))}
            />
            <small>{minConfidence.toFixed(2)}</small>
          </label>
          <label className="smriti-field">
            <span>Time window</span>
            <select value={timeRangeDays} onChange={(event) => onTimeRangeDaysChange(Number(event.target.value))}>
              <option value={0}>All time</option>
              <option value={7}>Last 7 days</option>
              <option value={30}>Last 30 days</option>
              <option value={90}>Last 90 days</option>
            </select>
          </label>
        </div>
      </section>

      <section className="panel panel--stable">
        <div className="smriti-panel-header">
          <div>
            <p className="eyebrow">Temporal Constellation</p>
            <h3>Result spread</h3>
          </div>
          <p className="muted">Setu-2 ranks by hybrid relevance while the constellation preserves time ordering.</p>
        </div>
        <svg
          className="smriti-constellation"
          viewBox="0 0 960 260"
          role="img"
          aria-label="Temporal constellation of recall results"
        >
          <defs>
            <linearGradient id="smriti-constellation-line" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="rgba(255, 140, 66, 0.25)" />
              <stop offset="100%" stopColor="rgba(67, 216, 201, 0.35)" />
            </linearGradient>
          </defs>
          <line x1="60" y1="220" x2="900" y2="220" stroke="rgba(255,255,255,0.18)" strokeDasharray="6 6" />
          {orderedResults.map((result, index) => {
            const createdAt = new Date(result.created_at).getTime();
            const x = 60 + ((createdAt - earliest) / range) * 840;
            const y = 30 + (1 - clamp(result.hybrid_score, 0, 1)) * 170;
            const radius = 7 + clamp((1 - result.hallucination_risk) * 9, 1.5, 10);
            return (
              <g key={result.media_id}>
                <circle
                  cx={x}
                  cy={y}
                  r={radius}
                  fill={result.hallucination_risk < 0.25 ? "rgba(67, 216, 201, 0.82)" : "rgba(255, 140, 66, 0.78)"}
                  stroke="rgba(255, 255, 255, 0.75)"
                  strokeWidth="1.5"
                />
                {index > 0 ? (
                  <line
                    x1={60 + ((new Date(orderedResults[index - 1].created_at).getTime() - earliest) / range) * 840}
                    y1={30 + (1 - clamp(orderedResults[index - 1].hybrid_score, 0, 1)) * 170}
                    x2={x}
                    y2={y}
                    stroke="rgba(255,255,255,0.12)"
                    strokeWidth="1"
                  />
                ) : null}
              </g>
            );
          })}
        </svg>
      </section>

      <section className="smriti-results-grid">
        {results.map((result) => {
          const tone = scoreTone(1 - result.hallucination_risk);
          return (
            <button
              key={result.media_id}
              type="button"
              className={`panel smriti-recall-card smriti-recall-card--${tone}`}
              onClick={() => onOpenMedia(result)}
            >
              <div className="smriti-recall-card__media">
                <img
                  src={assetUrl(result.thumbnail_path || result.file_path)}
                  alt=""
                  loading="lazy"
                />
              </div>
              <div className="smriti-recall-card__body">
                <div className="smriti-recall-card__header">
                  <p className="eyebrow">{new Date(result.created_at).toLocaleString()}</p>
                  <span className={`smriti-pill smriti-pill--${tone}`}>
                    risk {(result.hallucination_risk * 100).toFixed(0)}%
                  </span>
                </div>
                <h4>{result.primary_description}</h4>
                <p className="muted">
                  {result.anchor_basis} in {result.depth_stratum}
                </p>
                <div className="smriti-score-row">
                  <span>Hybrid {(result.hybrid_score * 100).toFixed(0)}</span>
                  <span>Setu {(result.setu_score * 100).toFixed(0)}</span>
                </div>
                <div className="smriti-chip-row">
                  {result.person_names.slice(0, 3).map((person) => (
                    <span key={`${result.media_id}-${person}`} className="smriti-chip">
                      {person}
                    </span>
                  ))}
                  {result.location_name ? <span className="smriti-chip">{result.location_name}</span> : null}
                </div>
              </div>
            </button>
          );
        })}
        {!busy && results.length === 0 ? (
          <div className="panel smriti-empty-state">
            <p className="eyebrow">No Results Yet</p>
            <h4>Start with a natural-language query</h4>
            <p className="muted">The recall surface fills as soon as the Smriti index contains media and the query text is specific enough to rank it.</p>
          </div>
        ) : null}
      </section>
    </div>
  );
}
