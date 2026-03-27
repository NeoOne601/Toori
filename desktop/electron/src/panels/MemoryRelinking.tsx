import { formatRelativeTime } from "../lib/formatting";
import type { SearchHit } from "../types";
import ObservationThumbnail from "../widgets/ObservationThumbnail";

type MemoryRelinkingProps = {
  title?: string;
  hits: SearchHit[];
  assetUrl: (path: string) => string;
  emptyLabel?: string;
  limit?: number;
  toneClassName?: string;
};

export default function MemoryRelinking({
  title = "Memory Relinking",
  hits,
  assetUrl,
  emptyLabel = "Searching history...",
  limit = 6,
  toneClassName = "panel--memory",
}: MemoryRelinkingProps) {
  const visibleHits = hits.slice(0, limit);

  return (
    <article className={`panel ${toneClassName} panel-scroll`} data-panel="memory-relinking">
      <div className="panel-head">
        <h3>{title}</h3>
        <span>{hits.length} scenes</span>
      </div>
      <div className="stack scroll-stack">
        {visibleHits.length ? (
          visibleHits.map((hit) => (
            <div key={hit.observation_id} className="list-row" title={hit.observation_id}>
              <ObservationThumbnail src={assetUrl(hit.thumbnail_path)} alt={hit.observation_id} />
              <div>
                <strong>{hit.summary || "Remembered scene"}</strong>
                <p>{formatRelativeTime(hit.created_at)}</p>
              </div>
              <span>{(hit.score * 100).toFixed(0)}%</span>
            </div>
          ))
        ) : (
          <p className="muted">{emptyLabel}</p>
        )}
      </div>
    </article>
  );
}
