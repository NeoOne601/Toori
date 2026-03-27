import type { Observation, ReasoningTraceEntry } from "../types";
import ObservationThumbnail from "../widgets/ObservationThumbnail";
import ReasoningTraceList from "../widgets/ReasoningTraceList";

type LiveUnderstandingProps = {
  title?: string;
  observation?: Observation | null;
  summary?: string | null;
  assetUrl: (path: string) => string;
  trace?: ReasoningTraceEntry[];
  chips?: string[];
  emptyLabel?: string;
  toneClassName?: string;
};

export default function LiveUnderstanding({
  title = "Live Understanding",
  observation,
  summary,
  assetUrl,
  trace = [],
  chips = [],
  emptyLabel = "Monitoring...",
  toneClassName = "panel--stable",
}: LiveUnderstandingProps) {
  return (
    <article className={`panel ${toneClassName}`} data-panel="live-understanding">
      <div className="panel-head">
        <h3>{title}</h3>
        <span>{observation ? observation.id : "waiting"}</span>
      </div>
      {observation ? (
        <div className="observation-card-stack">
          <div className="observation-card">
            <ObservationThumbnail src={assetUrl(observation.thumbnail_path)} alt={observation.id} />
            <div>
              <p className="muted">{new Date(observation.created_at).toLocaleString()}</p>
              <p>{summary}</p>
              {chips.length ? (
                <div className="chips chips--stable">
                  {chips.map((chip) => (
                    <span key={chip}>{chip}</span>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
          <ReasoningTraceList trace={trace} />
        </div>
      ) : (
        <p className="muted">{emptyLabel}</p>
      )}
    </article>
  );
}
