import { useEffect, useState } from "react";
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
  onShareObservation?: (observation: Observation) => Promise<string>;
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
  onShareObservation,
}: LiveUnderstandingProps) {
  const [shareState, setShareState] = useState<"idle" | "copying" | "copied" | "error">("idle");
  const [shareMessage, setShareMessage] = useState("Copy a ready-to-send recap.");

  useEffect(() => {
    setShareState("idle");
    setShareMessage("Copy a ready-to-send recap.");
  }, [observation?.id]);

  async function handleShare() {
    if (!observation || !onShareObservation) {
      return;
    }
    setShareState("copying");
    setShareMessage("Preparing share text...");
    try {
      const message = await onShareObservation(observation);
      setShareState("copied");
      setShareMessage(message || "Copied share text.");
    } catch (error) {
      setShareState("error");
      setShareMessage((error as Error).message);
    }
  }

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
            <div className="observation-card__body">
              <p className="muted">{new Date(observation.created_at).toLocaleString()}</p>
              <p>{summary}</p>
              {chips.length ? (
                <div className="chips chips--stable">
                  {chips.map((chip) => (
                    <span key={chip}>{chip}</span>
                  ))}
                </div>
              ) : null}
              {onShareObservation ? (
                <div className="observation-share">
                  <button type="button" onClick={handleShare} disabled={shareState === "copying"}>
                    {shareState === "copying" ? "Copying..." : shareState === "copied" ? "Copied" : "Copy Share Text"}
                  </button>
                  <span className="observation-share__status" data-state={shareState}>
                    {shareMessage}
                  </span>
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
