import type { CSSProperties, HTMLAttributes } from "react";

export type OcclusionTrack = {
  id: string;
  label: string;
  status: "visible" | "occluded" | "recovered" | "disappeared" | "violated";
  confidence?: number | null;
  note?: string;
};

type OcclusionPanelProps = {
  title?: string;
  summary?: string;
  score?: number | null;
  tracks?: OcclusionTrack[];
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

export default function OcclusionPanel({
  title = "Occlusion",
  summary,
  score,
  tracks = [],
  className,
  style,
  ...rest
}: OcclusionPanelProps) {
  if (!tracks.length && score == null && !summary) {
    return null;
  }

  return (
    <section
      className={mergeClassNames("panel occlusion-panel", className)}
      style={style}
      {...rest}
    >
      <div className="panel-head">
        <div>
          <p className="eyebrow">{title}</p>
          <h3>Ghost recovery and occlusion state</h3>
        </div>
        <div className="occlusion-panel__score">
          <span>Recovery</span>
          <strong>{formatPct(score)}</strong>
        </div>
      </div>

      {summary ? <p className="muted">{summary}</p> : null}

      <div className="occlusion-panel__tracks">
        {tracks.map((track) => (
          <article
            key={track.id}
            className="occlusion-panel__track"
            data-status={track.status}
          >
            <div className="occlusion-panel__track-top">
              <strong>{track.label}</strong>
              <span className="occlusion-panel__ghost-badge">
                {track.status === "occluded" ? "Ghost bbox" : track.status}
              </span>
            </div>
            <div className="occlusion-panel__track-body">
              <span>{formatPct(track.confidence)}</span>
              {track.note ? <small>{track.note}</small> : null}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
