import { formatEntityDisplayLabel } from "../lib/formatting";
import type { EntityTrack } from "../types";

type EntityTracksPanelProps = {
  tracks: EntityTrack[];
  showAllTracks: boolean;
  onToggleShowAll: () => void;
};

function statusClass(status: string) {
  return status.replace(/\s+/g, "-").toLowerCase();
}

export default function EntityTracksPanel({
  tracks,
  showAllTracks,
  onToggleShowAll,
}: EntityTracksPanelProps) {
  return (
    <article className="panel panel--persistence panel-scroll" data-panel="entity-tracks">
      <div className="panel-head">
        <div>
          <h3>Entity Tracks</h3>
          <p className="muted">Persistent threads across visibility, occlusion, recovery, and re-identification.</p>
        </div>
        <span>{tracks.length} tracks</span>
      </div>
      <div className="stack scroll-stack">
        {tracks.map((track) => (
          <div key={track.id} className="track-card" data-status={track.status}>
            <div className="track-top">
              <strong className="track-label">{formatEntityDisplayLabel(track)}</strong>
              <span className={`track-badge badge-${statusClass(track.status)}`}>{track.status}</span>
            </div>
            <div className="track-sparkline" aria-hidden="true">
              {(track.status_history ?? []).slice(-20).map((status, index) => (
                <span
                  key={`${track.id}-${status}-${index}`}
                  className={`spark-pip spark-${statusClass(status)}`}
                  title={status}
                />
              ))}
            </div>
            <div className="chips chips--persist">
              <span>continuity {track.continuity_score.toFixed(2)}</span>
              <span>persistence {track.persistence_confidence.toFixed(2)}</span>
              <span>re-id {track.reidentification_count}</span>
            </div>
          </div>
        ))}
      </div>
      {tracks.length > 8 ? (
        <button type="button" className="secondary-link" onClick={onToggleShowAll}>
          {showAllTracks ? "Show fewer tracks" : `Show all ${tracks.length} tracks`}
        </button>
      ) : null}
    </article>
  );
}
