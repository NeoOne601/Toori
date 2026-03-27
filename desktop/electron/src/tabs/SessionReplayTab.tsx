import { useDesktopApp } from "../state/DesktopAppContext";
import ObservationThumbnail from "../widgets/ObservationThumbnail";

export default function SessionReplayTab() {
  const app = useDesktopApp();

  return (
    <section className="panel panel-scroll">
      <div className="panel-head">
        <h3>Observation Timeline</h3>
        <span>{app.world.observations.length} frames captured</span>
      </div>
      <div className="timeline">
        {app.world.observations.map((observation) => (
          <article key={observation.id} className="timeline-card">
            <ObservationThumbnail src={app.assetUrl(observation.thumbnail_path)} alt={observation.id} />
            <div>
              <p className="muted">{new Date(observation.created_at).toLocaleString()}</p>
              <h4>{observation.summary || observation.id}</h4>
              <p>{observation.source_query || "Auto-captured lens observation"}</p>
              <div className="chips">
                <span>{observation.providers.join(" + ")}</span>
                <span>novelty {observation.novelty.toFixed(2)}</span>
              </div>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
