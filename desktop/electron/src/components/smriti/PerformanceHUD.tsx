import type { SmritiMetrics, SmritiStatus } from "../../types";

type PerformanceHUDProps = {
  metrics: SmritiMetrics | null;
  status: SmritiStatus | null;
  onRefresh: () => Promise<void>;
};

export default function PerformanceHUD({ metrics, status, onRefresh }: PerformanceHUDProps) {
  const utilization = Math.round((status?.ingestion.queue_utilization || 0) * 100);
  const workerCount = metrics?.workers.length || 0;

  return (
    <div className="smriti-stack">
      <section className="panel panel--comparison">
        <div className="smriti-panel-header">
          <div>
            <p className="eyebrow">Performance HUD</p>
            <h3>Pipeline health</h3>
          </div>
          <button type="button" onClick={() => void onRefresh()}>
            Refresh
          </button>
        </div>
        <div className="smriti-kpi-grid">
          <div className="smriti-kpi-card">
            <span>Workers</span>
            <strong>{workerCount}</strong>
          </div>
          <div className="smriti-kpi-card">
            <span>Queue depth</span>
            <strong>{status?.ingestion.queue_depth || 0}</strong>
          </div>
          <div className="smriti-kpi-card">
            <span>Queue utilization</span>
            <strong>{utilization}%</strong>
          </div>
          <div className="smriti-kpi-card">
            <span>Pending media</span>
            <strong>{metrics?.pending_media || 0}</strong>
          </div>
        </div>
      </section>

      <section className="smriti-hud-grid">
        <article className="panel smriti-hud-card">
          <p className="eyebrow">Workers</p>
          {(metrics?.workers || []).map((worker) => (
            <div key={worker.worker_id} className="smriti-worker-row">
              <div>
                <strong>Worker {worker.worker_id}</strong>
                <p className="muted">pid {worker.pid} · {worker.alive ? "alive" : "stopped"}</p>
              </div>
              <div className="smriti-worker-metrics">
                <span>q {worker.queue_size}</span>
                <span>p {worker.pending}</span>
                <span>c {worker.completed}</span>
              </div>
            </div>
          ))}
          {workerCount === 0 ? <p className="muted">Worker pool not active.</p> : null}
        </article>

        <article className="panel smriti-hud-card">
          <p className="eyebrow">Energy EMA</p>
          {Object.entries(metrics?.energy_ema || {}).map(([sessionId, value]) => (
            <div key={sessionId} className="smriti-energy-row">
              <span>{sessionId}</span>
              <strong>{value.toFixed(3)}</strong>
            </div>
          ))}
          {Object.keys(metrics?.energy_ema || {}).length === 0 ? <p className="muted">No live sessions reported.</p> : null}
        </article>

        <article className="panel smriti-hud-card">
          <p className="eyebrow">Ingestion</p>
          <p className="muted">Queued: {status?.ingestion.queued || 0}</p>
          <p className="muted">Processed: {status?.ingestion.processed || 0}</p>
          <p className="muted">Failed: {status?.ingestion.failed || 0}</p>
          <p className="muted">Duplicates: {status?.ingestion.skipped_duplicate || 0}</p>
        </article>

        <article className="panel smriti-hud-card">
          <p className="eyebrow">Recent sessions</p>
          <div className="smriti-chip-row">
            {(metrics?.recent_sessions || []).map((sessionId) => (
              <span key={sessionId} className="smriti-chip">{sessionId}</span>
            ))}
          </div>
          {(metrics?.recent_sessions || []).length === 0 ? <p className="muted">No recent Smriti sessions.</p> : null}
        </article>
      </section>
    </div>
  );
}
