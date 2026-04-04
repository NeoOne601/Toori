import { TABS } from "../constants";
import { useDesktopApp } from "../state/DesktopAppContext";
import { Gemma4StatusBadge } from "../components/Gemma4Panel";

export default function NavSidebar() {
  const app = useDesktopApp();

  return (
    <aside className="nav">
      <div className="nav-header">
        <p className="eyebrow">Toori</p>
        <h1>Lens Assistant</h1>
        <p className="muted nav-copy">
          Camera-first memory, live reasoning, and plugin-ready runtime across desktop, iOS, and Android.
        </p>
      </div>
      <div className="nav-scroll">
        <div className="status-card">
          <div className="status-line">
            <span className="status-dot" data-online={app.world.eventsState === "connected"} />
            <span>Event stream: {app.world.eventsState}</span>
          </div>
          <div className="muted">{app.world.status}</div>
          <div className="status-grid">
            <div className="status-metric">
              <span>Session</span>
              <strong>{app.sessionId}</strong>
            </div>
            <div className="status-metric">
              <span>Providers</span>
              <strong>{app.readyProviders} ready / {app.world.health.length || 0}</strong>
            </div>
            <div className="status-metric">
              <span>Fallbacks</span>
              <strong>{app.degradedProviders}</strong>
            </div>
          </div>
          <button className="secondary-link" onClick={() => app.setActiveTab("Integrations")}>
            View provider health
          </button>
        </div>
        <nav className="tabs">
          {TABS.map((tab) => (
            <button
              key={tab}
              className={tab === app.activeTab ? "tab active" : "tab"}
              onClick={() => app.setActiveTab(tab)}
            >
              {tab}
            </button>
          ))}
        </nav>
        <div className="sidebar-health">
          {(() => {
            const mlxHealth = app.world.health?.find((h: any) => h.name === "mlx");
            const isAvailable = mlxHealth?.healthy ?? false;
            return (
              <Gemma4StatusBadge 
                available={isAvailable}
                model_label="Gemma 4 e4b"
                mean_latency_ms={mlxHealth?.latency_ms ?? 0}
                call_count={0} 
              />
            );
          })()}
          {app.topHealth.map((item) => (
            <div key={item.name} className="health-badge" data-healthy={item.healthy}>
              <span>{item.name}</span>
              <strong>{item.healthy ? "ready" : "degraded"}</strong>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
