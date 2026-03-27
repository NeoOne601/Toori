import { useDesktopApp } from "../state/DesktopAppContext";
import ReasoningTraceList from "../widgets/ReasoningTraceList";

export default function IntegrationsTab() {
  const app = useDesktopApp();

  return (
    <section className="panel-grid">
      <article className="panel">
        <div className="panel-head">
          <h3>Runtime Endpoints</h3>
          <span>Loopback by default</span>
        </div>
        <pre className="code-block">{`POST http://127.0.0.1:7777/v1/analyze
POST http://127.0.0.1:7777/v1/living-lens/tick
POST http://127.0.0.1:7777/v1/query
POST http://127.0.0.1:7777/v1/challenges/evaluate
GET  http://127.0.0.1:7777/v1/world-state
GET  http://127.0.0.1:7777/v1/settings
PUT  http://127.0.0.1:7777/v1/settings
GET  http://127.0.0.1:7777/v1/providers/health
WS   ws://127.0.0.1:7777/v1/events`}</pre>
      </article>

      <article className="panel">
        <div className="panel-head">
          <h3>Plugin Pattern</h3>
          <span>Any host app</span>
        </div>
        <pre className="code-block">{`const hit = await toori.query({
  query: "Where did I last see the blue notebook?",
  session_id: "desktop-live",
  top_k: 5
});`}</pre>
        <p className="muted">
          Generated SDKs live under `sdk/` and target TypeScript, Python, Swift, and Kotlin.
        </p>
      </article>

      <article className="panel">
        <div className="panel-head">
          <h3>Provider Health</h3>
          <span>Fallback order visibility</span>
        </div>
        <div className="stack">
          {app.world.health.map((item) => (
            <div key={item.name} className="list-row compact">
              <div>
                <strong>{item.name}</strong>
                <p>{item.message}</p>
              </div>
              <span>{item.healthy ? "ready" : "fallback"}</span>
            </div>
          ))}
        </div>
        <ReasoningTraceList
          trace={app.world.analysis?.reasoning_trace || app.world.queryResult?.reasoning_trace || []}
        />
      </article>
    </section>
  );
}
