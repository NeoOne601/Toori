import { useDesktopApp } from "../state/DesktopAppContext";
import MemoryRelinking from "../panels/MemoryRelinking";
import ReasoningTraceList from "../widgets/ReasoningTraceList";

export default function MemorySearchTab() {
  const app = useDesktopApp();

  return (
    <section className="panel-grid memory-grid">
      <article className="panel">
        <div className="panel-head">
          <h3>Semantic Search</h3>
          <span>Search across stored observations</span>
        </div>
        <label className="field">
          <span>Search prompt</span>
          <input
            placeholder="desk mug, hallway motion, screen state..."
            value={app.searchText}
            onChange={(event) => app.setSearchText(event.target.value)}
          />
        </label>
        {app.world.queryResult?.answer ? (
          <div className="answer-box">
            <p className="eyebrow">Answer</p>
            <p>{app.world.queryResult.answer.text}</p>
          </div>
        ) : null}
        <ReasoningTraceList trace={app.world.queryResult?.reasoning_trace || []} />
      </article>
      <MemoryRelinking
        title="Results"
        hits={app.world.queryResult?.hits || []}
        assetUrl={app.assetUrl}
        emptyLabel="Search results will appear here."
        limit={24}
        toneClassName="wide panel--memory"
      />
    </section>
  );
}
