import { formatLatency } from "../lib/formatting";
import type { ReasoningTraceEntry } from "../types";

type ReasoningTraceListProps = {
  trace: ReasoningTraceEntry[];
};

export default function ReasoningTraceList({ trace }: ReasoningTraceListProps) {
  if (!trace.length) {
    return null;
  }

  return (
    <div className="trace-list">
      {trace.map((entry) => (
        <div
          key={`${entry.provider}-${entry.attempted}-${entry.success}-${entry.error || "ok"}`}
          className="trace-item"
        >
          <div className="trace-head">
            <strong>{entry.provider}</strong>
            <span data-success={entry.success}>
              {entry.success ? "answered" : entry.attempted ? "failed" : "skipped"}
            </span>
          </div>
          <p>
            {entry.success
              ? `${entry.health_message} in ${formatLatency(entry.latency_ms)}`
              : entry.error || entry.health_message}
          </p>
        </div>
      ))}
    </div>
  );
}
