import type { AudioQueryResult } from "../../types";
import { BROWSER_RUNTIME_URL as API_BASE_URL } from "../../hooks/useRuntimeBridge";

interface Props {
  results: AudioQueryResult[];
  latencyMs: number;
  indexSize: number;
  onClear: () => void;
}

export function AudioResultsPanel({ results, latencyMs, indexSize, onClear }: Props) {
  if (results.length === 0) return null;

  return (
    <div style={{ marginTop: "1rem" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.6rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span style={{ fontSize: "0.8rem", fontWeight: 500 }}>Audio recall</span>
          <span className="smriti-chip">{Math.round(latencyMs)}ms</span>
          <span className="smriti-chip">{indexSize} indexed</span>
        </div>
        <button
          type="button"
          onClick={onClear}
          aria-label="Clear audio results"
          style={{ fontSize: "0.75rem", color: "var(--muted)", background: "transparent", border: "none", cursor: "pointer", padding: "2px 6px" }}
        >
          Clear
        </button>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        {results.map((r) => (
          <div key={r.media_id} className="smriti-detail-card" style={{ display: "flex", gap: "0.75rem", alignItems: "flex-start" }}>
            {r.thumbnail_path && (
              <img
                src={`${API_BASE_URL}/v1/file?path=${encodeURIComponent(r.thumbnail_path)}`}
                alt="Memory thumbnail"
                style={{ width: 52, height: 52, objectFit: "cover", borderRadius: 6, flexShrink: 0 }}
                onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
              />
            )}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginBottom: "0.3rem", flexWrap: "wrap" }}>
                <span className="smriti-chip">rank #{r.rank}</span>
                <span className="smriti-chip">{(r.audio_score * 100).toFixed(0)}% match</span>
                {r.audio_duration_seconds != null && (
                  <span className="smriti-chip">{r.audio_duration_seconds.toFixed(1)}s</span>
                )}
              </div>
              {r.gemma4_narration && (
                <div style={{ fontSize: "0.82rem", fontStyle: "italic", color: "var(--fg)", marginBottom: "0.25rem", lineHeight: 1.4 }}>
                  {r.gemma4_narration}
                  <span className="smriti-chip" style={{ marginLeft: "0.4rem", fontStyle: "normal" }}>gemma-4</span>
                </div>
              )}
              <div style={{ height: 4, background: "var(--bg2)", borderRadius: 2, overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${Math.max(2, r.audio_score * 100)}%`, background: "var(--accent-2)", borderRadius: 2 }} />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
