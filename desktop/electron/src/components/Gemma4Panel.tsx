import type { CSSProperties } from "react";

export type Gemma4Alert = { text: string; is_alert: boolean; latency_ms: number };
export type Gemma4QueryReformulation = {
  setu_query: string; depth_stratum: string|null; person_filter: string|null;
  time_hint: string|null; confidence_min: number; gemma4_latency_ms?: number;
};

export function Gemma4StatusBadge({ available, model_label, mean_latency_ms, call_count, style }:
  { available: boolean; model_label: string; mean_latency_ms: number; call_count: number; style?: CSSProperties }) {
  const ms = mean_latency_ms > 0 ? `${Math.round(mean_latency_ms)}ms` : "--";
  return (
    <div className="health-badge" data-healthy={available} style={style}>
      <span>{model_label || "gemma-4"}</span>
      <strong>{available ? `local · ${ms}` : "offline"}</strong>
      {available && call_count > 0 && <span style={{color:"var(--muted)",fontSize:"0.74rem",display:"block"}}>{call_count} calls</span>}
    </div>
  );
}

export function Gemma4AlertBanner({ alert, onDismiss }: { alert: Gemma4Alert|null; onDismiss?: ()=>void }) {
  if (!alert?.is_alert || !alert.text || alert.text === "stable") return null;
  return (
    <div role="alert" aria-live="assertive" className="camera-health panel--change"
         style={{display:"flex",alignItems:"center",justifyContent:"space-between",gap:"0.75rem",padding:"0.65rem 0.9rem",borderRadius:"14px"}}>
      <div style={{display:"flex",alignItems:"center",gap:"0.55rem"}}>
        <span style={{width:8,height:8,borderRadius:"50%",background:"var(--kpi-watch)",flexShrink:0}} />
        <span style={{fontSize:"0.85rem"}}>{alert.text}</span>
      </div>
      <div style={{display:"flex",alignItems:"center",gap:"0.5rem",flexShrink:0}}>
        <span style={{fontSize:"0.74rem",color:"var(--muted)",borderRadius:"999px",border:"1px solid var(--line)",padding:"0.15rem 0.42rem"}}>
          gemma-4 · {Math.round(alert.latency_ms)}ms
        </span>
        {onDismiss && <button type="button" onClick={onDismiss} style={{background:"transparent",border:"none",color:"var(--muted)",cursor:"pointer"}} aria-label="Dismiss">x</button>}
      </div>
    </div>
  );
}

export function Gemma4QueryPanel({ reformulation, originalQuery }: { reformulation: Gemma4QueryReformulation|null; originalQuery: string }) {
  if (!reformulation || !originalQuery.trim()) return null;
  const chips = [
    reformulation.depth_stratum && `depth: ${reformulation.depth_stratum}`,
    reformulation.person_filter && `person: ${reformulation.person_filter}`,
    reformulation.time_hint && `time: ${reformulation.time_hint}`,
    reformulation.confidence_min > 0.3 && `min: ${(reformulation.confidence_min*100).toFixed(0)}%`,
  ].filter(Boolean) as string[];
  if (reformulation.setu_query === originalQuery && chips.length === 0) return null;
  return (
    <div style={{padding:"0.65rem 0.85rem",borderRadius:"14px",border:"1px solid rgba(67,216,201,0.2)",background:"rgba(67,216,201,0.06)",display:"flex",flexDirection:"column",gap:"0.45rem"}}>
      <span style={{fontSize:"0.72rem",textTransform:"uppercase",letterSpacing:"0.12em",color:"var(--accent-2)"}}>
        gemma-4 parsed{reformulation.gemma4_latency_ms != null && ` · ${Math.round(reformulation.gemma4_latency_ms)}ms`}
      </span>
      {reformulation.setu_query !== originalQuery && <div style={{fontSize:"0.84rem"}}><span style={{color:"var(--muted)"}}>Search: </span><strong>{reformulation.setu_query}</strong></div>}
      {chips.length > 0 && <div style={{display:"flex",flexWrap:"wrap",gap:"0.4rem"}}>{chips.map(c=><span key={c} style={{fontSize:"0.76rem",borderRadius:"999px",border:"1px solid rgba(67,216,201,0.24)",background:"rgba(67,216,201,0.1)",color:"var(--accent-2)",padding:"0.18rem 0.52rem"}}>{c}</span>)}</div>}
    </div>
  );
}

export function Gemma4InsightCard({ narrationText, anchorName, confidence, hallucinationRisk, latencyMs }:
  { narrationText: string; anchorName: string; confidence: number; hallucinationRisk: number; latencyMs: number }) {
  if (!narrationText) return null;
  const rc = hallucinationRisk < 0.25 ? "var(--kpi-healthy)" : hallucinationRisk < 0.5 ? "var(--kpi-watch)" : "var(--kpi-danger)";
  return (
    <div className="smriti-detail-card" style={{borderColor:"rgba(67,216,201,0.28)"}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",gap:"0.5rem",marginBottom:"0.45rem"}}>
        <span style={{fontSize:"0.72rem",textTransform:"uppercase",letterSpacing:"0.1em",color:"var(--accent-2)"}}>gemma-4 · {Math.round(latencyMs)}ms</span>
        <span style={{fontSize:"0.72rem",color:rc,borderRadius:"999px",border:`1px solid ${rc}`,padding:"0.12rem 0.4rem"}}>risk {(hallucinationRisk*100).toFixed(0)}%</span>
      </div>
      <strong style={{fontSize:"0.9rem",lineHeight:1.4}}>{narrationText}</strong>
      <div style={{display:"flex",gap:"0.5rem",marginTop:"0.45rem",flexWrap:"wrap"}}>
        <span className="smriti-chip">{anchorName.replace(/_/g," ")}</span>
        <span className="smriti-chip">conf {(confidence*100).toFixed(0)}%</span>
        <span className="smriti-chip">on-device</span>
      </div>
    </div>
  );
}
