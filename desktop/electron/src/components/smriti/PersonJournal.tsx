import { useEffect, useMemo, useRef } from "react";
import type { SmritiPersonJournal as JournalType } from "../../types";

type PersonJournalProps = {
  personName: string;
  onPersonNameChange: (value: string) => void;
  journal: JournalType | null;
  onLoadJournal: (name: string) => Promise<void>;
};

type AtlasNode = {
  entity_id?: string;
  label?: string;
  track_length?: number;
};

type AtlasEdge = {
  source_id?: string;
  target_id?: string;
  co_occurrence_count?: number;
  spatial_proximity?: number;
};

export default function PersonJournal({
  personName,
  onPersonNameChange,
  journal,
  onLoadJournal,
}: PersonJournalProps) {
  const hasAtlas = Boolean(journal?.atlas?.nodes?.length);

  return (
    <div className="smriti-stack">
      <section className="panel panel--stable">
        <div className="smriti-panel-header">
          <div>
            <p className="eyebrow">Person Journal</p>
            <h3>Entity timeline</h3>
          </div>
          <p className="muted">Inspect every confirmed media link for a named person, newest first.</p>
        </div>
        <form
          className="smriti-search-form"
          onSubmit={async (event) => {
            event.preventDefault();
            await onLoadJournal(personName);
          }}
        >
          <input
            value={personName}
            onChange={(event) => onPersonNameChange(event.target.value)}
            placeholder="Enter a person name"
          />
          <button type="submit" className="primary" disabled={!personName.trim()}>
            Load journal
          </button>
        </form>
      </section>

      {hasAtlas ? <CoOccurrenceGraph personName={journal?.person_name || personName} journal={journal} /> : null}

      <section className="smriti-journal-grid">
        {journal?.entries.length ? journal.entries.map((entry) => (
          <article key={entry.media_id} className="panel smriti-journal-entry">
            <p className="eyebrow">{new Date(entry.ingested_at).toLocaleString()}</p>
            <h4>{entry.media_id}</h4>
            <p className="muted">{entry.file_path}</p>
          </article>
        )) : (
          <div className="panel smriti-empty-state">
            <p className="eyebrow">No Journal Loaded</p>
            <h4>Search for a tagged person</h4>
            <p className="muted">After you confirm a person in Deepdive, Smriti propagates the label to similar media and exposes the resulting journal here.</p>
          </div>
        )}
      </section>
    </div>
  );
}

function CoOccurrenceGraph({
  personName,
  journal,
}: {
  personName: string;
  journal: JournalType | null;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const atlasNodes = useMemo(
    () => ((journal?.atlas?.nodes as AtlasNode[] | undefined) ?? []).filter((node) => node.label),
    [journal?.atlas?.nodes],
  );
  const atlasEdges = useMemo(
    () => ((journal?.atlas?.edges as AtlasEdge[] | undefined) ?? []),
    [journal?.atlas?.edges],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap || atlasNodes.length === 0) {
      return;
    }
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    const central =
      atlasNodes.find((node) => node.label?.toLowerCase() === personName.trim().toLowerCase()) ?? atlasNodes[0];
    const others = atlasNodes.filter((node) => node !== central);

    const draw = () => {
      const rect = wrap.getBoundingClientRect();
      const width = Math.max(rect.width, 320);
      const height = 320;
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
      }

      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, width, height);
      context.save();

      const centerX = width / 2;
      const centerY = height / 2;
      const radius = Math.max(Math.min(width, height) * 0.32, 70);
      const positions = new Map<string, { x: number; y: number; size: number; label: string; central: boolean }>();

      positions.set(String(central.entity_id || central.label), {
        x: centerX,
        y: centerY,
        size: 18 + Math.min(Number(central.track_length || 1), 16),
        label: central.label || personName,
        central: true,
      });

      others.forEach((node, index) => {
        const angle = (Math.PI * 2 * index) / Math.max(others.length, 1) - Math.PI / 2;
        positions.set(String(node.entity_id || node.label), {
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius,
          size: 11 + Math.min(Number(node.track_length || 1), 12),
          label: node.label || "Unknown",
          central: false,
        });
      });

      context.strokeStyle = "rgba(67,216,201,0.42)";
      atlasEdges.forEach((edge) => {
        const source = positions.get(String(edge.source_id));
        const target = positions.get(String(edge.target_id));
        if (!source || !target) {
          return;
        }
        context.beginPath();
        context.lineWidth = 1.5 + Number(edge.co_occurrence_count || edge.spatial_proximity || 0);
        context.moveTo(source.x, source.y);
        context.lineTo(target.x, target.y);
        context.stroke();
      });

      positions.forEach((node) => {
        context.beginPath();
        context.fillStyle = node.central ? "rgba(255,140,66,0.9)" : "rgba(67,216,201,0.88)";
        context.arc(node.x, node.y, node.size, 0, Math.PI * 2);
        context.fill();
        context.strokeStyle = "rgba(255,255,255,0.2)";
        context.lineWidth = 1.2;
        context.stroke();
        context.fillStyle = "rgba(236,243,251,0.96)";
        context.font = "12px system-ui, sans-serif";
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillText(node.label, node.x, node.y + node.size + 16);
      });

      context.restore();
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(wrap);
    return () => observer.disconnect();
  }, [atlasEdges, atlasNodes, personName]);

  if (atlasNodes.length === 0) {
    return null;
  }

  return (
    <section className="panel panel--memory">
      <div className="smriti-panel-header">
        <div>
          <p className="eyebrow">Co-occurrence Graph</p>
          <h3>Who appears with {personName || "this person"}</h3>
        </div>
        <p className="muted">Static circular layout from the Epistemic Atlas.</p>
      </div>
      <div className="co-occurrence-graph" ref={wrapRef}>
        <canvas ref={canvasRef} aria-hidden="true" />
      </div>
      <div className="co-occurrence-graph__legend">
        <span className="smriti-chip">Orange: primary person</span>
        <span className="smriti-chip">Teal: co-occurring people</span>
      </div>
    </section>
  );
}
