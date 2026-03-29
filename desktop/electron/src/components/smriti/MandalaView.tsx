import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { CSSProperties, MouseEvent as ReactMouseEvent } from "react";
import type { SmritiClusterNode, SmritiMandalaData } from "../../types";

type MandalaViewProps = {
  data: SmritiMandalaData | null;
  selectedClusterId: number | null;
  onNodeSelect: (clusterId: number) => void;
  onNodeExpand: (clusterId: number) => void;
  loading?: boolean;
  className?: string;
  style?: CSSProperties;
};

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function energyColor(energy: number, alpha = 1) {
  const red = Math.round(200 + energy * 55);
  const green = Math.round(200 - energy * 150);
  const blue = Math.round(255 - energy * 200);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function stratumColor(stratum: string | null) {
  switch (stratum) {
    case "foreground":
      return "rgba(255,140,66,0.85)";
    case "background":
      return "rgba(67,216,201,0.85)";
    case "midground":
      return "rgba(130,171,255,0.85)";
    default:
      return "rgba(200,200,220,0.85)";
  }
}

function mediaEnergy(node: SmritiClusterNode, maxMedia: number) {
  if (maxMedia <= 0) {
    return 0.35;
  }
  return clamp(node.media_count / maxMedia, 0.15, 1);
}

export default function MandalaView({
  data,
  selectedClusterId,
  onNodeSelect,
  onNodeExpand,
  loading = false,
  className,
  style,
}: MandalaViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const positionsRef = useRef<Map<number, { x: number; y: number }>>(new Map());
  const rafRef = useRef<number>(0);
  const dragRef = useRef<{ active: boolean; startX: number; startY: number; originX: number; originY: number }>({
    active: false,
    startX: 0,
    startY: 0,
    originX: 0,
    originY: 0,
  });
  const [hoveredClusterId, setHoveredClusterId] = useState<number | null>(null);
  const [zoom, setZoom] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [expandProgress, setExpandProgress] = useState(0);
  const [focusedIdx, setFocusedIdx] = useState(0);

  const nodes = data?.nodes ?? [];
  const maxMedia = useMemo(
    () => Math.max(...nodes.map((node) => node.media_count), 1),
    [nodes],
  );
  const nodeMap = useMemo(() => new Map(nodes.map((node) => [node.id, node])), [nodes]);
  const hoveredNode = hoveredClusterId != null ? nodeMap.get(hoveredClusterId) ?? null : null;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || nodes.length === 0) {
      return;
    }

    if (!workerRef.current) {
      workerRef.current = new Worker(new URL("./mandala-force-worker.ts", import.meta.url), {
        type: "module",
      });
      workerRef.current.onmessage = (event: MessageEvent<{ type: string; positions: Array<[number, number, number]> }>) => {
        if (event.data.type !== "positions") {
          return;
        }
        const next = new Map<number, { x: number; y: number }>();
        for (const [id, x, y] of event.data.positions) {
          next.set(id, { x, y });
        }
        positionsRef.current = next;
      };
    }

    const rect = canvas.getBoundingClientRect();
    workerRef.current.postMessage({
      type: "init",
      clusters: nodes.map((node) => ({ id: node.id, mediaCount: node.media_count })),
      width: rect.width || canvas.clientWidth || 800,
      height: rect.height || canvas.clientHeight || 600,
    });

    return () => {
      workerRef.current?.postMessage({ type: "tick_pause" });
    };
  }, [nodes]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    const draw = () => {
      const rect = container.getBoundingClientRect();
      const width = rect.width || 800;
      const height = rect.height || 600;
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
        canvas.width = Math.max(1, Math.floor(width * dpr));
        canvas.height = Math.max(1, Math.floor(height * dpr));
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
      }

      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, width, height);

      context.fillStyle = "rgba(7,17,27,0.22)";
      context.fillRect(0, 0, width, height);

      context.save();
      context.translate(panOffset.x, panOffset.y);
      context.scale(zoom, zoom);

      const lodNodes = zoom < 0.45 ? nodes.slice(0, 12) : nodes;
      const lodIds = new Set(lodNodes.map((node) => node.id));

      for (const edge of data?.edges ?? []) {
        if (!lodIds.has(edge.source) || !lodIds.has(edge.target)) {
          continue;
        }
        const source = positionsRef.current.get(edge.source);
        const target = positionsRef.current.get(edge.target);
        if (!source || !target) {
          continue;
        }
        context.beginPath();
        context.strokeStyle = `rgba(67,216,201,${clamp(edge.similarity, 0.12, 0.44)})`;
        context.lineWidth = 0.8 + edge.similarity * 2;
        context.moveTo(source.x, source.y);
        context.lineTo(target.x, target.y);
        context.stroke();
      }

      for (const node of lodNodes) {
        const pos = positionsRef.current.get(node.id);
        if (!pos) {
          continue;
        }
        const radius = clamp(12 + Math.sqrt(node.media_count) * 3, 12, 58);
        const energy = mediaEnergy(node, maxMedia);
        const isHovered = node.id === hoveredClusterId;
        const isSelected = node.id === selectedClusterId;
        const isExpanded = node.id === expandedId;
        const animatedRadius = radius * (isExpanded ? 1 + expandProgress * 0.25 : 1);

        if (isHovered || isSelected) {
          context.shadowColor = stratumColor(node.dominant_depth_stratum);
          context.shadowBlur = isSelected ? 24 : 14;
        } else {
          context.shadowBlur = 0;
        }

        context.beginPath();
        context.arc(pos.x, pos.y, animatedRadius, 0, Math.PI * 2);
        const gradient = context.createRadialGradient(
          pos.x - animatedRadius * 0.3,
          pos.y - animatedRadius * 0.3,
          0,
          pos.x,
          pos.y,
          animatedRadius,
        );
        gradient.addColorStop(0, energyColor(energy, 0.82));
        gradient.addColorStop(1, energyColor(energy, 0.34));
        context.fillStyle = gradient;
        context.fill();
        context.strokeStyle = isSelected ? stratumColor(node.dominant_depth_stratum) : "rgba(255,255,255,0.12)";
        context.lineWidth = isSelected ? 2 : 1;
        context.stroke();
        context.shadowBlur = 0;

        if (zoom > 0.55 || isHovered || isSelected) {
          context.fillStyle = "rgba(236,243,251,0.92)";
          context.font = `${Math.max(9, 12 / zoom)}px system-ui, sans-serif`;
          context.textAlign = "center";
          context.textBaseline = "middle";
          const label = node.label.length > 16 ? `${node.label.slice(0, 14)}…` : node.label;
          context.fillText(label, pos.x, pos.y + animatedRadius + 14 / zoom);
          if (zoom > 0.8 || isHovered || isSelected) {
            context.font = `${Math.max(8, 10 / zoom)}px system-ui, sans-serif`;
            context.fillStyle = "rgba(157,177,198,0.88)";
            context.fillText(`${node.media_count}`, pos.x, pos.y + animatedRadius + 28 / zoom);
          }
        }
      }

      context.restore();
      rafRef.current = window.requestAnimationFrame(draw);
    };

    rafRef.current = window.requestAnimationFrame(draw);
    return () => window.cancelAnimationFrame(rafRef.current);
  }, [data?.edges, expandProgress, expandedId, hoveredClusterId, maxMedia, nodes, panOffset, selectedClusterId, zoom]);

  useEffect(() => {
    if (expandedId == null) {
      setExpandProgress(0);
      return;
    }
    let frame = 0;
    let progress = 0;
    const animate = () => {
      progress = Math.min(progress + 0.08, 1);
      setExpandProgress(progress);
      if (progress < 1) {
        frame = window.requestAnimationFrame(animate);
      }
    };
    frame = window.requestAnimationFrame(animate);
    return () => window.cancelAnimationFrame(frame);
  }, [expandedId]);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) {
      return;
    }

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        workerRef.current?.postMessage({
          type: "resize",
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []);

  const hitTest = useCallback((clientX: number, clientY: number): number | null => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return null;
    }
    const rect = canvas.getBoundingClientRect();
    const x = (clientX - rect.left - panOffset.x) / zoom;
    const y = (clientY - rect.top - panOffset.y) / zoom;
    for (const node of nodes) {
      const pos = positionsRef.current.get(node.id);
      if (!pos) {
        continue;
      }
      const radius = clamp(12 + Math.sqrt(node.media_count) * 3, 12, 58);
      const dx = x - pos.x;
      const dy = y - pos.y;
      if (dx * dx + dy * dy <= radius * radius) {
        return node.id;
      }
    }
    return null;
  }, [nodes, panOffset.x, panOffset.y, zoom]);

  const handleMouseMove = useCallback((event: ReactMouseEvent<HTMLCanvasElement>) => {
    if (dragRef.current.active) {
      setPanOffset({
        x: dragRef.current.originX + (event.clientX - dragRef.current.startX),
        y: dragRef.current.originY + (event.clientY - dragRef.current.startY),
      });
      return;
    }
    const id = hitTest(event.clientX, event.clientY);
    setHoveredClusterId(id);
    if (canvasRef.current) {
      canvasRef.current.style.cursor = id != null ? "pointer" : "grab";
    }
  }, [hitTest]);

  const handleMouseDown = useCallback((event: ReactMouseEvent<HTMLCanvasElement>) => {
    const id = hitTest(event.clientX, event.clientY);
    if (id != null) {
      return;
    }
    dragRef.current = {
      active: true,
      startX: event.clientX,
      startY: event.clientY,
      originX: panOffset.x,
      originY: panOffset.y,
    };
  }, [hitTest, panOffset.x, panOffset.y]);

  const endDrag = useCallback(() => {
    dragRef.current.active = false;
    if (canvasRef.current) {
      canvasRef.current.style.cursor = "grab";
    }
  }, []);

  const handleClick = useCallback((event: ReactMouseEvent<HTMLCanvasElement>) => {
    const id = hitTest(event.clientX, event.clientY);
    if (id == null) {
      return;
    }
    onNodeSelect(id);
    setExpandedId(id);
    setExpandProgress(0);
  }, [hitTest, onNodeSelect]);

  const handleDoubleClick = useCallback((event: ReactMouseEvent<HTMLCanvasElement>) => {
    const id = hitTest(event.clientX, event.clientY);
    if (id == null) {
      return;
    }
    onNodeExpand(id);
  }, [hitTest, onNodeExpand]);

  const handleWheel = useCallback((event: ReactMouseEvent<HTMLCanvasElement> & WheelEvent) => {
    event.preventDefault();
    const delta = "deltaY" in event ? event.deltaY : 0;
    setZoom((current) => clamp(current - delta * 0.001, 0.2, 4));
  }, []);

  const handleKeyDown = useCallback((event: React.KeyboardEvent<HTMLDivElement>) => {
    if (nodes.length === 0) {
      return;
    }
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      const next = (focusedIdx + 1) % nodes.length;
      setFocusedIdx(next);
      onNodeSelect(nodes[next].id);
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      const prev = (focusedIdx - 1 + nodes.length) % nodes.length;
      setFocusedIdx(prev);
      onNodeSelect(nodes[prev].id);
    } else if (event.key === "Enter" || event.key === " ") {
      const node = nodes[focusedIdx];
      if (node) {
        onNodeExpand(node.id);
      }
    } else if (event.key === "+" || event.key === "=") {
      setZoom((current) => clamp(current + 0.2, 0.2, 4));
    } else if (event.key === "-") {
      setZoom((current) => clamp(current - 0.2, 0.2, 4));
    } else if (event.key === "0") {
      setZoom(1);
      setPanOffset({ x: 0, y: 0 });
    }
  }, [focusedIdx, nodes, onNodeExpand, onNodeSelect]);

  if (loading) {
    return (
      <div className="smriti-mandala-container smriti-mandala-container--loading">
        <p className="muted">Rendering memory clusters…</p>
      </div>
    );
  }

  if (nodes.length === 0) {
    return (
      <div
        ref={containerRef}
        className={`smriti-mandala-container ${className || ""}`}
        style={style}
        role="region"
        aria-label="Memory map — no clusters yet"
      >
        <div className="smriti-mandala-container--loading">
          <div style={{ textAlign: "center" }}>
            <p style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>◎</p>
            <p className="muted">No memory clusters yet.</p>
            <p className="muted" style={{ fontSize: "0.85rem" }}>
              Add a watch folder in Settings → Smriti Storage to start indexing media.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={`smriti-mandala-container ${className || ""}`}
      style={style}
      role="application"
      aria-label={`Memory map with ${nodes.length} clusters. Use arrow keys to navigate and Enter to expand.`}
      tabIndex={0}
      onKeyDown={handleKeyDown}
    >
      <canvas
        ref={canvasRef}
        className="smriti-mandala-canvas"
        aria-hidden="true"
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={endDrag}
        onMouseLeave={endDrag}
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
        onWheel={(event) => handleWheel(event as unknown as ReactMouseEvent<HTMLCanvasElement> & WheelEvent)}
      />

      <div className="mandala-controls" aria-label="Map controls">
        <button type="button" className="mandala-ctrl-btn" onClick={() => setZoom((current) => clamp(current + 0.25, 0.2, 4))} aria-label="Zoom in">
          +
        </button>
        <button
          type="button"
          className="mandala-ctrl-btn"
          onClick={() => {
            setZoom(1);
            setPanOffset({ x: 0, y: 0 });
          }}
          aria-label="Reset view"
        >
          ⊙
        </button>
        <button type="button" className="mandala-ctrl-btn" onClick={() => setZoom((current) => clamp(current - 0.25, 0.2, 4))} aria-label="Zoom out">
          −
        </button>
      </div>

      <div className="mandala-zoom-label" aria-live="polite">
        {Math.round(zoom * 100)}%
      </div>

      {hoveredNode ? (
        <div
          className="smriti-cluster-tooltip"
          style={{
            left: `${((positionsRef.current.get(hoveredNode.id)?.x ?? 0) + panOffset.x) / zoom}px`,
            top: `${((positionsRef.current.get(hoveredNode.id)?.y ?? 0) + panOffset.y) / zoom}px`,
          }}
          role="tooltip"
        >
          <p className="eyebrow">{hoveredNode.label}</p>
          <strong>{hoveredNode.media_count} memories</strong>
          <p className="muted">
            Depth: {hoveredNode.dominant_depth_stratum || "mixed"}
            <br />
            Span: {hoveredNode.temporal_span_days ? `${hoveredNode.temporal_span_days.toFixed(1)} days` : "recent"}
          </p>
        </div>
      ) : null}

      <ul className="smriti-sr-list" aria-label="Memory clusters">
        {nodes.map((node, index) => (
          <li key={`sr-${node.id}`}>
            <button
              type="button"
              onClick={() => {
                setFocusedIdx(index);
                onNodeSelect(node.id);
              }}
              onDoubleClick={() => onNodeExpand(node.id)}
              aria-pressed={selectedClusterId === node.id}
            >
              {node.label} with {node.media_count} memories
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
