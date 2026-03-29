import { useEffect, useRef, useState } from "react";
import type { SmritiClusterNode, SmritiMandalaData } from "../../types";

type MandalaViewProps = {
  data: SmritiMandalaData | null;
  selectedClusterId: number | null;
  onNodeClick: (clusterId: number) => void;
  loading?: boolean;
};

type PositionedNode = SmritiClusterNode & {
  x: number;
  y: number;
};

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function hexToRgba(hex: string, alpha: number) {
  const normalized = hex.replace("#", "");
  const expanded = normalized.length === 3
    ? normalized
        .split("")
        .map((char) => `${char}${char}`)
        .join("")
    : normalized;
  const safe = expanded.padEnd(6, "0").slice(0, 6);
  const red = Number.parseInt(safe.slice(0, 2), 16);
  const green = Number.parseInt(safe.slice(2, 4), 16);
  const blue = Number.parseInt(safe.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function layoutNodes(nodes: SmritiClusterNode[]): PositionedNode[] {
  return nodes.map((node, index) => {
    const centroidX = node.centroid[0];
    const centroidY = node.centroid[1];
    const hasCentroid = Number.isFinite(centroidX) && Number.isFinite(centroidY);
    const fallbackAngle = (Math.PI * 2 * index) / Math.max(nodes.length, 1);
    const fallbackRadius = 0.3 + (index % 3) * 0.08;
    const normalizedX = hasCentroid ? 0.5 + centroidX * 0.18 : 0.5 + Math.cos(fallbackAngle) * fallbackRadius;
    const normalizedY = hasCentroid ? 0.5 + centroidY * 0.18 : 0.5 + Math.sin(fallbackAngle) * fallbackRadius;
    return {
      ...node,
      x: clamp(normalizedX, 0.12, 0.88),
      y: clamp(normalizedY, 0.14, 0.86),
    };
  });
}

export default function MandalaView({
  data,
  selectedClusterId,
  onNodeClick,
  loading = false,
}: MandalaViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoveredClusterId, setHoveredClusterId] = useState<number | null>(null);
  const nodes = layoutNodes(data?.nodes || []);
  const hoveredNode = nodes.find((node) => node.id === hoveredClusterId) || null;

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) {
      return;
    }

    const draw = () => {
      const context = canvas.getContext("2d");
      if (!context) {
        return;
      }

      const { clientWidth, clientHeight } = container;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(clientWidth * dpr));
      canvas.height = Math.max(1, Math.floor(clientHeight * dpr));
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, clientWidth, clientHeight);

      context.fillStyle = hexToRgba("#07111b", 0.22);
      context.fillRect(0, 0, clientWidth, clientHeight);

      const positionMap = new Map(nodes.map((node) => [node.id, node] as const));
      for (const edge of data?.edges || []) {
        const source = positionMap.get(edge.source);
        const target = positionMap.get(edge.target);
        if (!source || !target) {
          continue;
        }
        context.beginPath();
        context.strokeStyle = hexToRgba("#43d8c9", clamp(edge.similarity, 0.16, 0.58));
        context.lineWidth = 1 + edge.similarity * 2;
        context.moveTo(source.x * clientWidth, source.y * clientHeight);
        context.lineTo(target.x * clientWidth, target.y * clientHeight);
        context.stroke();
      }

      for (const node of nodes) {
        const radius = clamp(10 + node.media_count * 0.65, 12, 34);
        const isSelected = node.id === selectedClusterId;
        const isHovered = node.id === hoveredClusterId;
        context.beginPath();
        context.fillStyle = isSelected
          ? hexToRgba("#ff8c42", 0.95)
          : isHovered
            ? hexToRgba("#43d8c9", 0.9)
            : hexToRgba("#9eacff", 0.78);
        context.shadowColor = isSelected ? hexToRgba("#ff8c42", 0.45) : hexToRgba("#43d8c9", 0.22);
        context.shadowBlur = isSelected ? 28 : 16;
        context.arc(node.x * clientWidth, node.y * clientHeight, radius, 0, Math.PI * 2);
        context.fill();
        context.shadowBlur = 0;
      }
    };

    draw();
    const observer = new ResizeObserver(() => draw());
    observer.observe(container);
    return () => observer.disconnect();
  }, [data?.edges, hoveredClusterId, nodes, selectedClusterId]);

  if (loading) {
    return (
      <div className="smriti-mandala-container smriti-mandala-container--loading">
        <p className="muted">Rendering memory clusters…</p>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="smriti-mandala-container"
      role="application"
      aria-label="Smriti semantic cluster map"
    >
      <canvas ref={canvasRef} className="smriti-mandala-canvas" aria-hidden="true" />
      {nodes.map((node) => (
        <button
          key={node.id}
          type="button"
          className={node.id === selectedClusterId ? "smriti-cluster-node is-selected" : "smriti-cluster-node"}
          style={{
            left: `${node.x * 100}%`,
            top: `${node.y * 100}%`,
          }}
          onClick={() => onNodeClick(node.id)}
          onMouseEnter={() => setHoveredClusterId(node.id)}
          onMouseLeave={() => setHoveredClusterId((current) => (current === node.id ? null : current))}
          onFocus={() => setHoveredClusterId(node.id)}
          onBlur={() => setHoveredClusterId((current) => (current === node.id ? null : current))}
          aria-label={`${node.label}, ${node.media_count} memories`}
        >
          <span>{node.label}</span>
          <strong>{node.media_count}</strong>
        </button>
      ))}
      {hoveredNode ? (
        <div
          className="smriti-cluster-tooltip"
          style={{ left: `${hoveredNode.x * 100}%`, top: `${hoveredNode.y * 100}%` }}
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
        {nodes.map((node) => (
          <li key={`sr-${node.id}`}>
            <button type="button" onClick={() => onNodeClick(node.id)}>
              {node.label} with {node.media_count} memories
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
