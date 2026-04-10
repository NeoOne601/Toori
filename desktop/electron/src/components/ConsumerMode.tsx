import { useEffect, useMemo, useRef, useState, type CSSProperties, type HTMLAttributes } from "react";
import cytoscape, { type Core, type EventObject } from "cytoscape";
import { humanizeLabel, isPlaceholderVisionLabel } from "../lib/formatting";

export type ConsumerModeNode = {
  id: string;
  label: string;
  x: number;
  y: number;
  radius?: number;
  tone?: "accent" | "memory" | "live" | "stable";
  depthStratum?: string;
  source?: string;
  confidence?: number;
  status?: string;
};

export type ConsumerModeLink = {
  source: string;
  target: string;
  strength?: number;
};

export type ConsumerModeCopy = {
  title: string;
  subtitle: string;
  searchLabel: string;
  statusLabel: string;
  actionLabel: string;
  emptyLabel: string;
};

type ConsumerModeProps = {
  copy?: Partial<ConsumerModeCopy>;
  nodes?: ConsumerModeNode[];
  links?: ConsumerModeLink[];
  query?: string;
  className?: string;
  style?: CSSProperties;
  onQueryChange?: (query: string) => void;
  onAction?: () => void;
} & HTMLAttributes<HTMLDivElement>;

const defaultCopy: ConsumerModeCopy = {
  title: "Consumer Mode",
  subtitle: "A lightweight discovery surface for everyday users.",
  searchLabel: "Search",
  statusLabel: "Live connections",
  actionLabel: "Open experience",
  emptyLabel: "No graph data yet",
};

function mergeClassNames(...parts: Array<string | undefined | false>) {
  return parts.filter(Boolean).join(" ");
}

function toneColor(tone?: ConsumerModeNode["tone"]) {
  switch (tone) {
    case "accent":
      return "#f7b441";
    case "memory":
      return "#88a7ff";
    case "stable":
      return "#54ceb0";
    default:
      return "#43d8c9";
  }
}

type DepthKey = "foreground" | "midground" | "background" | "unresolved";

function depthKey(depth?: string) {
  const normalized = humanizeLabel(depth || "").toLowerCase();
  if (normalized === "foreground" || normalized === "midground" || normalized === "background") {
    return normalized as Exclude<DepthKey, "unresolved">;
  }
  return "unresolved" as const;
}

function normalizeAxis(value: number) {
  if (!Number.isFinite(value)) {
    return 0.5;
  }
  if (value <= 1.5) {
    return Math.max(0, Math.min(value, 1));
  }
  if (value <= 100) {
    return Math.max(0, Math.min(value / 100, 1));
  }
  return Math.max(0, Math.min(value / 1000, 1));
}

function projectNodePosition(node: ConsumerModeNode, width: number, height: number) {
  const lane = depthKey(node.depthStratum);
  const laneCenters: Record<DepthKey, number> = {
    foreground: 0.22,
    midground: 0.48,
    background: 0.73,
    unresolved: 0.88,
  };
  const laneSpans: Record<DepthKey, number> = {
    foreground: 0.16,
    midground: 0.18,
    background: 0.16,
    unresolved: 0.08,
  };
  const paddingX = Math.max(34, Math.round(width * 0.07));
  const paddingY = Math.max(28, Math.round(height * 0.08));
  const usableWidth = Math.max(width - paddingX * 2, 1);
  const centeredY = laneCenters[lane] * height;
  const laneSpan = laneSpans[lane] * height;
  const x = paddingX + normalizeAxis(node.x) * usableWidth;
  const y = Math.max(
    paddingY,
    Math.min(height - paddingY, centeredY + (normalizeAxis(node.y) - 0.5) * laneSpan),
  );
  return { x, y };
}

function depthLabel(depth?: string) {
  switch (depthKey(depth)) {
    case "foreground":
      return "foreground";
    case "midground":
      return "midground";
    case "background":
      return "background";
    default:
      return "unresolved depth";
  }
}

function depthLegendLabel(depth?: string) {
  switch (depthKey(depth)) {
    case "foreground":
      return "Foreground";
    case "midground":
      return "Midground";
    case "background":
      return "Background";
    default:
      return "Unresolved";
  }
}

export default function ConsumerMode({
  copy,
  nodes = [],
  links = [],
  query = "",
  className,
  style,
  onAction,
  ...rest
}: ConsumerModeProps) {
  const ui = { ...defaultCopy, ...copy };
  const graphRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);
  const [graphSize, setGraphSize] = useState({ width: 0, height: 0 });
  const semanticNodes = useMemo(
    () =>
      nodes.filter((node) => {
        const label = humanizeLabel(node.label);
        return label && !isPlaceholderVisionLabel(label);
      }),
    [nodes],
  );
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(semanticNodes[0]?.id || null);
  const selectedNode = semanticNodes.find((node) => node.id === selectedNodeId) || semanticNodes[0] || null;
  const relatedNodeIds = useMemo(() => {
    if (!selectedNode) {
      return new Set<string>();
    }
    const next = new Set<string>();
    for (const link of links) {
      if (link.source === selectedNode.id) {
        next.add(link.target);
      }
      if (link.target === selectedNode.id) {
        next.add(link.source);
      }
    }
    return next;
  }, [links, selectedNode]);

  useEffect(() => {
    if (!semanticNodes.some((node) => node.id === selectedNodeId)) {
      setSelectedNodeId(semanticNodes[0]?.id || null);
    }
  }, [semanticNodes, selectedNodeId]);

  useEffect(() => {
    const container = graphRef.current;
    if (!container) {
      return;
    }
    const updateSize = () => {
      setGraphSize({
        width: container.clientWidth || 0,
        height: container.clientHeight || 0,
      });
    };
    updateSize();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", updateSize);
      return () => window.removeEventListener("resize", updateSize);
    }
    const observer = new ResizeObserver(() => updateSize());
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const graphElements = useMemo(() => {
    const width = Math.max(graphSize.width || 920, 1);
    const height = Math.max(graphSize.height || 480, 1);
    const nodesData = semanticNodes.map((node) => {
      const projected = projectNodePosition(node, width, height);
      const lane = depthKey(node.depthStratum);
      return {
        data: {
          id: node.id,
          label: humanizeLabel(node.label),
          tone: toneColor(node.tone),
          confidence: node.confidence ?? 0,
          status: node.status || "visible",
          source: node.source || "scene",
          depthStratum: depthLabel(node.depthStratum),
          depthKnown: lane !== "unresolved",
          size: Math.max((node.radius || 12) * 3.2, 28),
        },
        position: {
          x: projected.x,
          y: projected.y,
        },
        classes: lane === "unresolved" ? "is-unresolved-depth" : "",
      };
    });
    const linksData = links
      .filter(
        (link) =>
          semanticNodes.some((node) => node.id === link.source) &&
          semanticNodes.some((node) => node.id === link.target),
      )
      .map((link, index) => ({
        data: {
          id: `${link.source}-${link.target}-${index}`,
          source: link.source,
          target: link.target,
          strength: Math.max(link.strength ?? 0.25, 0.1),
        },
      }));
    return [...nodesData, ...linksData];
  }, [graphSize.height, graphSize.width, links, semanticNodes]);

  useEffect(() => {
    const container = graphRef.current;
    if (!container) {
      return;
    }
    cyRef.current?.destroy();
    if (!semanticNodes.length) {
      cyRef.current = null;
      return;
    }
    const cy = cytoscape({
      container,
      elements: graphElements,
      layout: { name: "preset", fit: false, padding: 0 },
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "background-color": "data(tone)",
            width: "data(size)",
            height: "data(size)",
            color: "#f7fafc",
            "font-size": 10,
            "font-weight": 700,
            "text-wrap": "wrap",
            "text-max-width": "88px",
            "text-valign": "center",
            "text-halign": "center",
            "border-width": 1.25,
            "border-color": "rgba(255,255,255,0.18)",
            "overlay-opacity": 0,
          },
        },
        {
          selector: "edge",
          style: {
            width: "mapData(strength, 0.1, 1, 1, 4)",
            "line-color": "rgba(157,177,198,0.45)",
            "curve-style": "straight",
            "opacity": 0.72,
            "target-arrow-shape": "none",
          },
        },
        {
          selector: ".selected",
          style: {
            "border-color": "#f7b441",
            "border-width": 2.5,
            "underlay-color": "#f7b441",
            "underlay-opacity": 0.28,
            "underlay-padding": 8,
          },
        },
        {
          selector: ".is-unresolved-depth",
          style: {
            "border-style": "dashed",
            "border-color": "rgba(255,255,255,0.34)",
            "background-opacity": 0.92,
          },
        },
      ],
      zoomingEnabled: false,
      panningEnabled: false,
      boxSelectionEnabled: false,
      autoungrabify: true,
      userPanningEnabled: false,
      userZoomingEnabled: false,
    });
    cy.on("tap", "node", (event: EventObject) => {
      const id = String(event.target.id());
      setSelectedNodeId(id);
    });
    if (selectedNodeId && cy.getElementById(selectedNodeId).length) {
      cy.getElementById(selectedNodeId).addClass("selected");
    }
    cy.resize();
    cyRef.current = cy;
    return () => cy.destroy();
  }, [graphElements, selectedNodeId, semanticNodes]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) {
      return;
    }
    cy.nodes().removeClass("selected");
    if (selectedNodeId && cy.getElementById(selectedNodeId).length) {
      cy.getElementById(selectedNodeId).addClass("selected");
    }
  }, [selectedNodeId]);

  const semanticLabels = Array.from(new Set(semanticNodes.map((node) => humanizeLabel(node.label)))).slice(0, 6);

  return (
    <section className={mergeClassNames("panel consumer-mode", className)} style={style} {...rest}>
      <div className="panel-head">
        <h3>{ui.title}</h3>
        <button type="button" className="primary" onClick={onAction} disabled={!onAction}>
          {ui.actionLabel}
        </button>
      </div>
      <div className="consumer-mode__layout">
        <div className="consumer-mode__graph-shell">
          <div className="consumer-mode__graph-hint">Spatial scene graph. Tap a node for details.</div>
          <div className="consumer-mode__depth-legend" aria-hidden="true">
            <span>{depthLegendLabel("foreground")}</span>
            <span>{depthLegendLabel("midground")}</span>
            <span>{depthLegendLabel("background")}</span>
            <span>{depthLegendLabel("unresolved")}</span>
          </div>
          <div ref={graphRef} className="consumer-mode__graph" />
          {!semanticNodes.length ? <div className="consumer-mode__empty">{ui.emptyLabel}</div> : null}
        </div>
        <div className="consumer-mode__copy">
          <div className="consumer-mode__status">
            <span>{ui.statusLabel}</span>
            <strong>{semanticNodes.length}</strong>
          </div>
          <div className="consumer-mode__insights">
            <strong>{ui.subtitle}</strong>
            <div className="chips consumer-mode__graph-related">
              {semanticLabels.length ? semanticLabels.map((label) => <span key={label}>{label}</span>) : <span>waiting for live world-state links</span>}
            </div>
          </div>
          <div className="consumer-mode__node-list">
            {semanticNodes.map((node) => (
              <button
                key={node.id}
                type="button"
                className={node.id === selectedNode?.id ? "consumer-mode__node-item is-active" : "consumer-mode__node-item"}
                onClick={() => setSelectedNodeId(node.id)}
              >
                <span className="consumer-mode__node-swatch" style={{ background: toneColor(node.tone) }} />
                <div>
                  <strong>{humanizeLabel(node.label)}</strong>
                  <small>
                    {node.source || "scene"} · {depthLabel(node.depthStratum)} · {node.status || "visible"}
                  </small>
                </div>
              </button>
            ))}
          </div>
          <div className="consumer-mode__graph-details">
            <strong>{selectedNode ? humanizeLabel(selectedNode.label) : ui.emptyLabel}</strong>
            {selectedNode ? (
              <>
                <span>source {selectedNode.source || "scene"}</span>
                <span>depth {depthLabel(selectedNode.depthStratum)}</span>
                <span>status {selectedNode.status || "visible"}</span>
                <span>confidence {(selectedNode.confidence ?? 0).toFixed(2)}</span>
                <div className="chips consumer-mode__graph-related">
                  {Array.from(relatedNodeIds)
                    .map((id) => semanticNodes.find((node) => node.id === id))
                    .filter((node): node is ConsumerModeNode => Boolean(node))
                    .slice(0, 5)
                    .map((node) => (
                      <span key={node.id}>{humanizeLabel(node.label)}</span>
                    ))}
                </div>
              </>
            ) : null}
          </div>
        </div>
      </div>
    </section>
  );
}
