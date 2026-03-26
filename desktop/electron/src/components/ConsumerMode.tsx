import type { CSSProperties, HTMLAttributes } from "react";

export type ConsumerModeNode = {
  id: string;
  label: string;
  x: number;
  y: number;
  radius?: number;
  tone?: "accent" | "memory" | "live" | "stable";
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

const defaultNodes: ConsumerModeNode[] = [
  { id: "you", label: "You", x: 50, y: 50, radius: 20, tone: "accent" },
  { id: "recent", label: "Recent", x: 24, y: 32, radius: 14, tone: "live" },
  { id: "saved", label: "Saved", x: 74, y: 30, radius: 14, tone: "memory" },
  { id: "shared", label: "Shared", x: 70, y: 72, radius: 16, tone: "stable" },
  { id: "discover", label: "Discover", x: 28, y: 70, radius: 16, tone: "accent" },
];

const defaultLinks: ConsumerModeLink[] = [
  { source: "you", target: "recent", strength: 0.8 },
  { source: "you", target: "saved", strength: 0.7 },
  { source: "you", target: "shared", strength: 0.6 },
  { source: "you", target: "discover", strength: 0.55 },
];

function buildNodeMap(nodes: ConsumerModeNode[]) {
  return new Map(nodes.map((node) => [node.id, node]));
}

export default function ConsumerMode({
  copy,
  nodes = defaultNodes,
  links = defaultLinks,
  query = "",
  className,
  style,
  onQueryChange,
  onAction,
  ...rest
}: ConsumerModeProps) {
  const ui = { ...defaultCopy, ...copy };
  const nodeMap = buildNodeMap(nodes);

  return (
    <section
      className={mergeClassNames("panel consumer-mode", className)}
      style={style}
      {...rest}
    >
      <div className="panel-head">
        <div>
          <p className="eyebrow">{ui.title}</p>
          <h3>{ui.subtitle}</h3>
        </div>
        <button type="button" className="primary" onClick={onAction}>
          {ui.actionLabel}
        </button>
      </div>

      <div className="consumer-mode__layout">
        <div className="consumer-mode__copy">
          <label className="consumer-mode__field">
            <span>{ui.searchLabel}</span>
            <input
              type="search"
              value={query}
              onChange={(event) => onQueryChange?.(event.target.value)}
              placeholder={ui.searchLabel}
            />
          </label>

          <div className="consumer-mode__status">
            <span>{ui.statusLabel}</span>
            <strong>{nodes.length}</strong>
          </div>

          <p className="muted">{ui.emptyLabel}</p>
        </div>

        <div className="consumer-mode__graph" aria-hidden="true">
          <svg viewBox="0 0 100 100" preserveAspectRatio="none">
            {links.map((link, index) => {
              const source = nodeMap.get(link.source);
              const target = nodeMap.get(link.target);
              if (!source || !target) {
                return null;
              }

              return (
                <line
                  key={`${link.source}-${link.target}-${index}`}
                  x1={source.x}
                  y1={source.y}
                  x2={target.x}
                  y2={target.y}
                  className="consumer-mode__link"
                  strokeWidth={Math.max((link.strength ?? 0.5) * 1.8, 0.75)}
                />
              );
            })}

            {nodes.map((node) => (
              <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
                <circle
                  r={node.radius ?? 12}
                  className={mergeClassNames(
                    "consumer-mode__node",
                    node.tone ? `is-${node.tone}` : undefined,
                  )}
                />
                <text className="consumer-mode__label" textAnchor="middle" dy="24">
                  {node.label}
                </text>
              </g>
            ))}
          </svg>
        </div>
      </div>
    </section>
  );
}
