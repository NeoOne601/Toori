import type { CSSProperties, HTMLAttributes, ReactNode } from "react";
import SpatialHeatmap from "../SpatialHeatmap";

export type SpatialCanvasGhostBox = {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  label?: string;
  score?: number | null;
  occluded?: boolean;
  depth?: number | null;
};

export type SpatialCanvasAnchor = {
  id: string;
  x: number;
  y: number;
  z?: number | null;
  label?: string;
  tone?: "live" | "stable" | "memory" | "accent";
};

type SpatialCanvas3DProps = {
  warmup?: boolean;
  energyMap?: number[][];
  statusLabel?: string | null;
  ghosts?: SpatialCanvasGhostBox[];
  anchors?: SpatialCanvasAnchor[];
  children?: ReactNode;
  className?: string;
  style?: CSSProperties;
} & HTMLAttributes<HTMLDivElement>;

function mergeClassNames(...parts: Array<string | undefined | false>) {
  return parts.filter(Boolean).join(" ");
}

export default function SpatialCanvas3D({
  warmup = false,
  energyMap = [],
  statusLabel = null,
  ghosts = [],
  anchors = [],
  children,
  className,
  style,
  ...rest
}: SpatialCanvas3DProps) {
  const flatEnergy = energyMap.flat();
  const hasContent = children || ghosts.length > 0 || anchors.length > 0 || flatEnergy.length > 0 || Boolean(statusLabel);
  if (!hasContent) {
    return null;
  }

  return (
    <div
      className={mergeClassNames("spatial-canvas3d", className)}
      style={style}
      aria-hidden="true"
      {...rest}
    >
      <div className="spatial-canvas3d__stage">
        <SpatialHeatmap energyMap={energyMap} warmup={warmup} statusLabel={statusLabel} />

        {anchors.map((anchor) => (
          <div
            key={anchor.id}
            className={mergeClassNames(
              "spatial-canvas3d__anchor",
              anchor.tone ? `is-${anchor.tone}` : undefined,
            )}
            style={
              {
                left: `${anchor.x}%`,
                top: `${anchor.y}%`,
                transform: `translate3d(-50%, -50%, ${anchor.z ?? 0}px)`,
              } as CSSProperties
            }
          >
            <span>{anchor.label ?? anchor.id}</span>
          </div>
        ))}

        {ghosts.map((ghost) => (
          <div
            key={ghost.id}
            className={mergeClassNames(
              "spatial-canvas3d__ghost",
              ghost.occluded ? "is-occluded" : undefined,
            )}
            style={
              {
                left: `${ghost.x}%`,
                top: `${ghost.y}%`,
                width: `${ghost.width}%`,
                height: `${ghost.height}%`,
                transform: `translate3d(0, 0, ${ghost.depth ?? 0}px)`,
              } as CSSProperties
            }
          >
            <span className="spatial-canvas3d__ghost-label">
              {ghost.label ?? ghost.id}
            </span>
            {ghost.score != null ? (
              <span className="spatial-canvas3d__ghost-score">
                {Math.round(ghost.score * 100)}%
              </span>
            ) : null}
          </div>
        ))}

        {children}
      </div>
    </div>
  );
}
