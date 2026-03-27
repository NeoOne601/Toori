import type { ReactNode, RefObject } from "react";
import SpatialCanvas3D from "../components/SpatialCanvas3D";
import type {
  BoundingBox,
  SpatialCanvasAnchor,
  SpatialCanvasGhostBox,
} from "../types";
import DetectionOverlay from "../widgets/DetectionOverlay";

type SceneMonitorProps = {
  title: string;
  subtitle?: string;
  videoRef: RefObject<HTMLVideoElement | null>;
  captureCanvasRef: RefObject<HTMLCanvasElement | null>;
  diagnosticsCanvasRef: RefObject<HTMLCanvasElement | null>;
  boxes: BoundingBox[];
  showEntities: boolean;
  showEnergyMap: boolean;
  energyMap?: number[][];
  ghosts?: SpatialCanvasGhostBox[];
  anchors?: SpatialCanvasAnchor[];
  uiMode?: "consumer" | "science";
  toneClassName?: string;
  controls?: ReactNode;
  overlay?: ReactNode;
  footer?: ReactNode;
};

export default function SceneMonitor({
  title,
  subtitle,
  videoRef,
  captureCanvasRef,
  diagnosticsCanvasRef,
  boxes,
  showEntities,
  showEnergyMap,
  energyMap = [],
  ghosts = [],
  anchors = [],
  uiMode = "science",
  toneClassName = "panel--live",
  controls,
  overlay,
  footer,
}: SceneMonitorProps) {
  return (
    <article className={`panel ${toneClassName} camera-panel`} data-panel="scene-monitor">
      <div className="panel-head">
        <h3>{title}</h3>
        {subtitle ? <span>{subtitle}</span> : null}
      </div>
      {controls}
      <div className="preview-surface">
        <video ref={videoRef as any} autoPlay muted playsInline />
        {showEntities ? <DetectionOverlay boxes={boxes} uiMode={uiMode} /> : null}
        {showEnergyMap ? (
          <SpatialCanvas3D
            warmup={false}
            ghosts={ghosts}
            anchors={anchors}
            energyMap={energyMap}
          />
        ) : null}
        {overlay}
      </div>
      <canvas ref={captureCanvasRef as any} hidden />
      <canvas ref={diagnosticsCanvasRef as any} hidden />
      {footer}
    </article>
  );
}
