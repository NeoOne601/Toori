import { formatEntityDisplayLabel } from "../lib/formatting";
import type { BoundingBox } from "../types";

const DETECTION_COLORS = [
  "rgba(67, 216, 201, 0.85)",
  "rgba(255, 140, 66, 0.85)",
  "rgba(130, 171, 255, 0.85)",
  "rgba(246, 199, 106, 0.85)",
  "rgba(255, 107, 107, 0.85)",
  "rgba(155, 107, 255, 0.85)",
  "rgba(84, 206, 176, 0.85)",
];

type DetectionOverlayProps = {
  boxes: BoundingBox[];
  uiMode?: "consumer" | "science";
};

function getLabelColor(label: string): string {
  let hash = 0;
  for (let index = 0; index < label.length; index += 1) {
    hash = label.charCodeAt(index) + ((hash << 5) - hash);
  }
  return DETECTION_COLORS[Math.abs(hash) % DETECTION_COLORS.length];
}

export default function DetectionOverlay({
  boxes,
  uiMode = "science",
}: DetectionOverlayProps) {
  if (!boxes.length) {
    return null;
  }

  const visibleBoxes =
    uiMode === "consumer"
      ? [...boxes].sort((left, right) => (right.score ?? 0) - (left.score ?? 0))
      : boxes;

  return (
    <div className="detection-overlay" aria-hidden="true">
      {visibleBoxes.map((box, index) => {
        const displayLabel = formatEntityDisplayLabel({
          label: box.label,
          metadata: box.metadata ?? undefined,
          last_similarity: box.score,
        });
        const color = getLabelColor(displayLabel);
        return (
          <div
            key={`${displayLabel}-${index}-${box.x}-${box.y}`}
            className="detection-box"
            style={{
              left: `${Math.max(0, Math.min(box.x, 0.96)) * 100}%`,
              top: `${Math.max(0, Math.min(box.y, 0.96)) * 100}%`,
              width: `${Math.max(0.04, Math.min(box.width, 1 - box.x)) * 100}%`,
              height: `${Math.max(0.04, Math.min(box.height, 1 - box.y)) * 100}%`,
              borderColor: color,
              backgroundColor: color.replace("0.85)", "0.12)"),
            }}
          >
            <span style={{ backgroundColor: color }}>{displayLabel}</span>
          </div>
        );
      })}
    </div>
  );
}
