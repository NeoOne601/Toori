import { useEffect, useState } from "react";

type SpatialHeatmapProps = {
  energyMap?: number[][];
};

export default function SpatialHeatmap({
  energyMap = [],
}: SpatialHeatmapProps) {
  const flatValues = Array.from({ length: 14 * 14 }, (_, index) => {
    const row = Math.floor(index / 14);
    const col = index % 14;
    return energyMap[row]?.[col] ?? 0;
  });
  const [trailCells, setTrailCells] = useState(() =>
    Array.from({ length: 14 * 14 }, (_, index) => ({
      hue: (index * 17) % 360,
      opacity: 0,
      active: false,
    })),
  );

  useEffect(() => {
    setTrailCells((current) =>
      flatValues.map((value, index) => {
        const previous = current[index] || { hue: (index * 17) % 360, opacity: 0, active: false };
        if (value > 0.15) {
          return {
            hue: (performance.now() / 16 + index * 13) % 360,
            opacity: Math.min(value * 0.6, 0.28),
            active: true,
          };
        }
        return {
          hue: (previous.hue + 2) % 360,
          opacity: Math.max(previous.opacity * 0.86 - 0.01, 0),
          active: false,
        };
      }),
    );
  }, [energyMap]);

  if (!trailCells.some((cell) => cell.opacity > 0.01)) {
    return null;
  }

  return (
    <div className="spatial-heatmap" aria-hidden="true">
      {trailCells.map((cell, index) => {
        return (
          <div
            key={`energy-cell-${index}`}
            className={cell.active ? "spatial-heatmap__cell is-active" : "spatial-heatmap__cell"}
            style={{
              opacity: cell.opacity,
              backgroundColor: `hsl(${cell.hue} 92% 62%)`,
            }}
          />
        );
      })}
    </div>
  );
}
