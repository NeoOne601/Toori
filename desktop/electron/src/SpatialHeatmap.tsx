import { useEffect, useState } from "react";

const GRID = 28; // display grid: bilinear upsample from backend's 14×14

/** Bilinear sample from a 14×14 source into a GRID×GRID output. */
function upsample(energyMap: number[][]): number[] {
  const src = 14;
  const dst = GRID;
  const out: number[] = [];
  for (let r = 0; r < dst; r++) {
    for (let c = 0; c < dst; c++) {
      const sr = (r / (dst - 1)) * (src - 1);
      const sc = (c / (dst - 1)) * (src - 1);
      const r0 = Math.floor(sr);
      const c0 = Math.floor(sc);
      const r1 = Math.min(r0 + 1, src - 1);
      const c1 = Math.min(c0 + 1, src - 1);
      const dr = sr - r0;
      const dc = sc - c0;
      const v00 = energyMap[r0]?.[c0] ?? 0;
      const v01 = energyMap[r0]?.[c1] ?? 0;
      const v10 = energyMap[r1]?.[c0] ?? 0;
      const v11 = energyMap[r1]?.[c1] ?? 0;
      out.push(
        v00 * (1 - dr) * (1 - dc) +
        v01 * (1 - dr) * dc +
        v10 * dr * (1 - dc) +
        v11 * dr * dc,
      );
    }
  }
  return out;
}

type SpatialHeatmapProps = {
  energyMap?: number[][];
};

export default function SpatialHeatmap({
  energyMap = [],
}: SpatialHeatmapProps) {
  const flatValues = energyMap.length ? upsample(energyMap) : Array(GRID * GRID).fill(0);

  const [trailCells, setTrailCells] = useState(() =>
    Array.from({ length: GRID * GRID }, (_, index) => ({
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
