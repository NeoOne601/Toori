import { useEffect, useMemo, useRef, useState } from "react";

const PATCH_GRID = 14;
const HEATMAP_RETAIN_MS = 420;

function normalizeEnergyGrid(energyMap: number[][]): number[][] {
  return Array.from({ length: PATCH_GRID }, (_, rowIndex) =>
    Array.from({ length: PATCH_GRID }, (_, colIndex) => Number(energyMap[rowIndex]?.[colIndex] ?? 0)),
  );
}

function quantile(values: number[], fraction: number) {
  if (!values.length) {
    return 0;
  }
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor((sorted.length - 1) * fraction)] ?? 0;
}

function energyColor(energy: number, alpha = 1) {
  const hue = Math.round(220 - (energy * 220));
  const saturation = 92;
  const lightness = Math.round(48 + (energy * 10));
  return `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;
}

type SpatialHeatmapProps = {
  energyMap?: number[][];
  warmup?: boolean;
  statusLabel?: string | null;
};

export default function SpatialHeatmap({
  energyMap = [],
  warmup = false,
  statusLabel = null,
}: SpatialHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const clearTimerRef = useRef<number | null>(null);
  const [displayGrid, setDisplayGrid] = useState<number[][]>([]);

  const normalizedInput = useMemo(
    () => (energyMap.length ? normalizeEnergyGrid(energyMap) : []),
    [energyMap],
  );

  useEffect(() => {
    const hasVisibleEnergy = normalizedInput.flat().some((value) => value > 0);
    if (hasVisibleEnergy) {
      if (clearTimerRef.current != null) {
        window.clearTimeout(clearTimerRef.current);
        clearTimerRef.current = null;
      }
      setDisplayGrid(normalizedInput);
      return;
    }
    if (!displayGrid.length) {
      return;
    }
    if (clearTimerRef.current != null) {
      window.clearTimeout(clearTimerRef.current);
    }
    clearTimerRef.current = window.setTimeout(() => {
      clearTimerRef.current = null;
      setDisplayGrid([]);
    }, HEATMAP_RETAIN_MS);
  }, [displayGrid.length, normalizedInput]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !displayGrid.length) {
      return;
    }
    const parent = canvas.parentElement;
    if (!parent) {
      return;
    }
    const width = Math.max(parent.clientWidth, 1);
    const height = Math.max(parent.clientHeight, 1);
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.scale(dpr, dpr);

    const values = displayGrid.flat();
    const floor = quantile(values, 0.2);
    const ceiling = Math.max(quantile(values, 0.95), floor + 1e-6);
    const cellWidth = width / PATCH_GRID;
    const cellHeight = height / PATCH_GRID;

    for (let rowIndex = 0; rowIndex < PATCH_GRID; rowIndex += 1) {
      for (let colIndex = 0; colIndex < PATCH_GRID; colIndex += 1) {
        const raw = Number(displayGrid[rowIndex]?.[colIndex] ?? 0);
        const normalized = Math.max(0, Math.min(1, (raw - floor) / (ceiling - floor)));
        if (normalized <= 0.02) {
          continue;
        }
        const centerX = (colIndex + 0.5) * cellWidth;
        const centerY = (rowIndex + 0.5) * cellHeight;
        const radius = Math.max(cellWidth, cellHeight) * (0.58 + (normalized * 0.68));
        const gradient = context.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
        gradient.addColorStop(0, energyColor(normalized, Math.min(0.88, 0.18 + (normalized * 0.64))));
        gradient.addColorStop(0.55, energyColor(Math.max(normalized - 0.08, 0), Math.min(0.56, 0.12 + (normalized * 0.34))));
        gradient.addColorStop(1, energyColor(Math.max(normalized - 0.22, 0), 0));
        context.fillStyle = gradient;
        context.fillRect(
          Math.max(0, centerX - radius),
          Math.max(0, centerY - radius),
          radius * 2,
          radius * 2,
        );
      }
    }
  }, [displayGrid]);

  useEffect(() => () => {
    if (clearTimerRef.current != null) {
      window.clearTimeout(clearTimerRef.current);
    }
  }, []);

  if (!displayGrid.length && !statusLabel) {
    return null;
  }

  return (
    <>
      {displayGrid.length ? (
        <div className="heatmap-canvas spatial-heatmap" aria-hidden="true">
          <canvas ref={canvasRef} />
        </div>
      ) : null}
      {(warmup || !displayGrid.length) && statusLabel ? (
        <div className="spatial-heatmap__status">{statusLabel}</div>
      ) : null}
    </>
  );
}
