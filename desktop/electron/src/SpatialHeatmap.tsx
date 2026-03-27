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

  if (!flatValues.some((value) => value > 0.15)) {
    return null;
  }

  return (
    <div className="spatial-heatmap" aria-hidden="true">
      {flatValues.map((value, index) => {
        const active = value > 0.15;
        const opacity = active ? Math.min(value * 0.6, 0.28) : 0;
        return (
          <div
            key={`energy-cell-${index}`}
            className={active ? "spatial-heatmap__cell is-active" : "spatial-heatmap__cell"}
            style={{ opacity, backgroundColor: active ? "rgba(67, 216, 201, 1)" : "transparent" }}
          />
        );
      })}
    </div>
  );
}
