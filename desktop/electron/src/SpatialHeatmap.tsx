import { useEffect, useState } from "react";

type EnergyMapPayload = {
  grid: [number, number]; // [height_rows, width_cols]
  values: number[]; // Flat array in row-major order
};

export default function SpatialHeatmap() {
  const [energyMap, setEnergyMap] = useState<EnergyMapPayload | null>(null);

  useEffect(() => {
    const socket = new WebSocket("ws://127.0.0.1:7777/v1/events");
    
    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "jepa.energy_map" && data.payload) {
          setEnergyMap(data.payload as EnergyMapPayload);
        }
      } catch (err) {
        // ignore parse error
      }
    };

    return () => socket.close();
  }, []);

  if (!energyMap || !energyMap.values || !energyMap.values.length) {
    return null;
  }

  const [rows, cols] = energyMap.grid;

  return (
    <div
      className={`energy-heatmap ${energyMap.values.some(v => v > 0.4) ? 'active' : ''}`}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 5,
        borderRadius: "16px",
        overflow: "hidden",
        display: "grid",
        gridTemplateColumns: `repeat(${cols}, 1fr)`,
        gridTemplateRows: `repeat(${rows}, 1fr)`,
        opacity: energyMap.values.some(v => v > 0.3) ? 1 : 0,
        transition: "opacity 0.5s ease-in-out",
      }}
      aria-hidden="true"
    >
      {energyMap.values.map((val, idx) => {
        // Higher threshold: only show significant surprise
        const active = val > 0.4;
        const opacity = active ? Math.min(val * 1.2, 0.8) : 0;

        return (
          <div
            key={idx}
            style={{
              width: "100%",
              height: "100%",
              backgroundColor: active ? `rgba(67, 216, 201, ${opacity * 0.25})` : 'transparent',
              backdropFilter: active ? `blur(${opacity * 6}px) brightness(1.2)` : 'none',
              border: active ? `1px solid rgba(67, 216, 201, ${opacity * 0.6})` : 'none',
              boxShadow: active ? `0 0 10px rgba(67, 216, 201, ${opacity * 0.3}) inset` : 'none',
              transition: "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
            }}
          />
        );
      })}
    </div>
  );
}
