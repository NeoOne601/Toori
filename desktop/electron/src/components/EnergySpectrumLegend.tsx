export default function EnergySpectrumLegend() {
  return (
    <div className="energy-spectrum-legend" aria-label="Energy heatmap legend">
      <div className="energy-spectrum-legend__bar" />
      <div className="energy-spectrum-legend__labels">
        <span>Violet / Blue: low change</span>
        <span>Green: stable</span>
        <span>Yellow: rising mismatch</span>
        <span>Orange: strong change</span>
        <span>Red: strongest violation</span>
      </div>
    </div>
  );
}
