import type { CSSProperties, HTMLAttributes } from "react";

type SigRegZone = {
  label: string;
  start: number;
  end: number;
  tone?: "calm" | "watch" | "guard" | "trip";
};

type SigRegGaugeProps = {
  value: number;
  guardState: string;
  zones?: SigRegZone[];
  className?: string;
  style?: CSSProperties;
} & HTMLAttributes<HTMLDivElement>;

const defaultZones: SigRegZone[] = [
  { label: "Calm", start: 0, end: 0.34, tone: "calm" },
  { label: "Watch", start: 0.34, end: 0.62, tone: "watch" },
  { label: "Guard", start: 0.62, end: 0.84, tone: "guard" },
  { label: "Trip", start: 0.84, end: 1, tone: "trip" },
];

function mergeClassNames(...parts: Array<string | undefined | false>) {
  return parts.filter(Boolean).join(" ");
}

function clamp(value: number) {
  if (Number.isNaN(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

export default function SigRegGauge({
  value,
  guardState,
  zones = defaultZones,
  className,
  style,
  ...rest
}: SigRegGaugeProps) {
  const clamped = clamp(value);

  return (
    <section
      className={mergeClassNames("panel sigreg-gauge", className)}
      style={style}
      {...rest}
    >
      <div className="panel-head">
        <div>
          <p className="eyebrow">Signal regulation</p>
          <h3>Zones and guard state</h3>
        </div>
        <div className="sigreg-gauge__guard">
          <span>Guard</span>
          <strong>{guardState}</strong>
        </div>
      </div>

      <div className="sigreg-gauge__meter" role="img" aria-label="Signal regulation gauge">
        <div className="sigreg-gauge__zones">
          {zones.map((zone) => (
            <div
              key={zone.label}
              className={mergeClassNames(
                "sigreg-gauge__zone",
                zone.tone ? `is-${zone.tone}` : undefined,
              )}
              style={{
                left: `${zone.start * 100}%`,
                width: `${Math.max((zone.end - zone.start) * 100, 0)}%`,
              }}
            >
              <span>{zone.label}</span>
            </div>
          ))}
        </div>
        <div className="sigreg-gauge__track">
          <div className="sigreg-gauge__fill" style={{ width: `${clamped * 100}%` }} />
          <div className="sigreg-gauge__marker" style={{ left: `${clamped * 100}%` }} />
        </div>
      </div>
    </section>
  );
}
