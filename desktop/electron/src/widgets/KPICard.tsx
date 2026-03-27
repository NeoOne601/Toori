import type { CSSProperties, HTMLAttributes } from "react";

type KPICardProps = {
  label: string;
  value: string;
  badge?: string;
  color?: string;
  tooltip?: string;
  priority?: "normal" | "high";
  hideValue?: boolean;
  className?: string;
  style?: CSSProperties;
} & HTMLAttributes<HTMLDivElement>;

function mergeClassNames(...parts: Array<string | undefined | false>) {
  return parts.filter(Boolean).join(" ");
}

export default function KPICard({
  label,
  value,
  badge,
  color,
  tooltip,
  priority = "normal",
  hideValue = false,
  className,
  style,
  ...rest
}: KPICardProps) {
  return (
    <article
      className={mergeClassNames("kpi-card", priority === "high" && "kpi-card--priority", className)}
      style={{ ...style, ["--kpi-color" as any]: color || "var(--text-primary)" }}
      title={tooltip}
      {...rest}
    >
      <div className="kpi-card__top">
        <span className="kpi-card__label">{label}</span>
        {badge ? <span className="kpi-card__badge">{badge}</span> : null}
      </div>
      {!hideValue ? <strong className="kpi-card__value">{value}</strong> : null}
    </article>
  );
}
