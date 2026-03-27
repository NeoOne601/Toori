import type {
  BoundingBox,
  EntityTrack,
  JEPATickPayload,
  Observation,
  SceneState,
  SpatialCanvasAnchor,
  SpatialCanvasGhostBox,
} from "../types";

export function formatLatency(latencyMs?: number | null): string {
  if (latencyMs == null) {
    return "n/a";
  }
  return `${latencyMs.toFixed(0)} ms`;
}

export function humanizeLabel(value?: string | null): string {
  return String(value || "").replace(/_/g, " ").trim();
}

function metadataValue(metadata: Record<string, unknown> | undefined, key: string): string {
  return humanizeLabel(typeof metadata?.[key] === "string" ? (metadata[key] as string) : "");
}

export function formatEntityDisplayLabel(entity: {
  label?: string | null;
  metadata?: Record<string, unknown> | null;
  last_similarity?: number | null;
}): string {
  const label = formatEntityBaseLabel(entity);
  const confidenceText =
    entity.last_similarity != null ? ` ${(entity.last_similarity * 100).toFixed(0)}%` : "";
  return `${label}${confidenceText}`;
}

export function formatEntityBaseLabel(entity: {
  label?: string | null;
  metadata?: Record<string, unknown> | null;
}): string {
  return (
    humanizeLabel(entity.label) ||
    metadataValue(entity.metadata ?? undefined, "caption") ||
    metadataValue(entity.metadata ?? undefined, "top_label") ||
    "tracked region"
  );
}

export function formatRelativeTime(value?: string | null): string {
  if (!value) {
    return "unknown time";
  }
  const target = new Date(value);
  if (Number.isNaN(target.getTime())) {
    return value;
  }
  const now = Date.now();
  const deltaMs = target.getTime() - now;
  const abs = Math.abs(deltaMs);
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  const units: Array<[Intl.RelativeTimeFormatUnit, number]> = [
    ["year", 1000 * 60 * 60 * 24 * 365],
    ["month", 1000 * 60 * 60 * 24 * 30],
    ["week", 1000 * 60 * 60 * 24 * 7],
    ["day", 1000 * 60 * 60 * 24],
    ["hour", 1000 * 60 * 60],
    ["minute", 1000 * 60],
  ];
  for (const [unit, size] of units) {
    if (abs >= size) {
      return rtf.format(Math.round(deltaMs / size), unit);
    }
  }
  return rtf.format(Math.round(deltaMs / 1000), "second");
}

export function normalizeBoxes(
  boxes: unknown,
  observation?: Observation | null,
): BoundingBox[] {
  if (!Array.isArray(boxes)) {
    return [];
  }
  const width = Math.max(typeof observation?.width === "number" ? observation.width : 1, 1);
  const height = Math.max(typeof observation?.height === "number" ? observation.height : 1, 1);
  return boxes.flatMap((item) => {
    if (!item || typeof item !== "object") {
      return [];
    }
    const raw = item as Record<string, unknown>;
    const x = Number(raw.x);
    const y = Number(raw.y);
    const boxWidth = Number(raw.width);
    const boxHeight = Number(raw.height);
    if (![x, y, boxWidth, boxHeight].every(Number.isFinite)) {
      return [];
    }
    const normalized =
      boxWidth <= 1.2 && boxHeight <= 1.2 && x <= 1.2 && y <= 1.2
        ? { x, y, width: boxWidth, height: boxHeight }
        : { x: x / width, y: y / height, width: boxWidth / width, height: boxHeight / height };
    const metadata =
      raw.metadata && typeof raw.metadata === "object" ? (raw.metadata as Record<string, unknown>) : null;
    return [
      {
        ...normalized,
        label: raw.label ? String(raw.label) : null,
        score: raw.score == null ? null : Number(raw.score),
        metadata,
      },
    ];
  });
}

export function explainSummary(
  observation?: Observation | null,
  answerText?: string | null,
): string {
  if (answerText && answerText.trim()) {
    return answerText.trim();
  }
  if (!observation) {
    return "Waiting for live scene updates.";
  }
  const rawSummary = String(observation.summary || "").trim();
  const metadata = (observation.metadata || {}) as Record<string, unknown>;
  const perception =
    metadata.perception && typeof metadata.perception === "object"
      ? (metadata.perception as Record<string, unknown>)
      : {};
  const topLabel = String(perception.top_label || observation.tags?.[0] || "")
    .replace(/_/g, " ")
    .trim();
  const color = String(perception.dominant_color || "").trim();
  const brightness = String(perception.brightness_label || "").trim();
  const edge = String(perception.edge_label || "").trim();
  if (rawSummary && !/(dominant|balanced|textured|smooth)\s+scene$/i.test(rawSummary)) {
    return rawSummary;
  }
  const parts = [];
  if (topLabel) {
    parts.push(`Likely primary object: ${topLabel}.`);
  }
  if (color || brightness || edge) {
    parts.push(
      `Visual cues: ${[color && `${color} color`, brightness && `${brightness} lighting`, edge && `${edge} detail`] 
        .filter(Boolean)
        .join(", ")}.`,
    );
  }
  if (!parts.length && rawSummary) {
    return `Local visual reading: ${rawSummary}.`;
  }
  return parts.join(" ");
}

export function explainWorldModel(sceneState?: SceneState | null): string {
  if (!sceneState) {
    return "The world model is waiting for a few frames before it can tell you what stayed stable and what changed.";
  }
  const stable = sceneState.stable_elements.slice(0, 3);
  const changed = sceneState.changed_elements.slice(0, 2);
  if (stable.length && changed.length) {
    return `Expected the scene to stay consistent around ${stable.join(", ")}. The biggest change detected was ${changed.join(", ")}.`;
  }
  if (stable.length) {
    return `The model expects continuity around ${stable.join(", ")} based on the recent scene history.`;
  }
  if (changed.length) {
    return `The model detected change in ${changed.join(", ")} and is updating its scene memory.`;
  }
  return "The model is still building a stable scene hypothesis from the recent frames.";
}

export function challengeStepExpectation(index: number): string {
  switch (index) {
    case 0:
      return "Give the model a stable baseline. Continuity should climb while novelty and surprise stay low.";
    case 1:
      return "A partial occlusion should keep persistence high while continuity dips only slightly.";
    case 2:
      return "A full occlusion should move the tracked entity to an occluded state without erasing it from memory.";
    case 3:
      return "When the object returns, the persistence graph should recover the same track instead of inventing a new one.";
    case 4:
      return "Moving away and back should preserve the main scene hypothesis and reconnect the remembered state.";
    case 5:
      return "An unexpected distractor should produce a visible surprise spike while stable objects remain persistent.";
    default:
      return "Follow the live instruction and watch the continuity, surprise, and persistence signals update.";
  }
}

export function toForecastValue(
  tick: JEPATickPayload | null | undefined,
  horizon: 1 | 2 | 5,
): number | null {
  if (!tick) {
    return null;
  }
  const raw = tick.forecast_errors?.[String(horizon)];
  return typeof raw === "number" ? raw : null;
}

export function toGhostBoxes(
  tracks: EntityTrack[],
  observation?: Observation | null,
): SpatialCanvasGhostBox[] {
  const width = Math.max(observation?.width || 1, 1);
  const height = Math.max(observation?.height || 1, 1);
  return tracks.flatMap((track) => {
    const metadata = (track.metadata || {}) as Record<string, any>;
    const bbox = metadata.ghost_bbox_pixels || metadata.bbox_pixels;
    if (!bbox || typeof bbox !== "object") {
      return [];
    }
    const x = Number(bbox.x);
    const y = Number(bbox.y);
    const boxWidth = Number(bbox.width);
    const boxHeight = Number(bbox.height);
    if (![x, y, boxWidth, boxHeight].every(Number.isFinite)) {
      return [];
    }
    return [
      {
        id: track.id,
        x: (x / width) * 100,
        y: (y / height) * 100,
        width: (boxWidth / width) * 100,
        height: (boxHeight / height) * 100,
        label: formatEntityDisplayLabel(track),
        score: track.last_similarity,
        occluded: track.status === "occluded",
        depth: track.status === "occluded" ? 18 : 0,
      },
    ];
  });
}

export function toEnergyAnchors(energyMap?: number[][] | null): SpatialCanvasAnchor[] {
  if (!energyMap?.length) {
    return [];
  }
  const flat = energyMap.flatMap((row, rowIndex) =>
    row.map((value, colIndex) => ({ rowIndex, colIndex, value })),
  );
  const max = flat.reduce((best, item) => Math.max(best, item.value), 0);
  if (max <= 0) {
    return [];
  }
  return flat
    .filter((item) => item.value > max * 0.35)
    .sort((left, right) => right.value - left.value)
    .slice(0, 18)
    .map((item) => ({
      id: `energy-${item.rowIndex}-${item.colIndex}`,
      x: ((item.colIndex + 0.5) / 14) * 100,
      y: ((item.rowIndex + 0.5) / 14) * 100,
      z: Math.round((item.value / max) * 90),
      label: item.value.toFixed(2),
      tone: item.value / max > 0.6 ? "accent" : "live",
    }));
}

export function consumerMessage(
  tick: JEPATickPayload | null | undefined,
  tracks: EntityTrack[],
): string {
  const lead = humanizeLabel(tracks[0]?.label) || "it";
  switch (tick?.talker_event) {
    case "OCCLUSION_START":
      return `I'm still tracking ${lead} even though I can't see it`;
    case "OCCLUSION_END":
      return `${lead} came back exactly where I expected`;
    case "PREDICTION_VIOLATION":
      return "That wasn't supposed to happen";
    case "ENTITY_APPEARED":
      return "Something new just entered";
    case "ENTITY_DISAPPEARED":
      return `${lead} left and I'm still remembering it`;
    case "SCENE_STABLE":
      return "Everything is as expected";
    default:
      if ((tick?.mean_energy ?? 0) < 0.2) {
        return "The room looks stable";
      }
      if ((tick?.mean_energy ?? 0) < 0.5) {
        return "Something shifted nearby";
      }
      return "Unexpected change detected";
  }
}

export function sigregBadge(loss?: number): string {
  if (loss == null) {
    return "—";
  }
  if (loss < 0.3) {
    return "COLLAPSE RISK";
  }
  if (loss < 0.6) {
    return "WATCH";
  }
  if (loss <= 2.0) {
    return "HEALTHY";
  }
  return "OVER-REG";
}

export function sigregColor(loss?: number): string {
  if (loss == null) {
    return "var(--text-muted)";
  }
  if (loss < 0.3) {
    return "var(--kpi-danger)";
  }
  if (loss < 0.6) {
    return "var(--kpi-watch)";
  }
  if (loss <= 2.0) {
    return "var(--kpi-healthy)";
  }
  return "#4361ee";
}

export const feColor = (value?: number | null) => {
  if (!value) {
    return "var(--text-muted)";
  }
  if (value < 50) {
    return "var(--kpi-healthy)";
  }
  if (value < 100) {
    return "var(--kpi-watch)";
  }
  return "var(--kpi-danger)";
};

export const feContext = (value?: number | null) => {
  if (!value) {
    return "Warming up";
  }
  if (value < 50) {
    return "Low surprise — prediction accurate";
  }
  if (value < 100) {
    return "Moderate prediction error";
  }
  return "High surprise — model uncertain";
};
