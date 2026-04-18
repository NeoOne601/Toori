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

export function isPlaceholderVisionLabel(value?: string | null): boolean {
  const normalized = humanizeLabel(value).toLowerCase().replace(/\s+/g, " ").trim();
  if (!normalized) {
    return true;
  }
  if (
    normalized === "unknown object" ||
    normalized === "object" ||
    normalized.includes("histogram") ||
    normalized.includes("descriptor") ||
    normalized.includes("rgb ") ||
    normalized.includes("rgb+") ||
    normalized.includes("dominant color") ||
    normalized.includes("brightness label") ||
    normalized.includes("edge label")
  ) {
    return true;
  }
  if (/\b(near|behind|beside|next to|left of|right of|in front of|under|over)\b/i.test(normalized)) {
    return true;
  }
  const compact = normalized.replace(/[\s_]+/g, "");
  return /^entity-?\d+$/i.test(compact) || normalized === "entity";
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
  const metadata = entity.metadata ?? {};
  
  // High-Priority: Latent Manifold Projection (Elite Zero-Shot)
  const primary = humanizeLabel(String(metadata.primary_object_label || ""));
  if (primary && metadata.label_source === "latent_manifold_gemma4") {
    return primary;
  }

  const candidates = [
    humanizeLabel(entity.label),
    metadataValue(metadata as Record<string, unknown>, "caption"),
    metadataValue(metadata as Record<string, unknown>, "top_label"),
  ].filter((item) => item && !isPlaceholderVisionLabel(item));
  
  return candidates[0] || primary || "localized object";
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

function clampUnit(value: number): number {
  return Math.max(0, Math.min(value, 1));
}

function anchorBoundsFromPatches(patchIndices: number[]): BoundingBox | null {
  if (!patchIndices.length) {
    return null;
  }
  const rows = patchIndices.map((index) => Math.floor(index / 14));
  const cols = patchIndices.map((index) => index % 14);
  const minRow = Math.min(...rows);
  const maxRow = Math.max(...rows);
  const minCol = Math.min(...cols);
  const maxCol = Math.max(...cols);
  return {
    x: clampUnit(minCol / 14),
    y: clampUnit(minRow / 14),
    width: clampUnit((maxCol - minCol + 1) / 14),
    height: clampUnit((maxRow - minRow + 1) / 14),
    label: null,
    score: null,
  };
}

function anchorBounds(raw: Record<string, unknown>): BoundingBox | null {
  const bbox = raw.bbox_normalized;
  if (bbox && typeof bbox === "object") {
    const normalized = bbox as Record<string, unknown>;
    const x = Number(normalized.x);
    const y = Number(normalized.y);
    const width = Number(normalized.width);
    const height = Number(normalized.height);
    if ([x, y, width, height].every(Number.isFinite)) {
      return {
        x: clampUnit(x),
        y: clampUnit(y),
        width: clampUnit(width),
        height: clampUnit(height),
        label: null,
        score: null,
      };
    }
  }
  const patchIndices = Array.isArray(raw.patch_indices)
    ? raw.patch_indices.flatMap((value) => (Number.isFinite(Number(value)) ? [Number(value)] : []))
    : [];
  return anchorBoundsFromPatches(patchIndices);
}

export function boxesFromAnchorMatches(
  anchorMatches?: Array<Record<string, unknown>> | null,
): BoundingBox[] {
  if (!anchorMatches?.length) {
    return [];
  }
  return anchorMatches
    .flatMap((match, index) => {
      if (!match || typeof match !== "object") {
        return [];
      }
      const label = humanizeLabel(
        typeof match.open_vocab_label === "string" && match.open_vocab_label.trim()
          ? match.open_vocab_label
          : typeof match.template_name === "string" && match.template_name.trim()
            ? match.template_name
            : typeof match.name === "string"
              ? match.name
              : "",
      );
      if (!label) {
        return [];
      }
      const bounds = anchorBounds(match);
      if (!bounds) {
        return [];
      }
      const confidence = Number(match.confidence);
      return [
        {
          ...bounds,
          label,
          score: Number.isFinite(confidence) ? clampUnit(confidence) : null,
          metadata: {
            template_name: typeof match.template_name === "string" ? match.template_name : `anchor-${index + 1}`,
            open_vocab_label: label,
            depth_stratum: typeof match.depth_stratum === "string" ? match.depth_stratum : "unknown",
            patch_count: Array.isArray(match.patch_indices) ? match.patch_indices.length : 0,
          },
        },
      ];
    })
    .sort((left, right) => (right.score ?? 0) - (left.score ?? 0));
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
  void energyMap;
  return [];
}

export function consumerMessage(
  tick: JEPATickPayload | null | undefined,
  tracks: EntityTrack[],
): string {
  if (!tick) {
    return tracks.length ? "Refreshing live JEPA associations" : "Waiting for live JEPA tick";
  }
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
