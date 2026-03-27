export const TABS = [
  "Living Lens",
  "Live Lens",
  "Memory Search",
  "Session Replay",
  "Integrations",
  "Settings",
] as const;

export const DEFAULT_CHALLENGE_STEPS = [
  "Hold one stable object in view for a few seconds so the model can learn a baseline scene.",
  "Partially cover that object while keeping the rest of the scene stable.",
  "Fully hide the object for at least one update cycle.",
  "Reveal the object again in roughly the same area.",
  "Move the camera away and return to the original scene.",
  "Introduce a distractor or new object and watch the surprise score react.",
] as const;

export const LIVING_SECTIONS = [
  { id: "overview", label: "Overview", detail: "Scene delta and live metrics" },
  { id: "memory", label: "Memory", detail: "Continuity memory and entity tracks" },
  { id: "challenge", label: "Challenge Lab", detail: "Guided proof run and baselines" },
] as const;

export type AppTab = (typeof TABS)[number];
export type LivingSection = (typeof LIVING_SECTIONS)[number]["id"];
