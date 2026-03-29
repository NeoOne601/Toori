type WorkerCluster = {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  pinned: boolean;
};

let clusters: WorkerCluster[] = [];
let width = 800;
let height = 600;
let alpha = 1;
let ticking = false;
let tickHandle: ReturnType<typeof setTimeout> | null = null;

const ALPHA_DECAY = 0.0228;
const VELOCITY_DECAY = 0.4;

function centerForce() {
  const cx = width / 2;
  const cy = height / 2;
  const strength = 0.05 * alpha;
  for (const cluster of clusters) {
    if (cluster.pinned) {
      continue;
    }
    cluster.vx += (cx - cluster.x) * strength;
    cluster.vy += (cy - cluster.y) * strength;
  }
}

function collide(a: WorkerCluster, b: WorkerCluster) {
  const dx = a.x - b.x || (Math.random() - 0.5) * 0.01;
  const dy = a.y - b.y || (Math.random() - 0.5) * 0.01;
  const dist = Math.sqrt(dx * dx + dy * dy) || 1;
  const minDist = a.radius + b.radius + 10;
  if (dist >= minDist) {
    return;
  }
  const force = ((minDist - dist) / dist) * 0.5 * alpha;
  a.vx += dx * force;
  a.vy += dy * force;
  b.vx -= dx * force;
  b.vy -= dy * force;
}

function applyVelocity() {
  for (const cluster of clusters) {
    if (cluster.pinned) {
      cluster.vx = 0;
      cluster.vy = 0;
      continue;
    }
    cluster.vx *= 1 - VELOCITY_DECAY;
    cluster.vy *= 1 - VELOCITY_DECAY;
    cluster.x += cluster.vx;
    cluster.y += cluster.vy;
    const r = cluster.radius;
    cluster.x = Math.max(r, Math.min(width - r, cluster.x));
    cluster.y = Math.max(r, Math.min(height - r, cluster.y));
  }
}

function emitPositions() {
  const positions = clusters.map((cluster) => [cluster.id, cluster.x, cluster.y] as [number, number, number]);
  self.postMessage({ type: "positions", positions });
}

function tick() {
  if (!ticking || clusters.length === 0) {
    return;
  }
  alpha *= 1 - ALPHA_DECAY;
  if (alpha < 0.001) {
    alpha = 0.001;
  }
  centerForce();
  for (let index = 0; index < clusters.length; index += 1) {
    for (let other = index + 1; other < clusters.length; other += 1) {
      collide(clusters[index], clusters[other]);
    }
  }
  applyVelocity();
  emitPositions();
  tickHandle = setTimeout(tick, 33);
}

function clearTick() {
  if (tickHandle) {
    clearTimeout(tickHandle);
    tickHandle = null;
  }
}

self.onmessage = (event: MessageEvent) => {
  const message = event.data as
    | { type: "init"; clusters: Array<{ id: number; mediaCount: number }>; width: number; height: number }
    | { type: "resize"; width: number; height: number }
    | { type: "tick_pause" }
    | { type: "tick_resume" }
    | { type: "pin"; clusterId: number; x: number; y: number }
    | { type: "release"; clusterId: number };

  switch (message.type) {
    case "init": {
      width = message.width;
      height = message.height;
      alpha = 1;
      clusters = message.clusters.map((cluster, index) => {
        const angle = (index / Math.max(message.clusters.length, 1)) * Math.PI * 2;
        const radius = Math.min(width, height) * 0.32;
        return {
          id: cluster.id,
          x: width / 2 + Math.cos(angle) * radius + (Math.random() - 0.5) * 24,
          y: height / 2 + Math.sin(angle) * radius + (Math.random() - 0.5) * 24,
          vx: 0,
          vy: 0,
          radius: Math.max(14, Math.min(54, Math.sqrt(cluster.mediaCount) * 3)),
          pinned: false,
        };
      });
      ticking = true;
      clearTick();
      tick();
      break;
    }
    case "resize":
      width = message.width;
      height = message.height;
      break;
    case "tick_pause":
      ticking = false;
      clearTick();
      break;
    case "tick_resume":
      if (!ticking) {
        ticking = true;
        alpha = 0.3;
        clearTick();
        tick();
      }
      break;
    case "pin": {
      const cluster = clusters.find((item) => item.id === message.clusterId);
      if (cluster) {
        cluster.pinned = true;
        cluster.x = message.x;
        cluster.y = message.y;
      }
      break;
    }
    case "release": {
      const cluster = clusters.find((item) => item.id === message.clusterId);
      if (cluster) {
        cluster.pinned = false;
      }
      break;
    }
    default:
      break;
  }
};
