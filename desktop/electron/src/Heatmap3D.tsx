import { Component, Suspense, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode, ErrorInfo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

/* ─── Types ───────────────────────────────────────────────── */

type EnergyMapPayload = {
  grid: [number, number]; // [rows, cols]
  values: number[];       // flat row-major
};

type HeatPoint = {
  x: number;         // normalized 0..1
  y: number;         // normalized 0..1
  intensity: number; // 0..1
  birthTime: number; // seconds (clock time)
};

/* ─── Custom Shader Material ─────────────────────────────── */

const heatmapVertexShader = /* glsl */ `
  attribute float aIntensity;
  attribute float aAge;
  varying float vIntensity;
  varying float vAge;

  void main() {
    vIntensity = aIntensity;
    vAge = aAge;

    // Scale point size based on intensity for organic feel
    float size = mix(0.04, 0.12, aIntensity);
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = size * (300.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const heatmapFragmentShader = /* glsl */ `
  varying float vIntensity;
  varying float vAge;

  void main() {
    // Soft circle falloff from center
    vec2 center = gl_PointCoord - vec2(0.5);
    float dist = length(center);
    if (dist > 0.5) discard;

    float softEdge = 1.0 - smoothstep(0.25, 0.5, dist);

    // Color gradient: Red(1.0) -> Orange(0.5) -> Yellow(0.1) -> Transparent(0.0)
    vec3 red    = vec3(1.0, 0.267, 0.267);   // #ff4444
    vec3 orange = vec3(1.0, 0.549, 0.259);   // #ff8c42
    vec3 yellow = vec3(1.0, 0.8, 0.267);     // #ffcc44

    vec3 color;
    if (vIntensity > 0.5) {
      color = mix(orange, red, (vIntensity - 0.5) * 2.0);
    } else if (vIntensity > 0.1) {
      color = mix(yellow, orange, (vIntensity - 0.1) / 0.4);
    } else {
      color = yellow;
    }

    // Age-based fade: 0.0 (fresh) -> 1.0 (expired)
    float ageFade = 1.0 - clamp(vAge, 0.0, 1.0);

    // Combined alpha (cap at 0.28 to prevent overlay obliteration)
    float alpha = min(vIntensity * softEdge * ageFade * 0.6, 0.28);

    // Additive glow bloom
    float glow = softEdge * vIntensity * ageFade * 0.3;
    color += vec3(glow);

    gl_FragColor = vec4(color, alpha);
  }
`;

/* ─── Point Cloud Component (inside R3F Canvas) ──────────── */

const MAX_POINTS = 2048;
const MAX_AGE_S = 5.0; // seconds before a point fully fades
const WS_URL = "ws://127.0.0.1:7777/v1/events";

function useEnergyWebSocket(onData: (payload: EnergyMapPayload) => void) {
  const callbackRef = useRef(onData);
  callbackRef.current = onData;

  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let isMounted = true;

    function connect() {
      if (!isMounted) return;
      try {
        socket = new WebSocket(WS_URL);
      } catch {
        // WebSocket construction can throw in some environments
        reconnectTimer = setTimeout(connect, 5000);
        return;
      }

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "jepa.energy_map" && data.payload) {
            callbackRef.current(data.payload as EnergyMapPayload);
          }
        } catch {
          // ignore parse errors
        }
      };

      socket.onerror = () => {
        // Silently reconnect on error
      };

      socket.onclose = () => {
        if (isMounted) {
          reconnectTimer = setTimeout(connect, 3000);
        }
      };
    }

    connect();

    return () => {
      isMounted = false;
      if (socket) {
        if (socket.readyState === WebSocket.CONNECTING) {
          socket.onopen = () => socket?.close();
          socket.onerror = () => {};
        } else {
          socket.close();
        }
      }
      if (reconnectTimer) clearTimeout(reconnectTimer);
    };
  }, []);
}

function HeatmapPoints() {
  const meshRef = useRef<THREE.Points>(null!);
  const pointsRef = useRef<HeatPoint[]>([]);
  const clockRef = useRef(0);

  // Pre-allocate buffer attributes
  const { positions, intensities, ages, geometry } = useMemo(() => {
    const pos = new Float32Array(MAX_POINTS * 3);
    const int = new Float32Array(MAX_POINTS);
    const ag = new Float32Array(MAX_POINTS);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("aIntensity", new THREE.BufferAttribute(int, 1));
    geo.setAttribute("aAge", new THREE.BufferAttribute(ag, 1));
    geo.setDrawRange(0, 0);

    return { positions: pos, intensities: int, ages: ag, geometry: geo };
  }, []);

  // WebSocket connection for energy map data
  useEnergyWebSocket((payload) => {
    const [rows, cols] = payload.grid;
    const now = clockRef.current;

    // Convert grid cells with energy > threshold to heat points
    const newPoints: HeatPoint[] = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = payload.values[r * cols + c];
        if (val > 0.05) {
          newPoints.push({
            x: (c + 0.5) / cols,
            y: (r + 0.5) / rows,
            intensity: Math.min(val, 1.0),
            birthTime: now,
          });
        }
      }
    }

    // Merge with existing, pruning old points
    const existing = pointsRef.current.filter(
      (p) => now - p.birthTime < MAX_AGE_S
    );
    pointsRef.current = [...existing, ...newPoints].slice(-MAX_POINTS);
  });

  // Animate every frame: update ages and buffer data
  useFrame((_, delta) => {
    clockRef.current += delta;
    const now = clockRef.current;
    const points = pointsRef.current;

    let count = 0;
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const age = (now - p.birthTime) / MAX_AGE_S;
      if (age >= 1.0) continue; // expired

      // Map normalized coords to 3D space (-1..1 on x/y, z=0)
      positions[count * 3] = (p.x - 0.5) * 2.0;
      positions[count * 3 + 1] = -(p.y - 0.5) * 2.0; // flip Y
      positions[count * 3 + 2] = 0;

      intensities[count] = p.intensity;
      ages[count] = age;
      count++;
    }

    // Update GPU buffers
    const posAttr = geometry.getAttribute("position") as THREE.BufferAttribute;
    const intAttr = geometry.getAttribute("aIntensity") as THREE.BufferAttribute;
    const ageAttr = geometry.getAttribute("aAge") as THREE.BufferAttribute;

    posAttr.needsUpdate = true;
    intAttr.needsUpdate = true;
    ageAttr.needsUpdate = true;
    geometry.setDrawRange(0, count);
  });

  const shaderMaterial = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: heatmapVertexShader,
        fragmentShader: heatmapFragmentShader,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    []
  );

  return (
    <points ref={meshRef} geometry={geometry} material={shaderMaterial} />
  );
}

/* ─── Error Boundary ─────────────────────────────────────── */

class HeatmapErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean }
> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): { hasError: boolean } {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.warn("[Heatmap3D] WebGL error, falling back to hidden:", error.message);
  }

  render() {
    if (this.state.hasError) {
      return null; // Silently hide heatmap if WebGL fails
    }
    return this.props.children;
  }
}

/* ─── Exported Heatmap3D Component ───────────────────────── */

function Heatmap3DCanvas() {
  return (
    <Canvas
      dpr={1}
      orthographic
      camera={{ zoom: 1, near: 0.1, far: 100, position: [0, 0, 5] }}
      gl={{
        alpha: true,
        antialias: true,
        premultipliedAlpha: false,
        powerPreference: "low-power",
        failIfMajorPerformanceCaveat: false,
      }}
      style={{
        width: "100%",
        height: "100%",
        background: "transparent",
      }}
      frameloop="always"
    >
      <HeatmapPoints />
    </Canvas>
  );
}

export default function Heatmap3D() {
  const [hasData, setHasData] = useState(false);

  // Listen for energy map events to show/hide the canvas
  useEnergyWebSocket(() => {
    if (!hasData) setHasData(true);
  });

  return (
    <div
      className="heatmap-canvas"
      style={{
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
        borderRadius: "20px",
        overflow: "hidden",
        zIndex: 5,
        opacity: hasData ? 1 : 0,
        transition: "opacity 0.6s ease-in-out",
      }}
      aria-hidden="true"
    >
      <HeatmapErrorBoundary>
        <Suspense fallback={null}>
          <Heatmap3DCanvas />
        </Suspense>
      </HeatmapErrorBoundary>
    </div>
  );
}
