import { useEffect, useRef, useState } from "react";
import type { AppTab } from "../constants";
import type {
  CameraAccessState,
  CameraDeviceOption,
  CameraDiagnostics,
  Settings,
} from "../types";
import { getDesktopBridge, isDesktopBridgeAvailable } from "./useRuntimeBridge";

const DEFAULT_CAMERA_DIAGNOSTICS: CameraDiagnostics = {
  phase: "idle",
  selectedDeviceId: "default",
  selectedLabel: "Auto camera",
  permissionStatus: "unknown",
  resolution: "0 x 0",
  readyState: "idle",
  trackState: "idle",
  trackMuted: false,
  trackEnabled: false,
  lastFrameAt: null,
  frameLuma: null,
  blackFrameDetected: false,
  message: "Camera not started",
  error: null,
};

type UseCameraStreamOptions = {
  activeTab: AppTab;
  settings: Settings | null;
  runtimeRequest: <T>(path: string, method?: string, body?: unknown) => Promise<T>;
  onStatusChange?: (message: string) => void;
  onSettingsChange?: (settings: Settings) => void;
};

async function getCameraAccessStateFallback(): Promise<{
  status: string;
  granted: boolean;
  canPrompt: boolean;
}> {
  try {
    if (!("permissions" in navigator) || !(navigator.permissions as any).query) {
      return { status: "unknown", granted: true, canPrompt: true };
    }
    const result = await (navigator.permissions as any).query({ name: "camera" });
    return {
      status: result.state,
      granted: result.state === "granted",
      canPrompt: result.state === "prompt",
    };
  } catch {
    return { status: "unknown", granted: true, canPrompt: true };
  }
}

async function requestCameraAccessFallback(): Promise<{
  status: string;
  granted: boolean;
  canPrompt: boolean;
}> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    stream.getTracks().forEach((track) => track.stop());
    return { status: "granted", granted: true, canPrompt: false };
  } catch (error) {
    const message = (error as Error).message.toLowerCase();
    if (message.includes("denied") || message.includes("permission")) {
      return { status: "denied", granted: false, canPrompt: false };
    }
    return { status: "unknown", granted: false, canPrompt: true };
  }
}

function cameraScore(label: string): number {
  const normalized = label.toLowerCase();
  let score = 0;
  if (normalized.includes("facetime")) score += 8;
  if (normalized.includes("continuity")) score += 7;
  if (normalized.includes("iphone")) score += 6;
  if (normalized.includes("usb")) score += 4;
  if (normalized.includes("camera")) score += 2;
  if (normalized.includes("virtual")) score -= 4;
  if (normalized.includes("obs")) score -= 4;
  return score;
}

function sortCameras(devices: CameraDeviceOption[]): CameraDeviceOption[] {
  return [...devices].sort((left, right) => {
    const scoreDelta = cameraScore(right.label) - cameraScore(left.label);
    if (scoreDelta !== 0) {
      return scoreDelta;
    }
    return left.label.localeCompare(right.label);
  });
}

function pickBestCamera(
  devices: CameraDeviceOption[],
  preferredDeviceId?: string,
  currentDeviceId?: string,
): string | undefined {
  if (preferredDeviceId && devices.some((device) => device.deviceId === preferredDeviceId)) {
    return preferredDeviceId;
  }
  if (currentDeviceId && devices.some((device) => device.deviceId === currentDeviceId)) {
    return currentDeviceId;
  }
  return sortCameras(devices)[0]?.deviceId;
}

function diagnosticsMessage(phase: string, error?: string | null): string {
  if (error) {
    return error;
  }
  switch (phase) {
    case "requesting permission":
      return "Requesting camera permission";
    case "camera retrying":
      return "Retrying camera stream";
    case "stream attached":
      return "Stream attached, waiting for frames";
    case "preview ready":
      return "Live preview ready";
    case "video stalled":
      return "Camera stream stalled";
    case "black frame detected":
      return "Camera is active but frames are effectively black";
    case "camera error":
      return "Unable to start the camera";
    default:
      return "Camera not started";
  }
}

async function waitForVideoReadiness(
  video: HTMLVideoElement,
  stream: MediaStream,
): Promise<{ ready: boolean; message: string }> {
  const track = stream.getVideoTracks()[0];
  const currentWidth = () => video.videoWidth || Number(track.getSettings().width || 0);
  const currentHeight = () => video.videoHeight || Number(track.getSettings().height || 0);
  if (
    video.readyState >= HTMLMediaElement.HAVE_METADATA &&
    currentWidth() > 0 &&
    currentHeight() > 0
  ) {
    return { ready: true, message: "Camera metadata loaded" };
  }
  try {
    await video.play();
  } catch {
    // Electron may reject autoplay before the media element settles.
  }
  return await new Promise<{ ready: boolean; message: string }>((resolve) => {
    const finish = (ready: boolean, message: string) => {
      cleanup();
      resolve({ ready, message });
    };
    const evaluate = () => {
      if (track.readyState === "live" && currentWidth() > 0 && currentHeight() > 0) {
        finish(true, "Camera stream attached");
      }
    };
    const timeout = window.setTimeout(() => {
      if (track.readyState === "live") {
        finish(false, "Camera metadata is delayed; keeping the stream attached");
        return;
      }
      finish(false, "Timed out waiting for camera metadata");
    }, 4500);
    const interval = window.setInterval(evaluate, 180);
    const handleReady = () => evaluate();
    const handleError = () => finish(false, "Camera metadata failed to load");
    function cleanup() {
      window.clearTimeout(timeout);
      window.clearInterval(interval);
      video.removeEventListener("loadedmetadata", handleReady);
      video.removeEventListener("loadeddata", handleReady);
      video.removeEventListener("canplay", handleReady);
      video.removeEventListener("resize", handleReady);
      video.removeEventListener("error", handleError);
    }
    video.addEventListener("loadedmetadata", handleReady);
    video.addEventListener("loadeddata", handleReady);
    video.addEventListener("canplay", handleReady);
    video.addEventListener("resize", handleReady);
    video.addEventListener("error", handleError);
    evaluate();
  });
}

export function useCameraStream({
  activeTab,
  settings,
  runtimeRequest,
  onStatusChange,
  onSettingsChange,
}: UseCameraStreamOptions) {
  const desktopBridgeAvailable = isDesktopBridgeAvailable();
  const [cameraDevices, setCameraDevices] = useState<CameraDeviceOption[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState("default");
  const [cameraDiagnostics, setCameraDiagnostics] =
    useState<CameraDiagnostics>(DEFAULT_CAMERA_DIAGNOSTICS);
  const [cameraAccess, setCameraAccess] = useState<CameraAccessState>({
    status: "unknown",
    granted: false,
    canPrompt: false,
  });
  const [cameraBusy, setCameraBusy] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraStreamLive, setCameraStreamLive] = useState(false);
  const [streamEpoch, setStreamEpoch] = useState(0);
  const liveVideoRef = useRef<HTMLVideoElement | null>(null);
  const livingVideoRef = useRef<HTMLVideoElement | null>(null);
  const liveCaptureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const liveDiagnosticsCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const livingCaptureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const livingDiagnosticsCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const activeCameraDeviceIdRef = useRef("default");

  function activeVideoElement() {
    return activeTab === "Living Lens" ? livingVideoRef.current : liveVideoRef.current;
  }

  function activeDiagnosticsCanvas() {
    return activeTab === "Living Lens"
      ? livingDiagnosticsCanvasRef.current
      : liveDiagnosticsCanvasRef.current;
  }

  function frameElements(mode: "live" | "living") {
    if (mode === "living") {
      return {
        video: livingVideoRef.current,
        canvas: livingCaptureCanvasRef.current,
      };
    }
    return {
      video: liveVideoRef.current,
      canvas: liveCaptureCanvasRef.current,
    };
  }

  function stopStream(stream: MediaStream | null) {
    if (!stream) {
      return;
    }
    for (const track of stream.getTracks()) {
      track.stop();
    }
  }

  async function enumerateVideoDevices(): Promise<CameraDeviceOption[]> {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return sortCameras(
      devices
        .filter((device) => device.kind === "videoinput")
        .map((device, index) => ({
          deviceId: device.deviceId,
          label: device.label || `Camera ${index + 1}`,
        })),
    );
  }

  async function persistCameraDevice(deviceId: string) {
    if (!deviceId || deviceId === "default") {
      return;
    }
    const latest = await runtimeRequest<Settings>("/v1/settings");
    if (latest.camera_device === deviceId) {
      onSettingsChange?.(latest);
      return;
    }
    latest.camera_device = deviceId;
    const saved = await runtimeRequest<Settings>("/v1/settings", "PUT", latest);
    onSettingsChange?.(saved);
  }

  async function attachStream(
    stream: MediaStream,
    devices: CameraDeviceOption[],
    requestedDeviceId?: string,
    permissionStatus = "unknown",
  ): Promise<{ ready: boolean; message: string }> {
    const video = activeVideoElement();
    if (!video) {
      throw new Error("Video element is unavailable");
    }
    stopStream(streamRef.current);
    streamRef.current = stream;
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    video.autoplay = true;
    setCameraDiagnostics((current) => ({
      ...current,
      phase: "stream attached",
      message: diagnosticsMessage("stream attached"),
      error: null,
    }));
    const readiness = await waitForVideoReadiness(video, stream);
    const track = stream.getVideoTracks()[0];
    if (track.readyState !== "live") {
      throw new Error("Camera track never became live");
    }
    setCameraStreamLive(true);
    const effectiveDeviceId = String(track.getSettings().deviceId || requestedDeviceId || "default");
    const effectiveLabel =
      devices.find((device) => device.deviceId === effectiveDeviceId)?.label ||
      track.label ||
      "Auto camera";
    activeCameraDeviceIdRef.current = effectiveDeviceId;
    setSelectedCameraId(effectiveDeviceId);
    setCameraDiagnostics({
      phase: readiness.ready ? "stream attached" : "camera retrying",
      selectedDeviceId: effectiveDeviceId,
      selectedLabel: effectiveLabel,
      permissionStatus,
      resolution: `${video.videoWidth || track.getSettings().width || 0} x ${video.videoHeight || track.getSettings().height || 0}`,
      readyState: String(video.readyState),
      trackState: track.readyState,
      trackMuted: track.muted,
      trackEnabled: track.enabled,
      lastFrameAt: new Date().toLocaleTimeString(),
      frameLuma: null,
      blackFrameDetected: false,
      message: readiness.message,
      error: readiness.ready ? null : readiness.message,
    });
    setStreamEpoch((current) => current + 1);
    await persistCameraDevice(effectiveDeviceId);
    return readiness;
  }

  async function startCamera(options: {
    preferredDeviceId?: string;
    phase: string;
    forceAuto?: boolean;
  }) {
    if (!navigator.mediaDevices?.getUserMedia) {
      const message = "This environment does not support camera capture";
      setCameraStreamLive(false);
      setCameraDiagnostics({
        ...DEFAULT_CAMERA_DIAGNOSTICS,
        phase: "camera error",
        message,
        error: message,
      });
      onStatusChange?.(message);
      return;
    }
    setCameraBusy(true);
    setCameraReady(false);
    setCameraStreamLive(false);
    const bridge = getDesktopBridge();
    let access = bridge.getCameraAccess
      ? await bridge.getCameraAccess()
      : await getCameraAccessStateFallback();
    if (!access.granted && access.canPrompt) {
      access = bridge.requestCameraAccess
        ? await bridge.requestCameraAccess()
        : await requestCameraAccessFallback();
    }
    setCameraAccess(access);
    if (!access.granted) {
      setCameraStreamLive(false);
      const message =
        access.status === "denied"
          ? "Camera permission denied in macOS Privacy & Security"
          : access.status === "restricted"
            ? "Camera access is restricted by macOS"
            : access.status === "not-determined"
              ? "macOS did not show the camera prompt"
              : "Camera access is not granted";
      setCameraDiagnostics({
        ...DEFAULT_CAMERA_DIAGNOSTICS,
        phase: "camera error",
        permissionStatus: access.status,
        message,
        error: message,
      });
      onStatusChange?.(message);
      setCameraBusy(false);
      return;
    }
    setCameraDiagnostics((current) => ({
      ...current,
      phase: options.phase,
      permissionStatus: access.status,
      message: diagnosticsMessage(options.phase),
      error: null,
      blackFrameDetected: false,
    }));
    let provisionalStream: MediaStream | null = null;
    let selectedStream: MediaStream | null = null;
    try {
      provisionalStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      const devices = await enumerateVideoDevices();
      setCameraDevices(devices);
      const currentDeviceId = String(provisionalStream.getVideoTracks()[0]?.getSettings().deviceId || "");
      const desiredDeviceId = options.forceAuto
        ? pickBestCamera(devices, undefined, currentDeviceId)
        : pickBestCamera(devices, options.preferredDeviceId, currentDeviceId);
      if (desiredDeviceId && desiredDeviceId !== currentDeviceId) {
        try {
          selectedStream = await navigator.mediaDevices.getUserMedia({
            video: {
              deviceId: { exact: desiredDeviceId },
              width: { ideal: 1280 },
              height: { ideal: 720 },
            },
            audio: false,
          });
        } catch {
          setCameraDiagnostics((current) => ({
            ...current,
            phase: "camera retrying",
            message: diagnosticsMessage("camera retrying"),
          }));
          selectedStream = provisionalStream;
          provisionalStream = null;
        }
      } else {
        selectedStream = provisionalStream;
        provisionalStream = null;
      }
      if (!selectedStream) {
        throw new Error("Unable to attach a camera stream");
      }
      const readiness = await attachStream(selectedStream, devices, desiredDeviceId, access.status);
      stopStream(provisionalStream);
      onStatusChange?.(readiness.ready ? "Camera preview ready" : readiness.message);
    } catch (error) {
      stopStream(selectedStream);
      stopStream(provisionalStream);
      setCameraStreamLive(false);
      const message = (error as Error).message;
      setCameraDiagnostics((current) => ({
        ...current,
        phase: "camera error",
        permissionStatus: access.status,
        message: diagnosticsMessage("camera error", message),
        error: message,
      }));
      onStatusChange?.(`Camera unavailable: ${message}`);
    } finally {
      setCameraBusy(false);
    }
  }

  async function retryCamera(forceAuto = false) {
    const preferredDeviceId = forceAuto
      ? undefined
      : selectedCameraId !== "default"
        ? selectedCameraId
        : settings?.camera_device !== "default"
          ? settings?.camera_device
          : undefined;
    await startCamera({
      preferredDeviceId,
      phase: forceAuto ? "requesting permission" : "camera retrying",
      forceAuto,
    });
  }

  async function requestCameraPermission() {
    setCameraBusy(true);
    try {
      const bridge = getDesktopBridge();
      const access = bridge.requestCameraAccess
        ? await bridge.requestCameraAccess()
        : await requestCameraAccessFallback();
      setCameraAccess(access);
      if (access.granted) {
        onStatusChange?.("Camera permission granted");
        await retryCamera(true);
        return;
      }
      const message =
        access.status === "denied"
          ? "Camera permission denied in macOS Privacy & Security"
          : access.status === "restricted"
            ? "Camera access is restricted by macOS"
            : access.status === "not-determined"
              ? "macOS did not show the camera prompt"
              : "Camera access is not granted";
      setCameraDiagnostics((current) => ({
        ...current,
        phase: "camera error",
        permissionStatus: access.status,
        message,
        error: message,
      }));
      setCameraStreamLive(false);
      onStatusChange?.(message);
    } finally {
      setCameraBusy(false);
    }
  }

  async function switchCamera() {
    if (cameraDevices.length < 2) {
      onStatusChange?.("No alternate camera is available");
      return;
    }
    const currentIndex = cameraDevices.findIndex((device) => device.deviceId === selectedCameraId);
    const nextDevice = cameraDevices[(currentIndex + 1 + cameraDevices.length) % cameraDevices.length];
    await startCamera({ preferredDeviceId: nextDevice.deviceId, phase: "camera retrying" });
  }

  function currentFrameBase64(mode: "live" | "living"): string | null {
    const { video, canvas } = frameElements(mode);
    if (!video || !canvas) {
      return null;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx || !video.videoWidth || !video.videoHeight) {
      return null;
    }
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/png").split(",")[1];
  }

  useEffect(() => {
    if (!settings) {
      return;
    }
    const desiredDeviceId =
      settings.camera_device && settings.camera_device !== "default"
        ? settings.camera_device
        : undefined;
    if (streamRef.current && (!desiredDeviceId || activeCameraDeviceIdRef.current === desiredDeviceId)) {
      setSelectedCameraId(desiredDeviceId || activeCameraDeviceIdRef.current || "default");
      return;
    }
    startCamera({ preferredDeviceId: desiredDeviceId, phase: "requesting permission" }).catch(() => undefined);
  }, [settings?.camera_device]);

  useEffect(() => {
    return () => stopStream(streamRef.current);
  }, []);

  useEffect(() => {
    const stream = streamRef.current;
    const video = activeVideoElement();
    if (!stream || !video) {
      return;
    }
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    video.autoplay = true;
    video.play().catch(() => undefined);
  }, [activeTab, streamEpoch]);

  useEffect(() => {
    const stream = streamRef.current;
    const video = activeVideoElement();
    const canvas = activeDiagnosticsCanvas();
    if (!stream || !video || !canvas) {
      return;
    }
    const track = stream.getVideoTracks()[0];
    let lastCurrentTime = video.currentTime;
    let lastFrameMs = Date.now();
    let darkFrameCount = 0;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    const interval = window.setInterval(() => {
      if (!context) {
        return;
      }
      const width = video.videoWidth || Number(track.getSettings().width || 0);
      const height = video.videoHeight || Number(track.getSettings().height || 0);
      const now = Date.now();
      if (video.currentTime !== lastCurrentTime) {
        lastCurrentTime = video.currentTime;
        lastFrameMs = now;
      }
      let frameLuma: number | null = null;
      if (width > 0 && height > 0 && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        canvas.width = 64;
        canvas.height = 36;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
        let totalLuma = 0;
        for (let index = 0; index < pixels.length; index += 4) {
          totalLuma +=
            pixels[index] * 0.299 +
            pixels[index + 1] * 0.587 +
            pixels[index + 2] * 0.114;
        }
        frameLuma = totalLuma / (pixels.length / 4);
      }
      if (frameLuma != null && frameLuma < 6) {
        darkFrameCount += 1;
      } else {
        darkFrameCount = 0;
      }
      const blackFrameDetected = darkFrameCount >= 3 && width > 0 && height > 0;
      const stalled = now - lastFrameMs > 2500;
      setCameraStreamLive(track.readyState === "live");
      const phase =
        track.readyState !== "live"
          ? "video stalled"
          : blackFrameDetected
            ? "black frame detected"
            : stalled
              ? "video stalled"
              : "preview ready";
      const resolution = width > 0 && height > 0 ? `${width} x ${height}` : "0 x 0";
      const lastFrameAt = width > 0 && height > 0 ? new Date(lastFrameMs).toLocaleTimeString() : null;
      setCameraReady(
        track.readyState === "live" && !blackFrameDetected && !stalled && width > 0 && height > 0,
      );
      setCameraDiagnostics((current) => ({
        ...current,
        phase,
        resolution,
        readyState: String(video.readyState),
        trackState: track.readyState,
        trackMuted: track.muted,
        trackEnabled: track.enabled,
        lastFrameAt,
        frameLuma,
        blackFrameDetected,
        message: diagnosticsMessage(phase, current.error),
      }));
    }, 700);
    return () => window.clearInterval(interval);
  }, [streamEpoch, activeTab]);

  return {
    desktopBridgeAvailable,
    cameraDevices,
    selectedCameraId,
    setSelectedCameraId,
    cameraDiagnostics,
    cameraAccess,
    cameraBusy,
    cameraReady,
    cameraStreamLive,
    streamEpoch,
    liveVideoRef,
    livingVideoRef,
    liveCaptureCanvasRef,
    liveDiagnosticsCanvasRef,
    livingCaptureCanvasRef,
    livingDiagnosticsCanvasRef,
    startCamera,
    retryCamera,
    requestCameraPermission,
    switchCamera,
    stopStream,
    currentFrameBase64,
  };
}
