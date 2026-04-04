import { useState, useRef, useCallback } from "react";
import { BROWSER_RUNTIME_URL as API_BASE_URL } from "./useRuntimeBridge";
import type { AudioQueryRequest, AudioQueryResponse, AudioQueryResult } from "../types";

export function useAudioQuery() {
  const [recording, setRecording] = useState(false);
  const [audioResults, setAudioResults] = useState<AudioQueryResult[]>([]);
  const [audioLatencyMs, setAudioLatencyMs] = useState(0);
  const [audioIndexSize, setAudioIndexSize] = useState(0);
  const [audioError, setAudioError] = useState<string | null>(null);
  const [isQuerying, setIsQuerying] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const autoStopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const submitAudio = useCallback(async (audio_base64: string) => {
    setIsQuerying(true);
    setAudioError(null);
    try {
      const body: AudioQueryRequest = {
        audio_base64,
        sample_rate: 16000,
        top_k: 10,
      };
      const response = await fetch(`${API_BASE_URL}/v1/audio/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        throw new Error(`Audio query failed: ${response.status} ${response.statusText}`);
      }
      const data: AudioQueryResponse = await response.json();
      setAudioResults(data.results);
      setAudioLatencyMs(data.latency_ms);
      setAudioIndexSize(data.index_size);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Audio query failed";
      setAudioError(msg);
      setAudioResults([]);
    } finally {
      setIsQuerying(false);
    }
  }, []);

  const startRecording = useCallback(async () => {
    setAudioError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true },
      });

      // Pick the best available MIME type
      const mimeType = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg"].find(
        (m) => MediaRecorder.isTypeSupported(m)
      ) || "";

      const mr = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = mr;
      chunksRef.current = [];

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setRecording(false);
        if (chunksRef.current.length === 0) return;
        const blob = new Blob(chunksRef.current, { type: mimeType || "audio/webm" });
        const arrayBuffer = await blob.arrayBuffer();
        const bytes = new Uint8Array(arrayBuffer);
        // Base64 encode
        let binary = "";
        bytes.forEach((b) => (binary += String.fromCharCode(b)));
        const b64 = btoa(binary);
        await submitAudio(b64);
      };

      mr.start(100); // collect chunks every 100ms
      setRecording(true);

      // Auto-stop after 5 seconds
      autoStopTimerRef.current = setTimeout(() => {
        if (mediaRecorderRef.current?.state === "recording") {
          mediaRecorderRef.current.stop();
        }
      }, 5000);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Microphone access denied";
      setAudioError(msg);
    }
  }, [submitAudio]);

  const stopRecording = useCallback(() => {
    if (autoStopTimerRef.current) clearTimeout(autoStopTimerRef.current);
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  }, []);

  const clearAudioResults = useCallback(() => {
    setAudioResults([]);
    setAudioLatencyMs(0);
    setAudioIndexSize(0);
    setAudioError(null);
  }, []);

  return {
    recording,
    audioResults,
    audioLatencyMs,
    audioIndexSize,
    audioError,
    isQuerying,
    startRecording,
    stopRecording,
    clearAudioResults,
  };
}
