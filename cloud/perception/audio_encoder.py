"""
Audio-JEPA Phase 1 Encoder.

Numpy + scipy Mel-spectrogram encoder. Decodes audio from files (video/audio)
using PyAV (av>=12). Returns 384-dim unit embeddings that land in the same
metric space as DINOv2 visual embeddings for future cross-modal alignment.

SEED invariant: SEED = 20260327 — NEVER change this.
Changing the seed invalidates all stored audio embeddings in Smriti.
"""
from __future__ import annotations

import importlib
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED: int = 20260327          # NEVER change — invalidates all stored embeddings
SAMPLE_RATE: int = 16000
N_MELS: int = 64
N_FFT: int = 512
HOP_LENGTH: int = 160         # 10 ms at 16 kHz
FMIN: float = 0.0
FMAX: float = 8000.0
MAX_DURATION_S: float = 30.0
TARGET_DIM: int = 384
EPSILON: float = 1e-8

_log = logging.getLogger("smriti.audio_encoder")


@dataclass(slots=True)
class AudioEmbedding:
    """Result of audio encoding — compatibility type matching the Sprint 6 stub API."""
    embedding: np.ndarray       # (384,) float32 L2-normalized
    energy: float               # RMS energy of the waveform
    duration_seconds: float     # Duration of input audio
    is_silent: bool             # True if amplitude too low for reliable encoding




# ---------------------------------------------------------------------------
# Mel filterbank (pure numpy)
# ---------------------------------------------------------------------------

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(sr: int, n_fft: int, n_mels: int,
                           fmin: float, fmax: float) -> np.ndarray:
    """Return (n_mels, n_fft//2+1) triangular filterbank matrix."""
    freq_bins = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        for j in range(l, c):
            if c > l:
                filterbank[i, j] = float(j - l) / (c - l)
        for j in range(c, r):
            if r > c:
                filterbank[i, j] = float(r - j) / (r - c)
    return filterbank


# Build and cache module-level filterbank + projection
_FILTERBANK: Optional[np.ndarray] = None
_PROJECTION: Optional[np.ndarray] = None


def _get_filterbank() -> np.ndarray:
    global _FILTERBANK
    if _FILTERBANK is None:
        _FILTERBANK = _build_mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS, FMIN, FMAX)
    return _FILTERBANK


def _get_projection(time_frames: int) -> np.ndarray:
    """Return (n_mels * time_frames, TARGET_DIM) random projection, seeded."""
    global _PROJECTION
    # Projection depends on n_mels; cache per time_frames would complicate things.
    # Instead use a fixed large matrix and slice/repeat — simpler: we pool time
    # before projecting so shape is always (N_MELS, TARGET_DIM).
    if _PROJECTION is None:
        rng = np.random.RandomState(SEED)
        _PROJECTION = rng.randn(N_MELS, TARGET_DIM).astype(np.float32)
    return _PROJECTION


# ---------------------------------------------------------------------------
# Core DSP
# ---------------------------------------------------------------------------

def _mel_spectrogram(waveform: np.ndarray) -> np.ndarray:
    """Compute log-Mel spectrogram → (N_MELS, n_frames). Pure numpy."""
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)

    n_samples = len(waveform)
    max_samples = int(MAX_DURATION_S * SAMPLE_RATE)
    if n_samples > max_samples:
        waveform = waveform[:max_samples]
        n_samples = max_samples

    if n_samples < N_FFT:
        # Pad so we have at least one frame
        waveform = np.pad(waveform, (0, N_FFT - n_samples))
        n_samples = N_FFT

    n_frames = 1 + (n_samples - N_FFT) // HOP_LENGTH
    if n_frames <= 0:
        return np.zeros((N_MELS, 1), dtype=np.float32)

    window = np.hanning(N_FFT).astype(np.float32)

    # Build frame matrix
    indices = np.arange(N_FFT)[None, :] + np.arange(n_frames)[:, None] * HOP_LENGTH
    frames = waveform[indices] * window  # (n_frames, N_FFT)

    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=-1)) ** 2  # (n_frames, N_FFT//2+1)
    filterbank = _get_filterbank()
    mel = filterbank @ power.T  # (N_MELS, n_frames)
    return np.log(mel + EPSILON).astype(np.float32)


def _resample_simple(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Naive linear resampling via numpy interpolation."""
    if orig_sr == target_sr:
        return audio
    orig_len = len(audio)
    target_len = int(orig_len * target_sr / orig_sr)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    x_orig = np.linspace(0, orig_len - 1, orig_len)
    x_new = np.linspace(0, orig_len - 1, target_len)
    return np.interp(x_new, x_orig, audio).astype(np.float32)


def _encode_array(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Core encode: float32 PCM → 384-dim unit embedding."""
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=-1)

    # 1. Resample to SAMPLE_RATE if needed
    if sample_rate != SAMPLE_RATE:
        pcm = _resample_simple(pcm, sample_rate, SAMPLE_RATE)

    # 2. Truncate to MAX_DURATION_S
    max_samples = int(MAX_DURATION_S * SAMPLE_RATE)
    if len(pcm) > max_samples:
        pcm = pcm[:max_samples]

    # 3. Mel spectrogram  (N_MELS, n_frames)
    mel = _mel_spectrogram(pcm)

    # 4. Time-pool → (N_MELS,)
    mel_pooled = mel.mean(axis=1).astype(np.float32)

    # 5. Random projection (N_MELS,) × (N_MELS, 384) → (384,)
    proj = _get_projection(mel.shape[1])
    embedding = mel_pooled @ proj  # (384,)

    # 6. Tanh activation
    embedding = np.tanh(embedding).astype(np.float32)

    # 7. L2 normalize to unit norm
    norm = float(np.linalg.norm(embedding)) + EPSILON
    return (embedding / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class AudioEncoder:
    """
    Lightweight Mel-spectrogram audio encoder.

    numpy + PyAV only. No torch dependency.
    Output: 384-dim unit-norm float32 embedding.

    SEED = 20260327 — fixed. Changing this invalidates stored embeddings.
    """

    # Silence detection threshold (RMS)
    SILENCE_THRESHOLD: float = 0.001

    def encode_array(self, pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """
        Encode a float32 PCM waveform to a 384-dim unit embedding.

        Args:
            pcm:         1-D (or 2-D mono/stereo) float32 numpy array.
            sample_rate: Original sample rate; resampled to 16000 internally.

        Returns:
            np.ndarray shape (384,), float32, L2-normalized.
        """
        try:
            return _encode_array(pcm, sample_rate)
        except Exception as exc:
            _log.warning("AudioEncoder.encode_array failed: %s", exc)
            raise ValueError(f"AudioEncoder failed: {exc}") from exc

    def encode(self, waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> AudioEmbedding:
        """Compatibility endpoint for Sprint 6 tests."""
        arr = self.encode_array(waveform, sample_rate)
        wave = np.asarray(waveform, dtype=np.float32)
        if wave.ndim > 1:
            wave = wave.mean(axis=-1)
        energy = float(np.sqrt(np.mean(wave ** 2))) if len(wave) > 0 else 0.0
        dur = len(wave) / sample_rate
        return AudioEmbedding(
            embedding=arr,
            energy=energy,
            duration_seconds=dur,
            is_silent=energy < self.SILENCE_THRESHOLD or len(wave) < N_FFT
        )

    def encode_bytes(self, pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE, dtype: str = "") -> np.ndarray | AudioEmbedding:
        """
        Encode raw audio bytes (PCM int16 or float32) to a 384-dim unit embedding.
        If dtype is provided (legacy API), returns an AudioEmbedding.
        """
        try:
            n_bytes = len(pcm_bytes)
            # Heuristic: try int16 first (2 bytes/sample)
            if n_bytes % 2 == 0 and dtype != "float32":
                waveform = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                waveform /= 32768.0
            else:
                # Assume float32 (4 bytes/sample)
                n_pad = (4 - n_bytes % 4) % 4
                waveform = np.frombuffer(pcm_bytes + b"\x00" * n_pad, dtype=np.float32)
            
            if dtype:
                return self.encode(waveform, sample_rate)
            return _encode_array(waveform, sample_rate)
        except Exception as exc:
            _log.warning("AudioEncoder.encode_bytes failed: %s", exc)
            raise ValueError(f"AudioEncoder failed: {exc}") from exc

    def encode_file(self, path: str) -> np.ndarray:
        """
        Decode audio from a file path using PyAV and return a 384-dim unit embedding.

        Extracts first MAX_DURATION_S (30s) of audio. Resamples to 16000 Hz mono.
        Works on any format supported by PyAV: mp4, mov, avi, mkv, mp3, wav, m4a, etc.

        Args:
            path: Absolute or relative path to audio/video file.

        Returns:
            np.ndarray shape (384,), float32, L2-normalized.

        Raises:
            ImportError: If PyAV is not installed.
            ValueError:  If decoding fails for any other reason.
        """
        try:
            av = importlib.import_module("av")
        except ImportError as exc:
            raise ImportError(
                "PyAV required for audio encoding. "
                "Run: pip install av>=12 --break-system-packages"
            ) from exc

        max_samples = int(MAX_DURATION_S * SAMPLE_RATE)
        chunks: list[np.ndarray] = []
        collected = 0

        try:
            container = av.open(str(path))
            try:
                audio_stream = next(
                    (s for s in container.streams if s.type == "audio"), None
                )
                if audio_stream is None:
                    _log.debug("AudioEncoder: no audio stream in %s — returning zero embedding", path)
                    return np.zeros(TARGET_DIM, dtype=np.float32)

                orig_sr: int = audio_stream.codec_context.sample_rate or SAMPLE_RATE

                for frame in container.decode(audio_stream):
                    # Convert frame to float32 ndarray (channels, samples)
                    arr = frame.to_ndarray()  # shape: (n_channels, n_samples) or (n_samples,)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    # Mono mix
                    mono = arr.mean(axis=0).astype(np.float32)
                    # Normalize int16-range if needed
                    if arr.dtype.kind in ("i", "u"):
                        max_val = float(np.iinfo(arr.dtype).max) if arr.dtype.kind == "i" else float(np.iinfo(arr.dtype).max)
                        mono = mono / max(max_val, 1.0)

                    # Resample frame to SAMPLE_RATE if needed
                    if orig_sr != SAMPLE_RATE:
                        mono = _resample_simple(mono, orig_sr, SAMPLE_RATE)

                    chunks.append(mono)
                    collected += len(mono)
                    if collected >= max_samples:
                        break
            finally:
                container.close()
        except Exception as exc:
            _log.warning("AudioEncoder.encode_file failed for %s: %s", path, exc)
            raise ValueError(f"AudioEncoder failed: {exc}") from exc

        if not chunks:
            return np.zeros(TARGET_DIM, dtype=np.float32)

        pcm = np.concatenate(chunks, axis=0)[:max_samples]
        return _encode_array(pcm, SAMPLE_RATE)
