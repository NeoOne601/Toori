"""
Audio JEPA Stub Encoder.

Numpy-based Mel-spectrogram feature encoder. Extracts audio embeddings
from raw waveforms using numpy FFT. No torch dependency.

This is a lightweight stub for Sprint 6 — produces 384-dim audio embeddings
that can be stored alongside visual JEPA ticks for future cross-modal alignment.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from cloud.runtime.observability import get_logger

log = get_logger("audio_encoder")

# Audio constants
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 64
TARGET_DIM = 384
EPSILON = 1e-8


@dataclass(slots=True)
class AudioEmbedding:
    """Result of audio encoding."""
    embedding: np.ndarray       # (384,) float32 L2-normalized
    energy: float               # RMS energy of the waveform
    duration_seconds: float     # Duration of input audio
    is_silent: bool             # True if energy below threshold


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Create a simple Mel filterbank matrix with numpy."""
    freq = np.linspace(0, sr / 2, n_fft // 2 + 1)

    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = _hz_to_mel(0.0)
    mel_max = _hz_to_mel(sr / 2)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])

    bins = np.round(hz_points / (sr / n_fft)).astype(int)
    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left = bins[i]
        center = bins[i + 1]
        right = bins[i + 2]
        for j in range(left, center):
            if center > left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filterbank[i, j] = (right - j) / (right - center)
    return filterbank


def _mel_spectrogram(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compute a Mel spectrogram using numpy FFT."""
    waveform = waveform.astype(np.float32)
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=-1)  # mono

    n_frames = 1 + (len(waveform) - N_FFT) // HOP_LENGTH
    if n_frames <= 0:
        return np.zeros((N_MELS, 1), dtype=np.float32)

    window = np.hanning(N_FFT).astype(np.float32)
    frames = np.stack([
        waveform[i * HOP_LENGTH : i * HOP_LENGTH + N_FFT] * window
        for i in range(n_frames)
    ])
    spectrogram = np.abs(np.fft.rfft(frames, n=N_FFT, axis=-1)) ** 2
    filterbank = _mel_filterbank(sr, N_FFT, N_MELS)
    mel_spec = np.dot(spectrogram, filterbank.T).T  # (N_MELS, n_frames)
    mel_spec = np.log(mel_spec + EPSILON)
    return mel_spec.astype(np.float32)


class AudioEncoder:
    """
    Lightweight audio feature encoder using Mel spectrograms.

    No torch required. Returns 384-dim embeddings compatible with
    the JEPA metric space for future cross-modal alignment.
    """

    SILENCE_THRESHOLD = 0.01

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self._projection = rng.standard_normal(
            (N_MELS, TARGET_DIM)
        ).astype(np.float32) / np.sqrt(N_MELS)

    def encode(
        self,
        waveform: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> AudioEmbedding:
        """
        Encode audio waveform to a 384-dim embedding.

        Args:
            waveform: 1D numpy array of audio samples
            sample_rate: sample rate of the audio

        Returns:
            AudioEmbedding with 384-dim L2-normalized embedding
        """
        waveform = np.asarray(waveform, dtype=np.float32)
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=-1)

        energy = float(np.sqrt(np.mean(waveform ** 2)))
        duration = len(waveform) / max(sample_rate, 1)
        is_silent = energy < self.SILENCE_THRESHOLD

        if is_silent or len(waveform) < N_FFT:
            return AudioEmbedding(
                embedding=np.zeros(TARGET_DIM, dtype=np.float32),
                energy=energy,
                duration_seconds=duration,
                is_silent=True,
            )

        mel = _mel_spectrogram(waveform, sample_rate)
        # Pool across time: mean over frames → (N_MELS,)
        mel_pooled = mel.mean(axis=1).astype(np.float32)  # (N_MELS,)
        # Project to 384-dim
        embedding = mel_pooled @ self._projection  # (384,)
        # L2 normalize
        norm = np.linalg.norm(embedding) + EPSILON
        embedding = (embedding / norm).astype(np.float32)

        return AudioEmbedding(
            embedding=embedding,
            energy=energy,
            duration_seconds=duration,
            is_silent=False,
        )

    def encode_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = SAMPLE_RATE,
        dtype: str = "int16",
    ) -> AudioEmbedding:
        """Convenience method to encode raw audio bytes."""
        dt = np.dtype(dtype)
        waveform = np.frombuffer(audio_bytes, dtype=dt).astype(np.float32)
        if dtype == "int16":
            waveform /= 32768.0
        return self.encode(waveform, sample_rate)
