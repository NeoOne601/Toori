"""Sprint 6: Audio JEPA Tests.

Tests the numpy-based audio feature encoder used as a stub for
Audio-JEPA integration. Modality alignment is the next milestone.
"""
from __future__ import annotations

import numpy as np
import pytest

from cloud.perception.audio_encoder import AudioEncoder, AudioEmbedding, TARGET_DIM


def _generate_sine_wave(freq: float, duration: float, sample_rate: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_audio_embedding_dataclass():
    emb = AudioEmbedding(
        embedding=np.zeros(TARGET_DIM, dtype=np.float32),
        energy=0.5,
        duration_seconds=1.0,
        is_silent=False,
    )
    assert len(emb.embedding) == TARGET_DIM
    assert emb.energy == 0.5
    assert emb.duration_seconds == 1.0
    assert not emb.is_silent


def test_audio_encoder_initializes_with_target_dim():
    encoder = AudioEncoder()
    # Ensure it can yield a vector of the correct dim
    assert len(encoder.encode(np.zeros(16000, dtype=np.float32)).embedding) == TARGET_DIM


def test_encode_silent_audio():
    encoder = AudioEncoder()
    silence = np.zeros(16000, dtype=np.float32)
    result = encoder.encode(silence)

    assert result.is_silent
    assert result.energy < encoder.SILENCE_THRESHOLD
    assert result.duration_seconds == 1.0
    assert len(result.embedding) == TARGET_DIM
    # Should be L2 normalized, even for silence
    norm = np.linalg.norm(result.embedding)
    assert norm == pytest.approx(1.0, rel=1e-3)


def test_encode_sine_wave():
    encoder = AudioEncoder()
    wave = _generate_sine_wave(440.0, 1.0)  # 1 second of 440Hz A
    result = encoder.encode(wave)

    assert not result.is_silent
    assert result.energy > encoder.SILENCE_THRESHOLD
    assert result.duration_seconds == 1.0
    assert len(result.embedding) == TARGET_DIM
    # Check L2 normalization
    norm = np.linalg.norm(result.embedding)
    assert norm == pytest.approx(1.0, rel=1e-3)


def test_encode_short_audio_is_silent_by_fallback():
    encoder = AudioEncoder()
    short = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    result = encoder.encode(short)

    # Too short -> marked silent, but still returns a unit vector
    assert result.is_silent
    norm = np.linalg.norm(result.embedding)
    assert norm == pytest.approx(1.0, rel=1e-3)


def test_encode_stereo_audio_averages_to_mono():
    encoder = AudioEncoder()
    # 2 channels, 1 second
    left = _generate_sine_wave(440.0, 1.0)
    right = _generate_sine_wave(880.0, 1.0)
    stereo = np.stack([left, right], axis=-1)

    result = encoder.encode(stereo)

    assert not result.is_silent
    assert len(result.embedding) == TARGET_DIM
    norm = np.linalg.norm(result.embedding)
    assert norm == pytest.approx(1.0, rel=1e-3)


def test_encode_bytes_int16():
    encoder = AudioEncoder()
    # 1 second of 440Hz A, scaled to int16
    wave = _generate_sine_wave(440.0, 1.0) * 32767.0
    wave_int16 = wave.astype(np.int16)
    
    result = encoder.encode_bytes(wave_int16.tobytes(), dtype="int16")
    
    assert not result.is_silent
    assert result.duration_seconds == 1.0
    norm = np.linalg.norm(result.embedding)
    assert norm == pytest.approx(1.0, rel=1e-3)

def test_encode_bytes_float32():
    encoder = AudioEncoder()
    wave = _generate_sine_wave(440.0, 1.0)
    wave_float32 = wave.astype(np.float32)
    
    result = encoder.encode_bytes(wave_float32.tobytes(), dtype="float32")
    
    assert not result.is_silent
    assert result.duration_seconds == 1.0
    norm = np.linalg.norm(result.embedding)
    assert norm == pytest.approx(1.0, rel=1e-3)
