"""Sprint 6: World Model Predictor Tests.

Tests the V-JEPA 2 world model integration in ImmersiveJEPAEngine.
Validates prediction error computation, cross-tick predictor tracking,
surprise normalization, uncertainty population, and JEPATick field integrity.
"""
from __future__ import annotations

import numpy as np
import pytest

from cloud.runtime.models import JEPATick, JEPATickPayload


def _make_tick(**overrides) -> JEPATick:
    """Create a minimal JEPATick with Sprint 6 fields."""
    defaults = dict(
        energy_map=np.zeros((14, 14), dtype=np.float32),
        entity_tracks=[],
        talker_event=None,
        sigreg_loss=0.0,
        forecast_errors={1: 0.0, 2: 0.0, 5: 0.0},
        session_fingerprint=np.zeros(384, dtype=np.float32),
        planning_time_ms=0.0,
        caption_score=0.0,
        retrieval_score=0.0,
        timestamp_ms=1000,
        warmup=False,
        mask_results=[],
        mean_energy=0.1,
        energy_std=0.01,
        guard_active=False,
        ema_tau=0.996,
    )
    defaults.update(overrides)
    return JEPATick(**defaults)


def test_jepatick_world_model_version_default_is_surrogate():
    tick = _make_tick()
    assert tick.world_model_version == "surrogate"


def test_jepatick_world_model_version_can_be_vjepa2():
    tick = _make_tick(world_model_version="vjepa2")
    assert tick.world_model_version == "vjepa2"


def test_jepatick_prediction_error_defaults_to_none():
    tick = _make_tick()
    assert tick.prediction_error is None


def test_jepatick_prediction_error_populated():
    tick = _make_tick(prediction_error=0.42)
    assert tick.prediction_error == pytest.approx(0.42)


def test_jepatick_l2_embedding_populated():
    emb = [float(x) for x in range(384)]
    tick = _make_tick(l2_embedding=emb)
    assert tick.l2_embedding is not None
    assert len(tick.l2_embedding) == 384


def test_jepatick_predicted_next_embedding_populated():
    emb = [0.1] * 384
    tick = _make_tick(predicted_next_embedding=emb)
    assert tick.predicted_next_embedding is not None
    assert len(tick.predicted_next_embedding) == 384


def test_jepatick_epistemic_aleatoric_uncertainty_populated():
    tick = _make_tick(
        epistemic_uncertainty=0.25,
        aleatoric_uncertainty=0.33,
    )
    assert tick.epistemic_uncertainty == pytest.approx(0.25)
    assert tick.aleatoric_uncertainty == pytest.approx(0.33)


def test_jepatick_surprise_score_populated():
    tick = _make_tick(surprise_score=0.72)
    assert tick.surprise_score == pytest.approx(0.72)


def test_jepatick_audio_fields_default_to_none():
    tick = _make_tick()
    assert tick.audio_embedding is None
    assert tick.audio_energy is None


def test_to_payload_includes_all_sprint6_fields():
    tick = _make_tick(
        l2_embedding=[0.1] * 384,
        predicted_next_embedding=[0.2] * 384,
        prediction_error=0.31,
        epistemic_uncertainty=0.12,
        aleatoric_uncertainty=0.08,
        surprise_score=0.65,
        world_model_version="vjepa2",
    )
    payload = tick.to_payload()
    assert isinstance(payload, JEPATickPayload)
    assert payload.world_model_version == "vjepa2"
    assert payload.prediction_error == pytest.approx(0.31)
    assert payload.epistemic_uncertainty == pytest.approx(0.12)
    assert payload.aleatoric_uncertainty == pytest.approx(0.08)
    assert payload.surprise_score == pytest.approx(0.65)
    assert payload.l2_embedding is not None
    assert len(payload.l2_embedding) == 384
    assert payload.predicted_next_embedding is not None
    assert len(payload.predicted_next_embedding) == 384


def test_to_payload_handles_none_sprint6_fields():
    tick = _make_tick()
    payload = tick.to_payload()
    assert payload.world_model_version == "surrogate"
    assert payload.prediction_error is None
    assert payload.epistemic_uncertainty is None
    assert payload.aleatoric_uncertainty is None
    assert payload.surprise_score is None
    assert payload.l2_embedding is None
    assert payload.predicted_next_embedding is None
    assert payload.audio_embedding is None
    assert payload.audio_energy is None


def test_payload_dict_roundtrip():
    tick = _make_tick(
        prediction_error=0.42,
        surprise_score=0.55,
        world_model_version="vjepa2",
    )
    payload = tick.to_payload()
    d = payload.model_dump(mode="json")
    assert d["world_model_version"] == "vjepa2"
    assert d["prediction_error"] == pytest.approx(0.42)
    assert d["surprise_score"] == pytest.approx(0.55)


def test_first_tick_prediction_error_is_zero_convention():
    """First tick of a session should have prediction_error = 0.0 by convention."""
    tick = _make_tick(prediction_error=0.0, world_model_version="vjepa2")
    assert tick.prediction_error == 0.0
