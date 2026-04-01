"""
cloud/jepa_service/tests/test_world_model_predictor.py
V-JEPA 2 world model tests — M1 8GB safe.
All tests marked @pytest.mark.vjepa2 — excluded from default suite.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.vjepa2


def test_encoder_type_is_vjepa2(vjepa2_encoder):
    assert vjepa2_encoder.encoder_type == "vjepa2"


def test_encoder_runs_on_cpu_in_tests(vjepa2_encoder):
    assert str(vjepa2_encoder.device) == "cpu"


def test_n_frames_is_safe_for_m1(vjepa2_encoder):
    assert vjepa2_encoder.n_frames <= 8


def test_encode_returns_correct_shapes(vjepa2_encoder, random_frame):
    enc, pred = vjepa2_encoder.encode(random_frame)
    assert enc.shape  == (384,)
    assert pred.shape == (384,)


def test_encode_returns_float32(vjepa2_encoder, random_frame):
    enc, pred = vjepa2_encoder.encode(random_frame)
    assert enc.dtype  == np.float32
    assert pred.dtype == np.float32


def test_embeddings_are_normalized(vjepa2_encoder, random_frame):
    enc, pred = vjepa2_encoder.encode(random_frame)
    assert abs(np.linalg.norm(enc)  - 1.0) < 0.01
    assert abs(np.linalg.norm(pred) - 1.0) < 0.01


def test_encoder_and_predictor_are_distinct(vjepa2_encoder, random_frame):
    enc, pred = vjepa2_encoder.encode(random_frame)
    cosine = float(np.dot(enc, pred))
    assert cosine < 0.999, (
        f"Encoder and predictor identical (cosine={cosine:.6f}). "
        "Predictor must differ — it predicts the future."
    )


def test_first_tick_prediction_error_is_zero():
    from cloud.jepa_service.engine import ImmersiveJEPAEngine
    engine = ImmersiveJEPAEngine(device="cpu")
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    tick = engine.tick(frame, session_id="test_first", observation_id="obs_0")
    assert tick.prediction_error == 0.0


def test_second_tick_prediction_error_nonzero():
    from cloud.jepa_service.engine import ImmersiveJEPAEngine
    rng = np.random.default_rng(1)
    engine = ImmersiveJEPAEngine(device="cpu")
    session = "test_second"
    engine.tick(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                session_id=session, observation_id="obs_0")
    tick2 = engine.tick(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                        session_id=session, observation_id="obs_1")
    assert tick2.prediction_error is not None
    assert tick2.prediction_error > 0.0


def test_world_model_version_is_vjepa2():
    from cloud.jepa_service.engine import ImmersiveJEPAEngine
    engine = ImmersiveJEPAEngine(device="cpu")
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    tick = engine.tick(frame, session_id="version_test", observation_id="obs_0")
    assert tick.world_model_version == "vjepa2"


def test_jepatick_has_all_sprint6_fields():
    from cloud.jepa_service.engine import ImmersiveJEPAEngine
    engine = ImmersiveJEPAEngine(device="cpu")
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    engine.tick(frame, session_id="fields_test", observation_id="obs_0")
    tick = engine.tick(frame, session_id="fields_test", observation_id="obs_1")
    payload = tick.to_payload()
    for field in ["prediction_error", "surprise_score", "world_model_version",
                  "epistemic_uncertainty", "aleatoric_uncertainty"]:
        assert hasattr(payload, field), f"JEPATickPayload missing: {field}"
