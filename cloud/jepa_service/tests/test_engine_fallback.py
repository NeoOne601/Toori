import numpy as np


def test_engine_uses_vits14_fallback_when_vjepa2_local_load_fails(monkeypatch):
    import cloud.jepa_service.engine as engine_module

    class BrokenVJepa2:
        device = "cpu"
        encoder_type = "vjepa2"
        model_id = "facebook/vjepa2-vitl-fpc64-256"
        n_frames = 8
        is_loaded = False

        def ensure_loaded(self):
            raise RuntimeError("V-JEPA2 weights are not cached locally")

        def encode(self, _frame):
            raise AssertionError("encode() should not run after ensure_loaded() fails")

    class FakeViTS14Encoder:
        def __init__(self, _model_path: str):
            pass

        @classmethod
        def is_available(cls) -> bool:
            return True

        @classmethod
        def default_model_path(cls) -> str:
            return "models/vision/dinov2_vits14.onnx"

        def encode(self, _frame):
            global_emb = np.ones(384, dtype=np.float32)
            global_emb /= np.linalg.norm(global_emb) + 1e-9
            patch_tokens = np.ones((196, 384), dtype=np.float32)
            patch_tokens /= np.linalg.norm(patch_tokens, axis=1, keepdims=True) + 1e-9
            return global_emb, patch_tokens

    monkeypatch.setattr(engine_module, "_is_test_environment", lambda: False)
    monkeypatch.setattr(engine_module, "get_vjepa2_encoder", lambda: BrokenVJepa2())
    monkeypatch.setattr("cloud.perception.vits14_onnx_encoder.ViTS14OnnxEncoder", FakeViTS14Encoder)

    engine = engine_module.ImmersiveJEPAEngine(device="cpu")
    frame = np.zeros((224, 224, 3), dtype=np.uint8)

    tick = engine.tick(frame, session_id="fallback-local-only", observation_id="obs_0")

    assert tick.world_model_version == "dinov2-vits14-onnx"
    assert tick.last_tick_encoder_type == "dinov2-vits14-onnx"
    assert tick.degraded is False


def test_engine_uses_vits14_fallback_when_worker_disables_vjepa2(monkeypatch):
    import cloud.jepa_service.engine as engine_module

    class FakeViTS14Encoder:
        def __init__(self, _model_path: str):
            pass

        @classmethod
        def is_available(cls) -> bool:
            return True

        @classmethod
        def default_model_path(cls) -> str:
            return "models/vision/dinov2_vits14.onnx"

        def encode(self, _frame):
            global_emb = np.ones(384, dtype=np.float32)
            global_emb /= np.linalg.norm(global_emb) + 1e-9
            patch_tokens = np.ones((196, 384), dtype=np.float32)
            patch_tokens /= np.linalg.norm(patch_tokens, axis=1, keepdims=True) + 1e-9
            return global_emb, patch_tokens

    monkeypatch.setattr(engine_module, "_is_test_environment", lambda: False)
    monkeypatch.setattr(
        engine_module,
        "get_vjepa2_encoder",
        lambda: (_ for _ in ()).throw(AssertionError("V-JEPA2 should stay disabled in worker fallback mode")),
    )
    monkeypatch.setattr("cloud.perception.vits14_onnx_encoder.ViTS14OnnxEncoder", FakeViTS14Encoder)
    monkeypatch.setenv("TOORI_VJEPA2_DISABLE", "1")

    engine = engine_module.ImmersiveJEPAEngine(device="cpu")
    frame = np.zeros((224, 224, 3), dtype=np.uint8)

    tick = engine.tick(frame, session_id="worker-fallback", observation_id="obs_1")

    assert tick.world_model_version == "dinov2-vits14-onnx"
    assert tick.last_tick_encoder_type == "dinov2-vits14-onnx"
    assert tick.degraded is False
