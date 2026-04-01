"""
cloud/jepa_service/tests/conftest.py
V-JEPA 2 test fixtures — M1 8GB safe.
"""
import os
import gc
import numpy as np
import pytest

os.environ.setdefault("TOORI_VJEPA2_ENV",    "test")
os.environ.setdefault("TOORI_VJEPA2_FRAMES", "4")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "vjepa2: marks tests that load V-JEPA 2. Excluded from default suite."
    )


@pytest.fixture(scope="session")
def vjepa2_encoder():
    """Load V-JEPA 2 once per session on CPU with 4 frames."""
    from cloud.perception.vjepa2_encoder import get_vjepa2_encoder

    encoder = get_vjepa2_encoder()
    assert encoder.is_loaded
    assert encoder.encoder_type == "vjepa2"
    assert str(encoder.device) == "cpu", (
        f"Tests must run on CPU, got {encoder.device}. Set TOORI_VJEPA2_ENV=test."
    )

    yield encoder

    from cloud.perception.vjepa2_encoder import reset_encoder_singleton
    reset_encoder_singleton()


@pytest.fixture(autouse=True)
def flush_memory_between_tests():
    yield
    gc.collect()
    gc.collect()
    import sys
    if "torch" in sys.modules:
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
                torch.mps.empty_cache()
        except Exception:
            pass


@pytest.fixture
def random_frame() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def black_frame() -> np.ndarray:
    return np.zeros((224, 224, 3), dtype=np.uint8)
