import subprocess
import sys

import numpy as np

from cloud.jepa_service.engine import ImmersiveJEPAEngine
from cloud.runtime.models import JEPATick


def _sample_frame() -> np.ndarray:
    rows = np.repeat(np.arange(224, dtype=np.uint8)[:, None], 224, axis=1)
    cols = np.repeat(np.arange(224, dtype=np.uint8)[None, :], 224, axis=0)
    return np.stack(
        [
            (cols * 2) % 255,
            (rows * 3) % 255,
            (rows + cols) % 255,
        ],
        axis=-1,
    )


def test_immersive_tick_returns_jepa_tick():
    engine = ImmersiveJEPAEngine(device="cpu")
    tick = engine.tick(_sample_frame(), session_id="immersive", observation_id="obs_test")
    assert isinstance(tick, JEPATick)
    assert tick.energy_map.shape == (14, 14)
    assert isinstance(tick.forecast_errors, dict)
    assert set(tick.forecast_errors) == {1, 2, 5}
    assert tick.session_fingerprint.ndim == 1
    assert tick.planning_time_ms >= 0.0


def test_culturally_agnostic_entity_tick_does_not_require_labels():
    engine = ImmersiveJEPAEngine(device="cpu")
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    frame[:, :, 0] = 37
    frame[:, :, 1] = 91
    frame[:, :, 2] = 153
    tick = engine.tick(frame, session_id="agnostic", observation_id="obs_agnostic")
    assert tick.energy_map.shape == (14, 14)
    assert np.all(np.isfinite(tick.energy_map))
    assert tick.entity_tracks


def test_torch_not_imported_by_engine_module_import():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import cloud.jepa_service.engine; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr.strip()
