"""Tests for TemporalParallaxDepthSeparator."""
import numpy as np
import pytest

from cloud.jepa_service.depth_separator import (
    BACKGROUND_THRESHOLD,
    FOREGROUND_THRESHOLD,
    DepthStrataMap,
    TemporalParallaxDepthSeparator,
)


def _zeros() -> np.ndarray:
    return np.zeros((14, 14), dtype=np.float32)


def _high_change() -> np.ndarray:
    """Energy map where all patches changed a lot (simulates person movement)."""
    arr = np.ones((14, 14), dtype=np.float32) * 2.0
    return arr


def _stable() -> np.ndarray:
    """Energy map where nothing changed (background)."""
    return np.full((14, 14), 0.05, dtype=np.float32)


def test_cold_start_returns_midground_with_zero_confidence():
    tpds = TemporalParallaxDepthSeparator()
    result = tpds.update(_stable(), _zeros(), np.zeros((196, 384), np.float32))
    assert result.confidence == 0.0
    assert result.midground_mask.all()
    assert not result.foreground_mask.any()
    assert not result.background_mask.any()


def test_cold_start_exactly_covers_first_two_frames():
    tpds = TemporalParallaxDepthSeparator()
    r1 = tpds.update(_stable(), _zeros(), np.zeros((196, 384), np.float32))
    r2 = tpds.update(_stable(), _stable(), np.zeros((196, 384), np.float32))
    assert r1.confidence == 0.0
    assert r2.confidence == 0.0
    r3 = tpds.update(_stable(), _stable(), np.zeros((196, 384), np.float32))
    assert r3.confidence >= 0.0


def test_stable_background_gets_low_depth_proxy():
    tpds = TemporalParallaxDepthSeparator()
    stable = _stable()
    for _ in range(5):
        result = tpds.update(stable, stable, np.zeros((196, 384), np.float32))
    assert result.background_mask.sum() > result.foreground_mask.sum()


def test_high_change_scene_gets_foreground_patches():
    tpds = TemporalParallaxDepthSeparator()
    high = _high_change()
    stable = _stable()
    for _ in range(5):
        result = tpds.update(high, stable, np.zeros((196, 384), np.float32))
    assert result.foreground_mask.any(), "High-change scene should have foreground patches"


def test_tpds_boosts_foreground_on_high_prediction_error():
    tpds_normal = TemporalParallaxDepthSeparator()
    tpds_boosted = TemporalParallaxDepthSeparator()
    
    # Create an energy map that is right on the boundary between midground and foreground
    borderline_change = np.full((14, 14), 1.0, dtype=np.float32)
    stable = _stable()
    
    # Warmup
    for _ in range(3):
        tpds_normal.update(stable, stable, np.zeros((196, 384), np.float32))
        tpds_boosted.update(stable, stable, np.zeros((196, 384), np.float32))

    # Apply borderline change
    res_normal = tpds_normal.update(borderline_change, stable, np.zeros((196, 384), np.float32))
    res_boosted = tpds_boosted.update(
        borderline_change, stable, np.zeros((196, 384), np.float32), prediction_error=0.8
    )

    # Boosted version should have a higher depth proxy proxy mean
    assert res_boosted.depth_proxy.mean() > res_normal.depth_proxy.mean()


def test_depth_strata_masks_are_mutually_exclusive():
    tpds = TemporalParallaxDepthSeparator()
    for _ in range(6):
        result = tpds.update(
            np.random.rand(14, 14).astype(np.float32),
            np.random.rand(14, 14).astype(np.float32),
            np.zeros((196, 384), np.float32),
        )
    union = result.foreground_mask | result.midground_mask | result.background_mask
    assert union.all(), "Every patch must belong to exactly one stratum"
    intersection_fg_bg = result.foreground_mask & result.background_mask
    assert not intersection_fg_bg.any(), "FG and BG must not overlap"


def test_entropy_higher_for_mixed_scene_than_uniform():
    tpds_mixed = TemporalParallaxDepthSeparator()
    tpds_uniform = TemporalParallaxDepthSeparator()
    mixed = np.zeros((14, 14), dtype=np.float32)
    mixed[:7, :] = 2.5
    mixed[7:, :] = 0.02
    uniform = np.full((14, 14), 0.02, dtype=np.float32)
    for _ in range(5):
        r_mixed = tpds_mixed.update(mixed, _zeros(), np.zeros((196, 384), np.float32))
        r_uniform = tpds_uniform.update(uniform, uniform, np.zeros((196, 384), np.float32))
    assert r_mixed.strata_entropy >= r_uniform.strata_entropy


def test_reset_clears_temporal_buffer_and_returns_cold_start():
    tpds = TemporalParallaxDepthSeparator()
    for _ in range(10):
        tpds.update(_high_change(), _stable(), np.zeros((196, 384), np.float32))
    tpds.reset()
    result = tpds.update(_high_change(), _stable(), np.zeros((196, 384), np.float32))
    assert result.confidence == 0.0, "After reset, must return to cold start"


def test_to_dict_produces_serializable_output():
    import json

    result = DepthStrataMap.cold_start()
    d = result.to_dict()
    json.dumps(d)
    assert "depth_proxy" in d
    assert "foreground_mask" in d
    assert "confidence" in d


@pytest.mark.skipif("torch" in __import__("sys").modules, reason="torch already loaded by other tests")
def test_pure_numpy_no_torch_import():
    import importlib
    import sys

    mods = [k for k in sys.modules if "depth_separator" in k]
    for module_name in mods:
        del sys.modules[module_name]
    mod = importlib.import_module("cloud.jepa_service.depth_separator")
    assert mod is not None
    assert "torch" not in sys.modules, "depth_separator must not import torch"


def test_jepa_tick_contains_depth_strata_field():
    """Verify JEPATick dataclass was correctly extended."""
    import dataclasses

    from cloud.runtime.models import JEPATick

    fields = {field.name for field in dataclasses.fields(JEPATick)}
    assert "depth_strata" in fields, "JEPATick must have depth_strata field"


def test_immersive_engine_tick_returns_depth_strata():
    """Integration: ImmersiveJEPAEngine.tick() populates depth_strata."""
    from cloud.jepa_service.engine import ImmersiveJEPAEngine

    engine = ImmersiveJEPAEngine(device="cpu")
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    frame[50:150, 50:150] = 128
    tick = engine.tick(frame, session_id="test_tpds", observation_id="obs_tpds")
    assert tick.depth_strata is not None
    assert "depth_proxy" in tick.depth_strata
    assert "confidence" in tick.depth_strata
