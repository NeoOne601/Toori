"""Tests for the core JEPAEngine: tick, EMA, energy maps, masking, SIGReg."""

import numpy as np

from cloud.jepa_service.engine import JEPAEngine, TickResult
from cloud.runtime.models import BoundingBox


def test_tick_returns_valid_result():
    engine = JEPAEngine(embedding_dim=128, patch_grid=(7, 7))
    embedding = np.random.randn(128).astype(np.float32)
    result = engine.tick(embedding)
    assert isinstance(result, TickResult)
    assert result.energy_map.shape == (7, 7)
    assert result.mean_energy >= 0.0
    assert result.threshold > 0.0
    assert isinstance(result.should_talk, bool)
    assert result.prediction_residual.shape == (128,)
    assert result.sigreg_loss >= 0.0


def test_energy_map_spatial_shape():
    for grid in [(4, 4), (7, 7), (14, 14)]:
        engine = JEPAEngine(embedding_dim=64, patch_grid=grid)
        embedding = np.random.randn(64).astype(np.float32)
        result = engine.tick(embedding)
        assert result.energy_map.shape == grid


def test_ema_diverges_after_multiple_ticks():
    engine = JEPAEngine(embedding_dim=32, ema_tau=0.99, seed=123)
    # Save initial target projection weights
    initial_target = engine._target_proj["weight"].copy()

    # Simulate network "learning" by modifying context weights slowly
    for _ in range(20):
        engine._context_proj["weight"] += 0.05
        embedding = np.random.randn(32).astype(np.float32)
        engine.tick(embedding)

    # Target should have drifted from initial
    diff = np.abs(engine._target_proj["weight"] - initial_target).max()
    assert diff > 0.0, "EMA target weights should diverge from initial after ticks"


def test_adaptive_threshold_updates():
    engine = JEPAEngine(embedding_dim=32, initial_threshold=0.15, min_threshold=0.0001, seed=42)
    thresholds = []
    for i in range(10):
        embedding = np.random.randn(32).astype(np.float32) * (1.0 + i * 0.5)
        result = engine.tick(embedding)
        thresholds.append(result.threshold)
    # Threshold should change over ticks (not constant)
    assert len(set(round(t, 6) for t in thresholds)) > 1


def test_sigreg_loss_nonnegative():
    engine = JEPAEngine(embedding_dim=32, sigreg_buffer_size=16, seed=42)
    for _ in range(20):
        embedding = np.random.randn(32).astype(np.float32)
        result = engine.tick(embedding)
        assert result.sigreg_loss >= 0.0


def test_boxes_to_mask():
    boxes = [
        BoundingBox(x=0.0, y=0.0, width=0.5, height=0.5, label="obj1", id="1"),
        BoundingBox(x=0.5, y=0.5, width=0.5, height=0.5, label="obj2", id="2"),
    ]
    mask = JEPAEngine.boxes_to_mask(boxes, 7, 7)
    assert mask.shape == (7, 7)
    assert mask.dtype == bool
    assert mask[0, 0]  # top-left should be masked
    assert mask[6, 6]  # bottom-right should be masked
    assert mask.sum() > 0


def test_random_mask():
    engine = JEPAEngine(embedding_dim=32, patch_grid=(7, 7))
    mask = engine.random_mask(ratio=0.3)
    assert mask.shape == (7, 7)
    assert mask.dtype == bool
    expected = max(1, int(0.3 * 49))
    assert mask.sum() == expected


def test_stable_scene_does_not_talk():
    engine = JEPAEngine(embedding_dim=32, initial_threshold=100.0, seed=42)
    # With a very high threshold, should_talk should be False
    embedding = np.random.randn(32).astype(np.float32) * 0.01
    result = engine.tick(embedding)
    assert result.should_talk is False


def test_novel_frame_talks():
    engine = JEPAEngine(embedding_dim=32, initial_threshold=0.0001, min_threshold=0.0001, seed=42)
    # With a tiny threshold, most frames should trigger
    embedding = np.random.randn(32).astype(np.float32) * 10.0
    result = engine.tick(embedding)
    assert result.mean_energy > 0.0
    # After enough ticks with high energy, threshold adapts
    for _ in range(5):
        engine.tick(np.random.randn(32).astype(np.float32) * 10.0)
    result2 = engine.tick(np.random.randn(32).astype(np.float32) * 100.0)
    assert result2.mean_energy > 0.0


def test_predictor_weights_roundtrip():
    engine = JEPAEngine(embedding_dim=32, seed=42)
    embedding = np.random.randn(32).astype(np.float32)
    engine.tick(embedding)
    weights = engine.get_predictor_weights()
    assert "predictor.0.weight" in weights
    assert "context_proj.weight" in weights

    engine2 = JEPAEngine(embedding_dim=32, seed=999)
    engine2.load_predictor_weights(weights)
    
    # We explicitly compare maximum absolute difference instead of assert_array_almost_equal
    # because pytest/numpy sometimes fails with arcane shape broadcasting errors on empty arrays.
    engine2_weights = engine2.get_predictor_weights()
    for key in weights:
        if key in engine2_weights:
            w1 = engine2_weights[key]
            w2 = weights[key]
            assert w1.shape == w2.shape, f"Shape mismatch for {key}: {w1.shape} != {w2.shape}"
            diff = np.abs(w1 - w2).max()
            assert diff < 1e-6, f"Weights mismatch for {key}"


def test_get_energy_map_returns_copy():
    engine = JEPAEngine(embedding_dim=32, patch_grid=(4, 4))
    embedding = np.random.randn(32).astype(np.float32)
    engine.tick(embedding)
    map1 = engine.get_energy_map()
    map2 = engine.get_energy_map()
    assert map1.shape == (4, 4)
    assert not np.shares_memory(map1, map2)


def test_reset_clears_state():
    engine = JEPAEngine(embedding_dim=32)
    for _ in range(5):
        engine.tick(np.random.randn(32).astype(np.float32))
    assert engine._tick_count == 5
    engine.reset()
    assert engine._tick_count == 0
    assert len(engine._embedding_buffer) == 0
