"""Sprint 6: Hierarchical Recall Tests.

Tests the Setu-2 retrieval capabilities and metrics.
"""
from __future__ import annotations

import numpy as np
import pytest

from cloud.runtime.setu2 import Setu2Bridge


def test_setu2_bridge_initialization_with_custom_w():
    w = np.full(128, 0.5, dtype=np.float32)
    bridge = Setu2Bridge(w_diagonal=w)
    assert np.allclose(bridge._W, 0.5)


def test_setu2_bridge_initialization_uses_ones_by_default():
    bridge = Setu2Bridge()
    assert np.allclose(bridge._W, 1.0)
    assert len(bridge._W) == 128


def test_project_query_truncates_oversized_query():
    bridge = Setu2Bridge()
    oversized = np.random.rand(512).astype(np.float32)
    corpus = np.random.rand(10, 128).astype(np.float32)

    energies = bridge.project_query(oversized, corpus)
    assert energies.shape == (10,)


def test_project_query_pads_undersized_query():
    bridge = Setu2Bridge()
    undersized = np.random.rand(256).astype(np.float32)
    corpus = np.random.rand(5, 128).astype(np.float32)

    energies = bridge.project_query(undersized, corpus)
    assert energies.shape == (5,)


def test_update_metric_w_clips_values():
    bridge = Setu2Bridge()
    query = np.random.rand(384).astype(np.float32)
    positive = np.random.rand(128).astype(np.float32)
    
    # Use huge learning rate to force clip
    bridge.update_metric_w([(query, positive)], [], learning_rate=1000.0)
    assert np.all(bridge._W >= 0.01)


def test_update_metric_w_pushes_negatives_apart():
    bridge = Setu2Bridge()
    w_before = bridge._W.copy()
    query = np.random.rand(384).astype(np.float32)
    negative = np.random.rand(128).astype(np.float32)
    
    bridge.update_metric_w([], [(query, negative)], learning_rate=0.1)
    
    # Negative examples should increase weights to penalize them more
    assert np.mean(bridge._W) > np.mean(w_before)
