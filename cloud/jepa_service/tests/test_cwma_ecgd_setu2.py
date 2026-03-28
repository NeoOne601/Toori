"""Tests for CWMA, ECGD, and Setu-2."""
import numpy as np
import pytest

from cloud.jepa_service.anchor_graph import AnchorMatch
from cloud.jepa_service.confidence_gate import EpistemicConfidenceGate, GateResult
from cloud.jepa_service.depth_separator import DepthStrataMap
from cloud.jepa_service.world_model_alignment import (
    BOOTSTRAP_SCPT,
    RELATIONS,
    TEMPLATE_NAMES,
    CrossModalWorldModelAligner,
)
from cloud.runtime.setu2 import Setu2Bridge, SetuDescription


def _make_match(
    template: str,
    patches: list[int],
    stratum: str,
    confidence: float = 0.80,
) -> AnchorMatch:
    embedding = np.random.default_rng(42).standard_normal(384).astype(np.float32)
    return AnchorMatch(
        template_name=template,
        confidence=confidence,
        patch_indices=patches,
        depth_stratum=stratum,
        centroid_patch=patches[0] if patches else 0,
        embedding_centroid=embedding / (np.linalg.norm(embedding) + 1e-8),
        is_novel=False,
        bbox_normalized={"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.3},
    )


def _fg_bg_strata() -> DepthStrataMap:
    fg = np.zeros((14, 14), dtype=bool)
    bg = np.zeros((14, 14), dtype=bool)
    fg[:7, :7] = True
    bg[7:, 7:] = True
    mid = ~fg & ~bg
    return DepthStrataMap(
        depth_proxy=np.where(fg, 2.0, np.where(bg, 0.1, 0.7)).astype(np.float32),
        foreground_mask=fg,
        midground_mask=mid,
        background_mask=bg,
        confidence=0.85,
        strata_entropy=1.2,
    )


def test_bootstrap_scpt_shape():
    assert BOOTSTRAP_SCPT.shape == (len(TEMPLATE_NAMES), len(TEMPLATE_NAMES), len(RELATIONS))


def test_bootstrap_scpt_rows_sum_to_one():
    sums = BOOTSTRAP_SCPT.sum(axis=2)
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_alignment_returns_same_shape_energy_map():
    cwma = CrossModalWorldModelAligner()
    energy = np.random.rand(14, 14).astype(np.float32)
    strata = _fg_bg_strata()
    matches = [_make_match("person_torso", list(range(49)), "foreground")]
    result_map, loss = cwma.apply_alignment(energy, matches, strata)
    assert result_map is not None
    assert result_map.shape == (14, 14)
    assert result_map.dtype == np.float32


def test_alignment_penalizes_cylinder_in_foreground_with_person():
    cwma = CrossModalWorldModelAligner(lambda_cwma=0.5)
    energy_before = np.zeros((14, 14), dtype=np.float32)
    strata = _fg_bg_strata()
    person = _make_match("person_torso", list(range(10)), "foreground")
    cyl_wrongly_foreground = _make_match("cylindrical_object", list(range(10, 20)), "foreground")
    energy_after, loss = cwma.apply_alignment(energy_before, [person, cyl_wrongly_foreground], strata)
    assert energy_after is not None
    assert energy_after.sum() > energy_before.sum()
    assert loss > 0.0


def test_alignment_does_not_penalize_consistent_config():
    cwma = CrossModalWorldModelAligner(lambda_cwma=0.5)
    energy_before = np.zeros((14, 14), dtype=np.float32)
    strata = _fg_bg_strata()
    person = _make_match("person_torso", list(range(10)), "foreground")
    desk = _make_match("desk_surface", list(range(10, 20)), "background")
    energy_after, loss = cwma.apply_alignment(energy_before, [person, desk], strata)
    assert energy_after is not None
    assert energy_after.sum() <= energy_before.sum() + 1e-3


def test_alignment_returns_original_on_empty_matches():
    cwma = CrossModalWorldModelAligner()
    energy = np.random.rand(14, 14).astype(np.float32)
    result, loss = cwma.apply_alignment(energy, [], _fg_bg_strata())
    assert result is not None
    assert np.allclose(result, energy)
    assert loss == 0.0


def test_gate_passes_high_confidence_clean_region():
    gate = EpistemicConfidenceGate()
    match = _make_match("person_torso", list(range(7 * 7)), "foreground", confidence=0.90)
    result = gate.evaluate(match, _fg_bg_strata(), [0.05] * 10)
    assert result.passes is True
    assert result.consistency_score > 0.5
    assert result.estimated_hallucination_risk < 0.5


def test_gate_fails_low_confidence_anchor():
    gate = EpistemicConfidenceGate(tau_anchor=0.80)
    match = _make_match("unknown", [0, 1], "midground", confidence=0.30)
    result = gate.evaluate(match, _fg_bg_strata(), [0.1] * 10)
    assert result.passes is False
    assert any("anchor_confidence" in reason for reason in result.failure_reasons)


def test_gate_fails_high_energy_variance():
    gate = EpistemicConfidenceGate(tau_variance=0.10)
    match = _make_match("chair_seated", list(range(10)), "midground", confidence=0.85)
    volatile_history = [float(i % 5) for i in range(8)]
    result = gate.evaluate(match, _fg_bg_strata(), volatile_history)
    assert result.passes is False


def test_gate_result_safe_embedding_is_none_when_fails():
    gate = EpistemicConfidenceGate(tau_anchor=0.99)
    match = _make_match("unknown", [0], "midground", confidence=0.10)
    result = gate.evaluate(match, _fg_bg_strata(), [0.5] * 10)
    assert result.passes is False
    assert result.safe_embedding is None


def test_uncertainty_map_has_correct_shape():
    gate = EpistemicConfidenceGate()
    match = _make_match("person_torso", [0, 1, 2], "foreground", confidence=0.70)
    result = gate.evaluate(match, _fg_bg_strata(), [0.1] * 10)
    assert result.uncertainty_map.shape == (14, 14)


def test_setu2_project_query_returns_correct_shape():
    bridge = Setu2Bridge()
    query = np.random.rand(384).astype(np.float32)
    corpus = np.random.rand(100, 128).astype(np.float32)
    energies = bridge.project_query(query, corpus)
    assert energies.shape == (100,)
    assert energies.dtype == np.float32


def test_setu2_energy_is_nonnegative():
    bridge = Setu2Bridge()
    query = np.random.rand(384).astype(np.float32)
    corpus = np.random.rand(50, 128).astype(np.float32)
    energies = bridge.project_query(query, corpus)
    assert (energies >= 0).all()


def test_setu2_describe_region_passed_gate_no_hallucination():
    bridge = Setu2Bridge()
    gate_result = GateResult(
        passes=True,
        consistency_score=0.85,
        failure_reasons=[],
        safe_embedding=np.zeros(384, dtype=np.float32),
        uncertainty_map=np.zeros((14, 14), dtype=np.float32),
        anchor_name="person_torso",
        depth_stratum="foreground",
        estimated_hallucination_risk=0.15,
    )
    desc = bridge.describe_region(gate_result)
    assert desc.is_uncertain is False
    assert desc.hallucination_risk < 0.20
    assert "foreground" in desc.text.lower() or "person" in desc.text.lower()
    assert desc.anchor_basis == "person_torso"


def test_setu2_describe_region_failed_gate_returns_uncertain():
    bridge = Setu2Bridge()
    gate_result = GateResult(
        passes=False,
        consistency_score=0.25,
        failure_reasons=["anchor_confidence=0.30 < τ=0.55"],
        safe_embedding=None,
        uncertainty_map=np.ones((14, 14), dtype=np.float32),
        anchor_name="unknown",
        depth_stratum="midground",
        estimated_hallucination_risk=0.75,
    )
    desc = bridge.describe_region(gate_result)
    assert desc.is_uncertain is True
    assert desc.uncertainty_map is not None


def test_telescope_described_as_cylindrical_background_not_body_part():
    bridge = Setu2Bridge()
    gate = EpistemicConfidenceGate()
    strata = _fg_bg_strata()

    telescope_patches = [7 * 14 + 7, 8 * 14 + 7, 9 * 14 + 7, 10 * 14 + 7, 11 * 14 + 7]
    tel_match = _make_match("cylindrical_object", telescope_patches, "background", 0.87)
    gate_result = gate.evaluate(tel_match, strata, [0.05] * 10)
    desc = bridge.describe_region(gate_result)

    body_part_words = ["shoulder", "arm", "torso", "wig", "hair", "neck", "head"]
    for word in body_part_words:
        assert word not in desc.text.lower(), f"Telescope description contains '{word}'"

    assert "cylindrical" in desc.text.lower() or "background" in desc.text.lower()


def test_setu2_w_update_changes_metric():
    bridge = Setu2Bridge(seed=42)
    w_before = bridge._W.copy()
    query = np.random.rand(384).astype(np.float32)
    positive = np.random.rand(128).astype(np.float32)
    bridge.update_metric_w([(query, positive)], [])
    assert not np.allclose(bridge._W, w_before)


def test_full_pipeline_produces_grounded_description():
    from cloud.jepa_service.engine import ImmersiveJEPAEngine

    engine = ImmersiveJEPAEngine(device="cpu")
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    frame[50:150, 50:150, 0] = 200
    frame[160:200, 160:200, 2] = 180
    tick = engine.tick(frame, session_id="regression", observation_id="obs_regr")
    assert tick.setu_descriptions is not None
    assert isinstance(tick.setu_descriptions, list)
    for item in tick.setu_descriptions:
        assert "description" in item
        assert "hallucination_risk" in item["description"]
