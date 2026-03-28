"""Tests for SemanticAnchorGraph."""
import numpy as np
import pytest

from cloud.jepa_service.anchor_graph import (
    BOOTSTRAP_TEMPLATES,
    AnchorMatch,
    SemanticAnchorGraph,
    SemanticAnchorTemplate,
    _patch_to_rowcol,
    _rowcol_to_patch,
    _spatial_relation,
)
from cloud.jepa_service.depth_separator import DepthStrataMap


def _uniform_tokens(n: int = 196) -> np.ndarray:
    rng = np.random.default_rng(42)
    tokens = rng.standard_normal((n, 384)).astype(np.float32)
    norms = np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-8
    return (tokens / norms).astype(np.float32)


def _midground_strata() -> DepthStrataMap:
    return DepthStrataMap.cold_start()


def test_bootstrap_templates_have_distinct_names():
    names = [template.name for template in BOOTSTRAP_TEMPLATES]
    assert len(names) == len(set(names)), "All template names must be unique"


def test_spatial_relation_above():
    assert _spatial_relation(14, 0) == "ABOVE"


def test_spatial_relation_below():
    assert _spatial_relation(0, 14) == "BELOW"


def test_spatial_relation_none_for_nonadjacent():
    assert _spatial_relation(0, 28) is None


def test_match_returns_list_of_anchor_matches():
    sag = SemanticAnchorGraph()
    tokens = _uniform_tokens()
    strata = _midground_strata()
    regions = [[0, 1, 2, 14, 15, 16], [100, 101, 114, 115]]
    results = sag.match(tokens, strata, regions)
    assert isinstance(results, list)
    assert all(isinstance(result, AnchorMatch) for result in results)


def test_empty_mask_regions_uses_fallback_grid():
    sag = SemanticAnchorGraph()
    tokens = _uniform_tokens()
    strata = _midground_strata()
    results = sag.match(tokens, strata, [])
    assert len(results) > 0, "Fallback grid regions should produce matches"


def test_cylindrical_template_prefers_background():
    cyl = next(template for template in BOOTSTRAP_TEMPLATES if template.name == "cylindrical_object")
    assert cyl.depth_preference == "background"


def test_chair_and_cylindrical_have_distinct_topologies():
    chair = next(template for template in BOOTSTRAP_TEMPLATES if template.name == "chair_seated")
    cyl = next(template for template in BOOTSTRAP_TEMPLATES if template.name == "cylindrical_object")
    assert chair.depth_preference != cyl.depth_preference or chair.min_nodes_required != cyl.min_nodes_required


def test_telescope_behind_person_assigned_different_depth_strata():
    h, w = 14, 14
    fg_mask = np.zeros((h, w), dtype=bool)
    bg_mask = np.zeros((h, w), dtype=bool)
    fg_mask[:7, :7] = True
    bg_mask[7:, 7:] = True
    mid_mask = ~fg_mask & ~bg_mask

    custom_strata = DepthStrataMap(
        depth_proxy=np.where(fg_mask, 2.0, np.where(bg_mask, 0.1, 0.7)).astype(np.float32),
        foreground_mask=fg_mask,
        midground_mask=mid_mask,
        background_mask=bg_mask,
        confidence=0.85,
        strata_entropy=1.2,
    )

    person_patches = [_rowcol_to_patch(row, col) for row in range(7) for col in range(4)]
    telescope_patches = [_rowcol_to_patch(row, 10) for row in range(7, 13)]

    sag = SemanticAnchorGraph()
    tokens = _uniform_tokens()
    results = sag.match(tokens, custom_strata, [person_patches, telescope_patches])

    assert len(results) == 2
    strata = [result.depth_stratum for result in results]
    assert "foreground" in strata
    assert "background" in strata
    assert strata[0] != strata[1]


def test_novel_region_returns_anchor_match_with_is_novel_true():
    sag = SemanticAnchorGraph()
    results = sag.match(_uniform_tokens(), _midground_strata(), [[77]])
    assert any(result.is_novel for result in results)


def test_learn_from_confirmation_adds_to_learned_templates():
    sag = SemanticAnchorGraph()
    sag.learn_template_from_confirmation(
        region_patches=[0, 1, 14, 15],
        confirmed_label="my_telescope",
        patch_tokens=_uniform_tokens(),
    )
    assert len(sag._learned_templates) == 1
    assert sag._learned_templates[0].name == "my_telescope"


def test_template_serialization_roundtrip():
    template = BOOTSTRAP_TEMPLATES[0]
    data = template.to_dict()
    clone = SemanticAnchorTemplate.from_dict(data)
    assert clone.name == template.name
    assert clone.min_nodes_required == template.min_nodes_required
    assert len(clone.edges) == len(template.edges)


def test_match_never_raises_on_empty_input():
    sag = SemanticAnchorGraph()
    result = sag.match(
        patch_tokens=np.zeros((196, 384), dtype=np.float32),
        depth_strata=_midground_strata(),
        mask_regions=[],
    )
    assert isinstance(result, list)


def test_anchor_match_to_dict_is_json_serializable():
    import json

    sag = SemanticAnchorGraph()
    results = sag.match(_uniform_tokens(), _midground_strata(), [[0, 1, 2]])
    for result in results:
        json.dumps(result.to_dict())


def test_no_torch_import_in_anchor_graph():
    import importlib
    import sys

    modules = [name for name in sys.modules if "anchor_graph" in name]
    for module_name in modules:
        del sys.modules[module_name]
    importlib.import_module("cloud.jepa_service.anchor_graph")
    assert "torch" not in sys.modules
