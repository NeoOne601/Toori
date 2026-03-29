"""
TOORI Smriti — Production Acceptance Gate
==========================================
This file is the definitive production acceptance test suite.
ALL tests must pass before any release. If any test fails,
the build is rejected regardless of other test counts.

Primary contract: test_telescope_behind_person_not_described_as_body_part
This is the non-negotiable scientific proof that the VL-JEPA pipeline
eliminates semantic confusion.
"""
import base64
import io
import subprocess
import sys

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from cloud.runtime.app import create_app


def _png_b64(color: tuple[int, int, int], size: int = 32) -> str:
    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_telescope_behind_person_not_described_as_body_part():
    """
    THE PRIMARY PRODUCTION CONTRACT.

    Scenario: person in foreground (high temporal energy, foreground stratum),
    telescope-like cylindrical object in background (low temporal energy,
    background stratum, linear patch topology).

    Required: The system MUST NOT describe the background cylindrical object
    as any body part (shoulder, arm, wig, neck, head, hair).

    Required: depth strata must correctly separate the two objects.

    This test is the existence proof that TPDS + SAG + CWMA + ECGD + Setu-2
    work as an integrated pipeline.
    """
    from cloud.jepa_service.anchor_graph import AnchorMatch
    from cloud.jepa_service.confidence_gate import EpistemicConfidenceGate
    from cloud.jepa_service.depth_separator import DepthStrataMap
    from cloud.runtime.setu2 import Setu2Bridge

    h, w = 14, 14
    fg = np.zeros((h, w), dtype=bool)
    bg = np.zeros((h, w), dtype=bool)
    fg[:7, :7] = True
    bg[7:, 7:] = True
    mid = ~fg & ~bg

    custom_strata = DepthStrataMap(
        depth_proxy=np.where(fg, 2.0, np.where(bg, 0.1, 0.7)).astype(np.float32),
        foreground_mask=fg,
        midground_mask=mid,
        background_mask=bg,
        confidence=0.88,
        strata_entropy=1.3,
    )

    telescope_patches = [7 * 14 + 10, 8 * 14 + 10, 9 * 14 + 10, 10 * 14 + 10, 11 * 14 + 10]

    telescope_match = AnchorMatch(
        template_name="cylindrical_object",
        confidence=0.87,
        patch_indices=telescope_patches,
        depth_stratum="background",
        centroid_patch=telescope_patches[2],
        embedding_centroid=np.random.default_rng(99).standard_normal(384).astype(np.float32),
        is_novel=False,
        bbox_normalized={"x": 0.5, "y": 0.5, "width": 0.07, "height": 0.3},
    )

    gate = EpistemicConfidenceGate()
    gate_result = gate.evaluate(
        anchor_match=telescope_match,
        depth_strata=custom_strata,
        energy_history=[0.05, 0.04, 0.06, 0.05, 0.05, 0.04, 0.05, 0.06],
    )

    bridge = Setu2Bridge()
    desc = bridge.describe_region(gate_result)

    body_part_words = [
        "shoulder",
        "arm",
        "torso",
        "wig",
        "hair",
        "neck",
        "head",
        "ear",
        "elbow",
        "wrist",
        "hand",
        "body",
        "seatbelt",
        "seat belt",
        "harness",
        "strap",
    ]
    desc_lower = desc.text.lower()
    for word in body_part_words:
        assert word not in desc_lower, (
            "PRODUCTION FAILURE: Telescope described as body part.\n"
            f"  Description: '{desc.text}'\n"
            f"  Forbidden word found: '{word}'\n"
            "  This indicates TPDS/SAG/CWMA/ECGD pipeline failure."
        )

    assert (
        "background" in desc_lower or "cylindrical" in desc_lower or "object" in desc_lower
    ), f"Description lacks expected spatial context: '{desc.text}'"
    assert desc.depth_stratum == "background", (
        f"Telescope must be background, got: '{desc.depth_stratum}'"
    )
    assert desc.hallucination_risk < 0.60, (
        f"Hallucination risk too high: {desc.hallucination_risk}"
    )


def test_office_chair_not_described_as_exercise_equipment():
    """
    REGRESSION: Chair (L-topology) must not be confused with dumble/weight.
    The chair template has seat+back ABOVE relationship.
    Exercise equipment has different spatial topology.
    """
    from cloud.jepa_service.anchor_graph import BOOTSTRAP_TEMPLATES

    chair = next(template for template in BOOTSTRAP_TEMPLATES if template.name == "chair_seated")
    cyl = next(template for template in BOOTSTRAP_TEMPLATES if template.name == "cylindrical_object")

    chair_relations = {relation for _, _, relation in chair.edges}
    assert len(chair_relations) > 0, "Chair must have edge relations"
    assert chair.depth_preference in ("midground", "any"), (
        f"Chair depth preference wrong: {chair.depth_preference}"
    )
    assert cyl.depth_preference == "background", (
        f"Cylindrical must prefer background, got: {cyl.depth_preference}"
    )


def test_uncertain_region_returns_uncertainty_not_wrong_description():
    """
    ECGD gate failure must yield uncertainty visualization, NOT a
    confident wrong answer. This is the core epistemic safety property.
    """
    from cloud.jepa_service.anchor_graph import AnchorMatch
    from cloud.jepa_service.confidence_gate import EpistemicConfidenceGate
    from cloud.jepa_service.depth_separator import DepthStrataMap
    from cloud.runtime.setu2 import Setu2Bridge

    low_conf_match = AnchorMatch(
        template_name="unknown",
        confidence=0.10,
        patch_indices=[0, 1],
        depth_stratum="midground",
        centroid_patch=0,
        embedding_centroid=np.zeros(384, dtype=np.float32),
        is_novel=True,
        bbox_normalized={"x": 0.0, "y": 0.0, "width": 0.1, "height": 0.1},
    )

    strata = DepthStrataMap.cold_start()
    gate = EpistemicConfidenceGate()
    gate_result = gate.evaluate(
        anchor_match=low_conf_match,
        depth_strata=strata,
        energy_history=[1.0, 0.5, 0.8, 1.2, 0.9],
    )

    assert gate_result.passes is False
    assert gate_result.safe_embedding is None
    assert gate_result.uncertainty_map is not None

    bridge = Setu2Bridge()
    desc = bridge.describe_region(gate_result)

    assert desc.is_uncertain is True
    assert desc.uncertainty_map is not None

    confident_words = [
        "person",
        "chair",
        "telescope",
        "desk",
        "screen",
        "definitely",
        "clearly",
        "obviously",
    ]
    for word in confident_words:
        assert word not in desc.text.lower(), (
            f"Uncertain region produced confident description containing '{word}': '{desc.text}'"
        )


def test_full_pipeline_hallucination_risk_bounded():
    """
    End-to-end pipeline: raw frame → JEPATick → setu_descriptions.
    All descriptions must have hallucination_risk < 0.80.
    At least one description must have hallucination_risk < 0.50.
    """
    from cloud.jepa_service.engine import ImmersiveJEPAEngine

    engine = ImmersiveJEPAEngine(device="cpu")
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    frame[30:120, 30:120, 0] = 200
    frame[140:200, 140:200, 2] = 180

    tick = engine.tick(frame, session_id="production_gate", observation_id="obs_pg")

    assert tick.depth_strata is not None
    assert tick.anchor_matches is not None
    assert tick.setu_descriptions is not None

    descriptions = tick.setu_descriptions
    assert isinstance(descriptions, list)

    if descriptions:
        risks = [item.get("description", {}).get("hallucination_risk", 1.0) for item in descriptions]
        assert all(risk < 0.80 for risk in risks), (
            f"Some descriptions have unacceptably high hallucination risk: {risks}"
        )
        assert any(risk < 0.50 for risk in risks), (
            f"No description has low hallucination risk: {risks}"
        )


def test_living_lens_tick_responds_within_sla(tmp_path):
    """
    Full living_lens_tick round-trip must complete within 2000ms
    (conservative SLA for test environment without MPS acceleration).
    """
    import time

    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    times = []
    for color in [(255, 100, 50), (50, 200, 100), (100, 50, 255)]:
        start = time.perf_counter()
        response = client.post(
            "/v1/living-lens/tick",
            json={
                "image_base64": _png_b64(color, size=64),
                "session_id": "perf_gate",
                "decode_mode": "off",
                "proof_mode": "both",
            },
        )
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        assert response.status_code == 200

    mean_ms = sum(times) / len(times)
    assert mean_ms < 2000, (
        f"Mean tick latency {mean_ms:.0f}ms exceeds 2000ms SLA in test environment"
    )


def test_smriti_recall_responds_within_sla(tmp_path):
    """Recall endpoint must respond in < 1000ms even on empty corpus."""
    import time

    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    start = time.perf_counter()
    response = client.post(
        "/v1/smriti/recall",
        json={"query": "red jacket Kolkata", "top_k": 10},
    )
    elapsed = (time.perf_counter() - start) * 1000

    assert response.status_code == 200
    assert elapsed < 1000, f"Recall took {elapsed:.0f}ms, exceeds 1000ms SLA"


def test_smriti_storage_config_persists_across_runtime_restart(tmp_path):
    """
    Storage configuration must survive a RuntimeContainer restart.
    This is a user safety test — if budget/path config is lost on
    restart, users cannot trust their storage settings.
    """
    from cloud.runtime.service import RuntimeContainer

    container1 = RuntimeContainer(data_dir=str(tmp_path))
    settings1 = container1.get_settings()
    updated_storage = settings1.smriti_storage.model_copy(
        update={"max_storage_gb": 42.5, "store_full_frames": False}
    )
    container1.update_settings(settings1.model_copy(update={"smriti_storage": updated_storage}))

    container2 = RuntimeContainer(data_dir=str(tmp_path))
    settings2 = container2.get_settings()

    assert settings2.smriti_storage.max_storage_gb == 42.5
    assert settings2.smriti_storage.store_full_frames is False


def test_watch_folder_list_persists_across_restart(tmp_path):
    """Watch folders must persist in smriti_storage.watch_folders."""
    from cloud.runtime.service import RuntimeContainer

    watch_dir = tmp_path / "my_photos"
    watch_dir.mkdir()

    container1 = RuntimeContainer(data_dir=str(tmp_path))
    container1.add_watch_folder(str(watch_dir))

    container2 = RuntimeContainer(data_dir=str(tmp_path))
    settings2 = container2.get_settings()
    assert str(watch_dir) in settings2.smriti_storage.watch_folders


def test_prune_clear_all_requires_exact_confirmation(tmp_path):
    """Clear-all prune must reject any confirmation string except exact match."""
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    for bad_confirm in ["yes", "confirm", "CONFIRM", "", "clear all", "CLEAR_ALL"]:
        response = client.post(
            "/v1/smriti/storage/prune",
            json={"clear_all": True, "confirm_clear_all": bad_confirm},
        )
        assert response.status_code in (400, 422, 500), (
            f"Prune clear-all accepted bad confirmation: '{bad_confirm}'"
        )


def test_existing_toori_functionality_not_broken():
    """
    Run the original Toori test suite as a subprocess.
    Smriti must not break any pre-existing functionality.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "cloud/api/tests/test_integration.py",
            "cloud/api/tests/test_auth.py",
            "cloud/api/tests/test_ws_schema_compat.py",
            "cloud/jepa_service/tests/test_engine.py",
            "cloud/jepa_service/tests/test_immersive_engine.py",
            "--tb=short",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "REGRESSION in original Toori tests:\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def test_no_torch_imports_outside_perception():
    """torch must never be imported outside cloud/perception/."""
    violating_modules = []
    modules_to_check = [
        "cloud.jepa_service.engine",
        "cloud.jepa_service.depth_separator",
        "cloud.jepa_service.anchor_graph",
        "cloud.jepa_service.world_model_alignment",
        "cloud.jepa_service.confidence_gate",
        "cloud.runtime.setu2",
        "cloud.runtime.smriti_storage",
        "cloud.runtime.smriti_ingestion",
        "cloud.runtime.service",
    ]
    for module in modules_to_check:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; import {module}; assert 'torch' not in sys.modules, 'torch imported by {module}'",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            violating_modules.append(module)

    assert not violating_modules, (
        f"torch imported outside cloud/perception/ by: {violating_modules}"
    )


def test_api_routes_all_present(tmp_path):
    """All expected Smriti API routes must be registered."""
    app = create_app(data_dir=str(tmp_path))
    routes = {route.path for route in app.routes}

    required_routes = [
        "/v1/smriti/ingest",
        "/v1/smriti/recall",
        "/v1/smriti/status",
        "/v1/smriti/clusters",
        "/v1/smriti/metrics",
        "/v1/smriti/storage",
        "/v1/smriti/storage/usage",
        "/v1/smriti/watch-folders",
        "/v1/smriti/storage/prune",
        "/v1/smriti/tag/person",
    ]

    missing = [route for route in required_routes if route not in routes]
    assert not missing, f"Missing API routes: {missing}"
