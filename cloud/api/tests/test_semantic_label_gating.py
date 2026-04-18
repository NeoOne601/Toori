from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from cloud.runtime.app import create_app
from cloud.runtime.models import BoundingBox, EntityTrack, Observation
from cloud.runtime.service import RuntimeContainer
from cloud.runtime.smriti_gemma4_enricher import SmetiGemma4Enricher
from cloud.runtime.world_model import build_object_summary, extract_entity_labels


def _make_observation(*, tags: list[str], summary: str | None = None, metadata: dict | None = None) -> Observation:
    return Observation.model_construct(
        id="obs-semantic-gate",
        session_id="demo",
        created_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
        world_state_id=None,
        observation_kind="camera",
        image_path="/tmp/image.png",
        thumbnail_path="/tmp/thumb.png",
        width=32,
        height=32,
        embedding=[0.0] * 128,
        summary=summary,
        source_query=None,
        tags=tags,
        confidence=0.8,
        novelty=0.1,
        providers=["basic"],
        metadata=metadata or {},
    )


def test_descriptor_labels_are_not_promoted_to_summary_or_tags():
    observation = _make_observation(
        tags=["rgb_histogram+edge_histogram", "dominant_color", "entity-1", "person standing"],
        summary="rgb histogram edge histogram",
        metadata={
            "primary_object_label": "rgb_histogram+edge_histogram",
            "summary_candidates": ["rgb_histogram+edge_histogram", "dominant_color", "entity-1"],
        },
    )

    summary, summary_metadata = build_object_summary(
        observation,
        provider_metadata={
            "top_label": "rgb_histogram+edge_histogram",
            "dominant_color": "red",
            "brightness_label": "bright",
            "edge_label": "textured",
        },
        proposal_boxes=[
            BoundingBox(x=0.1, y=0.1, width=0.3, height=0.3, label="rgb_histogram+edge_histogram", score=0.9),
        ],
        answer_text=None,
        query=None,
    )

    assert summary == "Observed scene"
    assert summary_metadata["primary_object_label"] is None
    assert summary_metadata["summary_candidates"] == []
    assert extract_entity_labels(observation) == ["person standing"]


def test_basic_provider_descriptor_labels_do_not_surface():
    app = create_app()
    runtime = app.state.runtime
    settings = runtime.get_settings()
    settings.providers["onnx"].enabled = False
    settings.providers["basic"].enabled = True
    runtime.update_settings(settings)

    image = Image.new("RGB", (64, 64), color=(160, 160, 160))

    runtime.providers.dinov2.object_proposals = lambda image, config, max_proposals=12: [  # type: ignore[method-assign]
        BoundingBox(x=0.2, y=0.2, width=0.4, height=0.4, label=None, score=0.9)
    ]

    def fake_basic_perceive(crop):
        return [0.0] * 128, 0.91, {"descriptor": "rgb_histogram+edge_histogram"}

    runtime.providers.basic.perceive = fake_basic_perceive  # type: ignore[method-assign]

    proposals = runtime.providers.object_proposals(settings, image, provider_name="dinov2", max_proposals=3)

    assert proposals
    assert proposals[0].label == "object"
    assert all("histogram" not in (proposal.label or "") for proposal in proposals)
    assert all(proposal.label not in {"dominant color", "brightness label", "edge label"} for proposal in proposals)


def test_open_vocab_label_requires_trained_tvlc_context(monkeypatch):
    enricher = SmetiGemma4Enricher(mlx_provider=object(), mlx_config=SimpleNamespace(enabled=True))

    async def fake_mlx(*args, **kwargs):
        return {"text": "telescope", "model": "gemma4"}

    monkeypatch.setattr(enricher, "_mlx", fake_mlx)

    trained = asyncio.run(
        enricher.get_open_vocab_label(
            anchor_name="chair_seated",
            depth_stratum="foreground",
            confidence=0.92,
            patch_count=24,
            tvlc_context="orange telescope on shelf",
            connector_type="tvlc_trained",
            image_base64=None,
        )
    )
    random_init = asyncio.run(
        enricher.get_open_vocab_label(
            anchor_name="chair_seated",
            depth_stratum="foreground",
            confidence=0.92,
            patch_count=24,
            tvlc_context="orange telescope on shelf",
            connector_type="tvlc_random_init",
            image_base64=None,
        )
    )

    assert trained == "telescope"
    assert random_init == "chair"


def test_context_hint_prefers_prototype_matches():
    enricher = SmetiGemma4Enricher(mlx_provider=None, mlx_config=None)
    hint = enricher._resolve_semantic_label(  # type: ignore[attr-defined]
        "telescope",
        fallback="chair",
        tvlc_context="[TVLC visual context (tvlc-trained): prototype_matches=telescope:0.82;chair:0.31; slot_activations=0.1,0.2]",
    )[1]["context_hint"]
    assert hint == "telescope"


def test_semantic_extractor_can_reject_gemma_label_against_tvlc_hint(monkeypatch):
    enricher = SmetiGemma4Enricher(mlx_provider=object(), mlx_config=SimpleNamespace(enabled=True))

    async def fake_mlx(*args, **kwargs):
        return {"text": "dumbbell", "model": "gemma4"}

    class FakeExtractor:
        def cosine_similarity(self, left: str, right: str) -> float:
            if left == "dumbbell" and right == "telescope":
                return 0.02
            return 0.75

    monkeypatch.setattr(enricher, "_mlx", fake_mlx)
    monkeypatch.setattr(enricher, "_get_semantic_extractor", lambda: FakeExtractor())

    label, evidence = asyncio.run(
        enricher.get_open_vocab_label_with_evidence(
            anchor_name="chair_seated",
            depth_stratum="foreground",
            confidence=0.92,
            patch_count=24,
            tvlc_context="[TVLC visual context (tvlc-trained): prototype_matches=telescope:0.82;chair:0.31; slot_activations=0.1,0.2]",
            connector_type="tvlc_trained",
            image_base64=None,
        )
    )

    assert label == "telescope"
    assert evidence["semantic_gate_passed"] is False
    assert evidence["reason"] == "semantic_mismatch_to_tvlc_hint"


def _make_track(*, track_id: str = "trk_1", session_id: str = "demo") -> EntityTrack:
    now = datetime(2026, 4, 18, tzinfo=timezone.utc)
    return EntityTrack(
        id=track_id,
        session_id=session_id,
        label="entity-1",
        status="visible",
        first_seen_at=now,
        last_seen_at=now,
        first_observation_id="obs-1",
        last_observation_id="obs-1",
        observations=["obs-1"],
        visibility_streak=1,
        occlusion_count=0,
        reidentification_count=0,
        persistence_confidence=0.8,
        continuity_score=0.8,
        last_similarity=0.8,
        prototype_embedding=[0.0] * 384,
        status_history=["visible"],
        metadata={},
    )


def test_slot_entropy_reports_spread_votes_as_high_entropy():
    from cloud.perception.tvlc_connector import _slot_entropy

    spread = np.array([4, 4, 4, 4, 4, 4, 4, 4], dtype=np.int64)
    concentrated = np.array([12, 0, 0, 0], dtype=np.int64)

    assert _slot_entropy(spread, 32) > 0.9
    assert _slot_entropy(concentrated, 12) == 0.0


def test_semantic_pass_rejects_incoherent_compound_labels(tmp_path):
    runtime = RuntimeContainer(data_dir=str(tmp_path))
    label, score = runtime._perform_semantic_pass(
        SimpleNamespace(score_anchor=lambda tokens, indices: ("woman man", 0.93)),
        {"patch_indices": [1, 2], "template_name": "person_torso", "depth_stratum": "foreground"},
        np.zeros((196, 384), dtype=np.float32),
        [],
        0,
    )

    assert label is None
    assert score == 0.0


def test_semantic_pass_vetoes_person_label_for_non_person_sag_template(tmp_path):
    runtime = RuntimeContainer(data_dir=str(tmp_path))
    label, score = runtime._perform_semantic_pass(
        SimpleNamespace(score_anchor=lambda tokens, indices: ("man", 0.93)),
        {"patch_indices": [10, 11], "template_name": "cylindrical_object", "depth_stratum": "background"},
        np.zeros((196, 384), dtype=np.float32),
        [],
        0,
    )

    assert label is None
    assert score == 0.0


def test_fallback_anchor_label_covers_new_sag_aliases(tmp_path):
    runtime = RuntimeContainer(data_dir=str(tmp_path))

    assert runtime._fallback_anchor_label("cylindrical_object", "foreground") == "object"
    assert runtime._fallback_anchor_label("cylindrical_object", "midground") == "cylindrical object"
    assert runtime._fallback_anchor_label("screen_display", "background") == "screen"
    assert runtime._fallback_anchor_label("desk_surface", "background") == "surface"
    assert runtime._fallback_anchor_label("background_plane", "background") == "background"
    assert runtime._fallback_anchor_label("chair_seated", "midground") == "chair"
    assert runtime._fallback_anchor_label("hand_region", "foreground") == "hand"
    assert runtime._fallback_anchor_label("person_torso", "foreground") == "person"
    assert runtime._fallback_anchor_label("unknown", "midground") == "object"


def test_confirm_entity_label_route_persists_confirmed_label(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    runtime = app.state.runtime
    runtime.store.save_entity_tracks([_make_track(track_id="trk_demo")])
    client = TestClient(app)

    response = client.post(
        "/v1/entity-tracks/trk_demo/label",
        params={"session_id": "demo"},
        json={"label": "telescope", "confirmed": True},
    )

    assert response.status_code == 200
    assert response.json() == {
        "updated": True,
        "track_id": "trk_demo",
        "label": "telescope",
    }

    tracks = runtime.store.list_entity_tracks(session_id="demo", limit=8)
    assert tracks[0].label == "telescope"
    assert tracks[0].metadata["confirmed_label"] == "telescope"
