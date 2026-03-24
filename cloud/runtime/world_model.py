from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from statistics import mean
from typing import Iterable
from uuid import uuid4

import numpy as np

from .models import (
    BaselineComparison,
    BaselineModeScore,
    ChallengeRun,
    ContinuitySignal,
    BoundingBox,
    EntityTrack,
    Observation,
    PersistenceSignal,
    PredictionWindow,
    SceneState,
    SearchHit,
    WorldModelMetrics,
)


STOP_WORDS = {
    "the",
    "and",
    "with",
    "that",
    "this",
    "from",
    "into",
    "while",
    "there",
    "their",
    "about",
    "scene",
    "shows",
    "showing",
    "image",
    "camera",
    "current",
    "frame",
    "visual",
    "various",
    "objects",
    "room",
}

NON_ENTITY_TOKENS = {
    "bright",
    "dark",
    "balanced",
    "smooth",
    "textured",
    "edge",
    "color",
    "dominant",
    "scene",
    "light",
    "shadow",
    "neutral",
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
    "grey",
    "visible",
    "occluded",
    "reidentified",
    "re-identified",
    "onnx",
    "basic",
    "cloud",
    "mlx",
    "coreml",
    "tflite",
    "local",
    "perception",
    "reasoning",
}

LIVE_CHALLENGE_GUIDE = [
    "Show a stable object to the lens for a few ticks.",
    "Partially occlude it while keeping the scene otherwise stable.",
    "Fully occlude it for at least one tick.",
    "Reveal the object again and keep it in roughly the same place.",
    "Move the camera away and then return to the scene.",
    "Introduce a distractor or unexpected change and watch the surprise spike.",
]


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_vec = np.array(left, dtype=np.float32)
    right_vec = np.array(right, dtype=np.float32)
    denom = (np.linalg.norm(left_vec) or 1.0) * (np.linalg.norm(right_vec) or 1.0)
    return clamp_unit(np.dot(left_vec, right_vec) / denom)


def mean_embedding(observations: list[Observation], fallback: list[float]) -> list[float]:
    if not observations:
        return fallback
    matrix = np.array([observation.embedding for observation in observations], dtype=np.float32)
    return list(np.mean(matrix, axis=0))


def ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        value = raw.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def normalize_token(raw: str) -> str:
    return (
        raw.lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace(";", " ")
        .strip()
    )


def text_tokens(text: str | None) -> list[str]:
    if not text:
        return []
    tokens: list[str] = []
    for raw in normalize_token(text).split():
        token = raw.strip()
        if len(token) < 3 or token in STOP_WORDS:
            continue
        tokens.append(token)
    return ordered_unique(tokens)


def _metadata_dict(observation: Observation) -> dict:
    return observation.metadata if isinstance(observation.metadata, dict) else {}


def _metadata_boxes(observation: Observation) -> list[BoundingBox]:
    metadata = _metadata_dict(observation)
    raw_boxes = metadata.get("object_proposals") or metadata.get("proposal_boxes") or []
    boxes: list[BoundingBox] = []
    if not isinstance(raw_boxes, list):
        return boxes
    for raw in raw_boxes:
        try:
            if isinstance(raw, BoundingBox):
                boxes.append(raw)
            elif isinstance(raw, dict):
                boxes.append(BoundingBox.model_validate(raw))
        except Exception:
            continue
    return boxes


def _metadata_labels(observation: Observation) -> list[str]:
    metadata = _metadata_dict(observation)
    labels: list[str] = []
    primary = metadata.get("primary_object_label")
    if primary:
        labels.append(str(primary))
    summary_candidates = metadata.get("summary_candidates")
    if isinstance(summary_candidates, list):
        labels.extend(str(item) for item in summary_candidates if item)
    labels.extend(box.label for box in _metadata_boxes(observation) if box.label)
    return ordered_unique(normalize_token(label) for label in labels if label)


def _meaningful_label(label: object) -> str:
    tokens = [token for token in normalize_token(str(label or "")).split() if len(token) >= 3 and token not in NON_ENTITY_TOKENS]
    return " ".join(tokens)


def _strip_caption_boilerplate(text: str) -> str:
    normalized = normalize_token(text)
    prefixes = (
        "the image shows ",
        "the image depicts ",
        "the image appears to show ",
        "this image shows ",
        "this image depicts ",
        "a scene of ",
        "a photo of ",
        "a picture of ",
        "overall the image shows ",
    )
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    for delimiter in (".", ";", " while ", " where ", " with ", " and "):
        if delimiter in normalized:
            normalized = normalized.split(delimiter, 1)[0]
    return normalized.strip(" :-,")


def _compress_answer_text(text: str) -> str:
    cleaned = _strip_caption_boilerplate(text)
    tokens = [token for token in cleaned.split() if token not in NON_ENTITY_TOKENS and token not in STOP_WORDS]
    if tokens:
        return " ".join(tokens[:4])
    if cleaned:
        return " ".join(cleaned.split()[:5])
    return ""


def build_object_summary(
    observation: Observation,
    provider_metadata: dict,
    proposal_boxes: list[BoundingBox],
    *,
    answer_text: str | None = None,
    query: str | None = None,
) -> tuple[str, dict]:
    proposal_labels = ordered_unique(
        _meaningful_label(box.label)
        for box in proposal_boxes
        if box.label and _meaningful_label(box.label)
    )
    full_label = _meaningful_label(provider_metadata.get("top_label") or "")
    candidate_labels = ordered_unique(
        [
            *proposal_labels,
            full_label,
            *[label for label in _metadata_labels(observation) if label not in proposal_labels],
        ]
    )
    primary_label = candidate_labels[0] if candidate_labels else ""
    secondary_labels = candidate_labels[1:4]

    if primary_label:
        summary = primary_label
        if secondary_labels:
            if "person" in primary_label and any(label in {"chair", "seat", "sofa", "couch"} for label in secondary_labels):
                seat = next(label for label in secondary_labels if label in {"chair", "seat", "sofa", "couch"})
                summary = f"{primary_label} seated on {seat}"
            elif secondary_labels[0] not in primary_label:
                summary = f"{primary_label} near {secondary_labels[0]}"
        source = "proposals" if proposal_labels else "perception"
    elif answer_text:
        summary = _compress_answer_text(answer_text)
        source = "answer"
    else:
        descriptor_bits = [
            str(provider_metadata.get("dominant_color") or "").strip(),
            str(provider_metadata.get("brightness_label") or "").strip(),
            str(provider_metadata.get("edge_label") or "").strip(),
        ]
        summary = " ".join(bit for bit in descriptor_bits if bit).strip()
        summary = f"{summary} scene".strip()
        source = "descriptor"

    if not summary:
        summary = "Observed scene"

    summary_metadata = {
        "primary_object_label": primary_label or None,
        "secondary_object_labels": secondary_labels,
        "summary_candidates": candidate_labels[:6],
        "summary_source": source,
        "proposal_labels": proposal_labels,
    }
    return summary, summary_metadata


def extract_observed_elements(observation: Observation) -> list[str]:
    metadata = _metadata_dict(observation)
    perception = metadata.get("perception") if isinstance(metadata, dict) else None
    perception = perception if isinstance(perception, dict) else {}
    values: list[str] = []
    top_label = str(perception.get("top_label") or "").replace("_", " ").strip()
    if top_label:
        values.append(top_label)
    primary_object_label = metadata.get("primary_object_label")
    if primary_object_label:
        values.append(str(primary_object_label))
    values.extend(_metadata_labels(observation))
    values.extend(observation.tags)
    values.extend(text_tokens(observation.summary)[:6])
    return ordered_unique(normalize_token(value) for value in values if value)


def extract_entity_labels(observation: Observation) -> list[str]:
    labels = [
        token
        for token in extract_observed_elements(observation)
        if token not in NON_ENTITY_TOKENS and len(token) >= 3
    ]
    return ordered_unique(labels[:4] or extract_observed_elements(observation)[:2])


def stable_elements_from_history(recent_observations: list[Observation]) -> list[str]:
    counter: Counter[str] = Counter()
    for observation in recent_observations:
        counter.update(extract_entity_labels(observation))
    threshold = 2 if len(recent_observations) >= 3 else 1
    stable = [token for token, count in counter.most_common() if count >= threshold]
    return stable[:6]


def build_prediction_window(
    observation: Observation,
    previous_state: SceneState | None,
    recent_observations: list[Observation],
    entity_tracks: list[EntityTrack],
) -> PredictionWindow:
    stable_history = stable_elements_from_history(recent_observations[:4])
    predicted_tags = ordered_unique(
        [
            *(previous_state.stable_elements if previous_state else []),
            *(previous_state.observed_elements if previous_state else []),
            *stable_history,
            *[track.label for track in entity_tracks if track.status in {"visible", "re-identified", "occluded"}],
        ]
    )[:8]
    stable_elements = predicted_tags[:4]
    confidence_sources = [
        clamp_unit(1.0 - observation.novelty),
        clamp_unit(len(stable_history) / max(len(recent_observations), 1)),
        clamp_unit(len([track for track in entity_tracks if track.status != "disappeared"]) / max(len(entity_tracks), 1))
        if entity_tracks
        else 0.5,
    ]
    predicted_summary = (
        f"Expect continuity around {'; '.join(stable_elements)}"
        if stable_elements
        else "Expect a broadly similar scene to recent observations"
    )
    return PredictionWindow(
        previous_observation_id=recent_observations[0].id if recent_observations else None,
        context_observation_ids=[item.id for item in recent_observations[:4]],
        expected_track_ids=[track.id for track in entity_tracks if track.status in {"visible", "re-identified", "occluded"}],
        predicted_tags=predicted_tags,
        predicted_summary=predicted_summary,
        stable_elements=stable_elements,
        confidence=round(mean(confidence_sources), 4),
    )


def _updated_track(
    track: EntityTrack,
    *,
    observation: Observation,
    status: str,
    similarity: float,
    label: str | None = None,
) -> EntityTrack:
    reidentified = status == "re-identified"
    visible = status in {"visible", "re-identified"}
    history = [*track.status_history[-7:], status]
    prototype = (
        [
            float(value)
            for value in (
                (np.array(track.prototype_embedding, dtype=np.float32) + np.array(observation.embedding, dtype=np.float32))
                / 2.0
            ).tolist()
        ]
        if track.prototype_embedding
        else [float(value) for value in observation.embedding]
    )
    continuity = clamp_unit((track.continuity_score * 0.6) + (similarity * 0.4))
    persistence = clamp_unit((track.persistence_confidence * 0.7) + (1.0 if visible else 0.55) * 0.3)
    return track.model_copy(
        update={
            "label": label or track.label,
            "status": status,
            "last_seen_at": observation.created_at,
            "last_observation_id": observation.id,
            "observations": [*track.observations[-11:], observation.id],
            "visibility_streak": track.visibility_streak + 1 if visible else 0,
            "occlusion_count": track.occlusion_count + (1 if status == "occluded" else 0),
            "reidentification_count": track.reidentification_count + (1 if reidentified else 0),
            "persistence_confidence": persistence,
            "continuity_score": continuity,
            "last_similarity": similarity,
            "prototype_embedding": prototype,
            "status_history": history,
        }
    )


def update_entity_tracks(
    observation: Observation,
    previous_state: SceneState | None,
    existing_tracks: list[EntityTrack],
) -> tuple[list[EntityTrack], PersistenceSignal]:
    labels = extract_entity_labels(observation)
    tracks = [track.model_copy() for track in existing_tracks]
    matched_ids: set[str] = set()
    visible_ids: list[str] = []
    recovered_ids: list[str] = []
    violated_ids: list[str] = []

    for label in labels:
        exact = next(
            (
                track
                for track in tracks
                if track.id not in matched_ids and normalize_token(track.label) == normalize_token(label)
            ),
            None,
        )
        fuzzy = None
        if exact is None:
            fuzzy = next(
                (
                    track
                    for track in tracks
                    if track.id not in matched_ids
                    and track.prototype_embedding
                    and cosine_similarity(track.prototype_embedding, observation.embedding) >= 0.92
                ),
                None,
            )
        match = exact or fuzzy
        if match is None:
            track = EntityTrack(
                id=f"trk_{uuid4().hex[:12]}",
                session_id=observation.session_id,
                label=label,
                status="visible",
                first_seen_at=observation.created_at,
                last_seen_at=observation.created_at,
                first_observation_id=observation.id,
                last_observation_id=observation.id,
                observations=[observation.id],
                visibility_streak=1,
                persistence_confidence=0.62,
                continuity_score=0.55,
                last_similarity=1.0,
                prototype_embedding=observation.embedding,
                status_history=["visible"],
                metadata={"source": "world-model"},
            )
            tracks.append(track)
            matched_ids.add(track.id)
            visible_ids.append(track.id)
            continue

        similarity = cosine_similarity(match.prototype_embedding or observation.embedding, observation.embedding)
        status = "re-identified" if match.status in {"occluded", "disappeared", "violated prediction"} else "visible"
        updated = _updated_track(match, observation=observation, status=status, similarity=similarity, label=label)
        tracks[tracks.index(match)] = updated
        matched_ids.add(updated.id)
        visible_ids.append(updated.id)
        if status == "re-identified":
            recovered_ids.append(updated.id)

    occluded_ids: list[str] = []
    disappeared_ids: list[str] = []
    for index, track in enumerate(list(tracks)):
        if track.id in matched_ids:
            continue
        age_s = max((observation.created_at - track.last_seen_at).total_seconds(), 0.0)
        predicted = previous_state and track.id in previous_state.prediction_window.expected_track_ids
        if age_s <= 18:
            status = "occluded"
            updated = _updated_track(track, observation=observation, status=status, similarity=track.last_similarity or 0.0)
            occluded_ids.append(updated.id)
        else:
            status = "violated prediction" if predicted else "disappeared"
            updated = _updated_track(track, observation=observation, status=status, similarity=track.last_similarity or 0.0)
            if status == "violated prediction":
                violated_ids.append(updated.id)
            else:
                disappeared_ids.append(updated.id)
        tracks[index] = updated

    expected = max(len(previous_state.prediction_window.expected_track_ids), 1) if previous_state else max(len(visible_ids), 1)
    persistence_confidence = clamp_unit((len(visible_ids) + 0.7 * len(occluded_ids)) / expected)
    signal = PersistenceSignal(
        visible_track_ids=visible_ids,
        occluded_track_ids=occluded_ids,
        recovered_track_ids=recovered_ids,
        disappeared_track_ids=disappeared_ids,
        violated_track_ids=violated_ids,
        persistence_confidence=round(persistence_confidence, 4),
    )
    tracks.sort(key=lambda item: item.last_seen_at, reverse=True)
    return tracks, signal


def build_scene_state(
    observation: Observation,
    hits: list[SearchHit],
    previous_state: SceneState | None,
    recent_observations: list[Observation],
    existing_tracks: list[EntityTrack],
) -> tuple[SceneState, list[EntityTrack]]:
    metadata = _metadata_dict(observation)
    observed_elements = extract_observed_elements(observation)
    prediction_window = build_prediction_window(observation, previous_state, recent_observations, existing_tracks)
    tracks, persistence_signal = update_entity_tracks(observation, previous_state, existing_tracks)
    proposal_boxes = _metadata_boxes(observation)
    primary_object_label = str(metadata.get("primary_object_label") or "").strip() or None

    predicted = prediction_window.predicted_tags
    stable_elements = [element for element in observed_elements if element in predicted]
    new_elements = [f"new:{element}" for element in observed_elements if element not in predicted]
    lost_elements = [f"lost:{element}" for element in predicted if element not in observed_elements]
    changed_elements = new_elements + lost_elements

    predicted_embedding = mean_embedding(recent_observations[:4], observation.embedding)
    prediction_consistency = cosine_similarity(observation.embedding, predicted_embedding)
    expected_span = max(len(set(predicted) | set(observed_elements)), 1)
    predicted_support = clamp_unit(len(stable_elements) / expected_span)
    nearest_memory_score = clamp_unit(hits[0].score) if hits else 0.0
    continuity_signal = ContinuitySignal(
        stable_elements=stable_elements,
        changed_elements=changed_elements,
        continuity_score=0.0,
        predicted_support=round(predicted_support, 4),
        nearest_memory_score=round(nearest_memory_score, 4),
    )
    temporal_continuity_score = clamp_unit(
        (prediction_consistency * 0.42)
        + (predicted_support * 0.28)
        + (persistence_signal.persistence_confidence * 0.2)
        + (nearest_memory_score * 0.1)
    )
    continuity_signal.continuity_score = round(temporal_continuity_score, 4)

    if persistence_signal.recovered_track_ids:
        occlusion_recovery_score = clamp_unit(
            len(persistence_signal.recovered_track_ids)
            / max(len(persistence_signal.occluded_track_ids) + len(persistence_signal.recovered_track_ids), 1)
        )
    elif persistence_signal.occluded_track_ids:
        occlusion_recovery_score = 0.35
    else:
        occlusion_recovery_score = 0.0

    metrics = WorldModelMetrics(
        prediction_consistency=round(prediction_consistency, 4),
        surprise_score=round(clamp_unit(1.0 - prediction_consistency), 4),
        temporal_continuity_score=round(temporal_continuity_score, 4),
        persistence_confidence=round(persistence_signal.persistence_confidence, 4),
        occlusion_recovery_score=round(occlusion_recovery_score, 4),
        continuity_signal=continuity_signal,
        persistence_signal=persistence_signal,
    )
    scene_state = SceneState(
        id=f"ws_{observation.id}",
        session_id=observation.session_id,
        created_at=observation.created_at,
        observation_id=observation.id,
        previous_world_state_id=previous_state.id if previous_state else None,
        nearest_memory_observation_id=hits[0].observation_id if hits else None,
        primary_object_label=primary_object_label,
        proposal_boxes=proposal_boxes,
        entity_track_ids=[track.id for track in tracks],
        persisted_track_ids=persistence_signal.visible_track_ids + persistence_signal.recovered_track_ids,
        occluded_track_ids=persistence_signal.occluded_track_ids,
        observed_elements=observed_elements,
        stable_elements=stable_elements,
        changed_elements=changed_elements,
        predicted_state_summary=prediction_window.predicted_summary,
        observed_state_summary=observation.summary or "Observed scene captured",
        prediction_window=prediction_window,
        metrics=metrics,
        metadata={
            "proof_mode": "hybrid",
            "nearest_memory_score": round(nearest_memory_score, 4),
            "primary_object_label": primary_object_label,
            "proposal_boxes": [box.model_dump(mode="json") for box in proposal_boxes],
            "summary_candidates": metadata.get("summary_candidates", []),
            "summary_source": metadata.get("summary_source"),
        },
    )
    return scene_state, tracks


def _summary_overlap(left: Observation, right: Observation) -> float:
    left_tokens = set(text_tokens(left.summary) + [normalize_token(tag) for tag in left.tags])
    right_tokens = set(text_tokens(right.summary) + [normalize_token(tag) for tag in right.tags])
    if not left_tokens and not right_tokens:
        return 0.0
    return clamp_unit(len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1))


def _surprise_separation(scene_states: list[SceneState]) -> float:
    if not scene_states:
        return 0.0
    changed = [state.metrics.surprise_score for state in scene_states if state.changed_elements]
    stable = [state.metrics.surprise_score for state in scene_states if not state.changed_elements]
    if changed and stable:
        return clamp_unit(mean(changed) - mean(stable) + 0.5)
    if changed:
        return clamp_unit(mean(changed))
    return 0.0


def build_baseline_comparison(observations: list[Observation], scene_states: list[SceneState]) -> BaselineComparison:
    if not observations:
        return BaselineComparison(summary="No observations available for comparison.")

    adjacent_pairs = list(zip(observations, observations[1:]))
    caption_continuity = mean([_summary_overlap(left, right) for left, right in adjacent_pairs]) if adjacent_pairs else 0.0
    embedding_continuity = mean(
        [cosine_similarity(left.embedding, right.embedding) for left, right in adjacent_pairs]
    ) if adjacent_pairs else 0.0

    jepa_continuity = mean([state.metrics.temporal_continuity_score for state in scene_states]) if scene_states else 0.0
    jepa_persistence = mean([state.metrics.persistence_confidence for state in scene_states]) if scene_states else 0.0
    jepa_surprise = _surprise_separation(scene_states)

    frame_captioning = BaselineModeScore(
        continuity=round(caption_continuity, 4),
        persistence=round(clamp_unit(caption_continuity * 0.85), 4),
        surprise_separation=round(clamp_unit(1.0 - caption_continuity), 4),
    )
    frame_captioning.composite = round(
        clamp_unit(
            (frame_captioning.continuity * 0.45)
            + (frame_captioning.persistence * 0.35)
            + (frame_captioning.surprise_separation * 0.2)
        ),
        4,
    )

    embedding_retrieval = BaselineModeScore(
        continuity=round(embedding_continuity, 4),
        persistence=round(clamp_unit((embedding_continuity * 0.75) + 0.1), 4),
        surprise_separation=round(clamp_unit(1.0 - embedding_continuity), 4),
    )
    embedding_retrieval.composite = round(
        clamp_unit(
            (embedding_retrieval.continuity * 0.45)
            + (embedding_retrieval.persistence * 0.35)
            + (embedding_retrieval.surprise_separation * 0.2)
        ),
        4,
    )

    jepa_hybrid = BaselineModeScore(
        continuity=round(jepa_continuity, 4),
        persistence=round(jepa_persistence, 4),
        surprise_separation=round(jepa_surprise, 4),
    )
    jepa_hybrid.composite = round(
        clamp_unit(
            (jepa_hybrid.continuity * 0.45)
            + (jepa_hybrid.persistence * 0.35)
            + (jepa_hybrid.surprise_separation * 0.2)
        ),
        4,
    )

    ranking = {
        "jepa_hybrid": jepa_hybrid.composite,
        "frame_captioning": frame_captioning.composite,
        "embedding_retrieval": embedding_retrieval.composite,
    }
    winner = max(ranking, key=ranking.get)
    if winner == "jepa_hybrid":
        summary = "Hybrid JEPA mode leads on continuity and persistence across the evaluated sequence."
    else:
        summary = f"{winner.replace('_', ' ')} currently leads; the sequence likely lacks enough temporal change to expose persistence gains."
    return BaselineComparison(
        winner=winner,
        jepa_hybrid=jepa_hybrid,
        frame_captioning=frame_captioning,
        embedding_retrieval=embedding_retrieval,
        summary=summary,
    )


def build_challenge_run(
    *,
    session_id: str,
    challenge_set: str,
    proof_mode: str,
    observations: list[Observation],
    scene_states: list[SceneState],
) -> ChallengeRun:
    comparison = build_baseline_comparison(observations, scene_states)
    success_criteria = {
        "surprise_increases_on_change": any(
            state.changed_elements and state.metrics.surprise_score >= 0.45 for state in scene_states
        ),
        "continuity_survives_occlusion": any(
            state.metrics.persistence_signal.occluded_track_ids and state.metrics.temporal_continuity_score >= 0.45
            for state in scene_states
        ),
        "identity_recovers_after_reappearance": any(
            state.metrics.persistence_signal.recovered_track_ids for state in scene_states
        ),
        "jepa_hybrid_outperforms_baselines": comparison.winner == "jepa_hybrid",
    }
    met = sum(1 for value in success_criteria.values() if value)
    summary = (
        f"Challenge evaluated over {len(scene_states)} world states; {met}/{len(success_criteria)} JEPA criteria met. "
        f"{comparison.summary}"
    )
    return ChallengeRun(
        id=f"challenge_{uuid4().hex[:12]}",
        session_id=session_id,
        created_at=observations[-1].created_at if observations else datetime.now(timezone.utc),
        challenge_set=challenge_set,
        proof_mode=proof_mode,
        observation_ids=[item.id for item in observations],
        world_state_ids=[item.id for item in scene_states],
        guide_steps=LIVE_CHALLENGE_GUIDE,
        success_criteria=success_criteria,
        baseline_comparison=comparison,
        summary=summary,
    )
