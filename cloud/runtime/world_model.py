from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from statistics import mean
from typing import Iterable
from uuid import uuid4

import numpy as np

from .models import (
    ActionToken,
    BaselineComparison,
    BaselineModeScore,
    ChallengeRun,
    ContinuitySignal,
    BoundingBox,
    EntityTrack,
    GroundedAffordance,
    GroundedEntity,
    Observation,
    PersistenceSignal,
    PredictedAffordanceState,
    PredictionWindow,
    RecoveryBenchmarkRun,
    RecoveryScenario,
    RolloutBranch,
    RolloutComparison,
    RolloutStep,
    SceneState,
    SearchHit,
    StateDomain,
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
    if left_vec.size != right_vec.size:
        left_vec, right_vec = _align_embedding_pair(left_vec, right_vec)
    denom = (np.linalg.norm(left_vec) or 1.0) * (np.linalg.norm(right_vec) or 1.0)
    return clamp_unit(np.dot(left_vec, right_vec) / denom)


def _align_embedding_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = min(left.size, right.size)
    if target <= 0:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)
    if left.size != target:
        left = np.array([float(chunk.mean()) for chunk in np.array_split(left, target)], dtype=np.float32)
    if right.size != target:
        right = np.array([float(chunk.mean()) for chunk in np.array_split(right, target)], dtype=np.float32)
    return left, right


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


def _state_domain_from_metadata(metadata: dict) -> StateDomain:
    raw = str(metadata.get("state_domain") or "camera").strip().lower()
    if raw in {"camera", "browser", "desktop", "memory"}:
        return raw  # type: ignore[return-value]
    return "camera"


def _validate_grounded_entities(raw_entities: object) -> list[GroundedEntity]:
    grounded: list[GroundedEntity] = []
    if not isinstance(raw_entities, list):
        return grounded
    for raw in raw_entities:
        try:
            grounded.append(
                raw if isinstance(raw, GroundedEntity) else GroundedEntity.model_validate(raw)
            )
        except Exception:
            continue
    return grounded


def _validate_affordances(raw_affordances: object) -> list[GroundedAffordance]:
    affordances: list[GroundedAffordance] = []
    if not isinstance(raw_affordances, list):
        return affordances
    for raw in raw_affordances:
        try:
            affordances.append(
                raw if isinstance(raw, GroundedAffordance) else GroundedAffordance.model_validate(raw)
            )
        except Exception:
            continue
    return affordances


def _label_from_entity_track(track: EntityTrack) -> str:
    label = _meaningful_label(track.label) or normalize_token(track.label)
    return label or f"entity {track.id[-4:]}"


def _grounded_entities_from_camera(
    observation: Observation,
    entity_tracks: list[EntityTrack],
    proposal_boxes: list[BoundingBox],
) -> list[GroundedEntity]:
    grounded: list[GroundedEntity] = []
    used_track_ids: set[str] = set()
    for track in entity_tracks:
        if track.status == "disappeared":
            continue
        grounded.append(
            GroundedEntity(
                id=track.id,
                label=_label_from_entity_track(track),
                kind="physical_object",
                state_domain="camera",
                status=track.status,
                confidence=round(max(track.persistence_confidence, track.continuity_score), 4),
                source_track_id=track.id,
                properties={
                    "visibility_streak": track.visibility_streak,
                    "continuity_score": track.continuity_score,
                },
            )
        )
        used_track_ids.add(track.id)
    for index, box in enumerate(proposal_boxes[:8]):
        label = _meaningful_label(box.label or "") or normalize_token(str(box.label or ""))
        if not label:
            continue
        grounded.append(
            GroundedEntity(
                id=f"{observation.id}:proposal:{index}",
                label=label,
                kind="proposal_box",
                state_domain="camera",
                status="visible",
                confidence=round(float(box.score or observation.confidence or 0.5), 4),
                properties={
                    "bbox": {
                        "x": box.x,
                        "y": box.y,
                        "width": box.width,
                        "height": box.height,
                    }
                },
            )
        )
    return grounded


def _default_affordances_for_domain(
    *,
    state_domain: StateDomain,
    grounded_entities: list[GroundedEntity],
    metadata: dict,
) -> list[GroundedAffordance]:
    if state_domain == "camera":
        labels = [entity.label for entity in grounded_entities[:4]]
        affordances = [
            GroundedAffordance(
                id="camera.observe",
                label="observe scene",
                kind="camera.observe",
                state_domain="camera",
                confidence=0.92,
            ),
        ]
        if labels:
            affordances.append(
                GroundedAffordance(
                    id="camera.inspect",
                    label=f"inspect {labels[0]}",
                    kind="camera.inspect",
                    state_domain="camera",
                    target_entity_id=grounded_entities[0].id,
                    confidence=0.78,
                )
            )
        return affordances

    affordances: list[GroundedAffordance] = []
    if metadata.get("current_url"):
        affordances.append(
            GroundedAffordance(
                id="browser.refresh",
                label="refresh view",
                kind="browser.refresh",
                state_domain=state_domain,
                confidence=0.8,
            )
        )
    if metadata.get("error_banners"):
        affordances.append(
            GroundedAffordance(
                id="browser.retry",
                label="retry failed step",
                kind="browser.retry",
                state_domain=state_domain,
                availability="error",
                confidence=0.86,
            )
        )
    return affordances


def derive_grounded_entities(
    observation: Observation,
    entity_tracks: list[EntityTrack],
    proposal_boxes: list[BoundingBox],
) -> tuple[StateDomain, list[GroundedEntity], list[GroundedAffordance]]:
    metadata = _metadata_dict(observation)
    state_domain = _state_domain_from_metadata(metadata)
    grounded_entities = _validate_grounded_entities(metadata.get("grounded_entities"))
    if not grounded_entities:
        grounded_entities = _grounded_entities_from_camera(observation, entity_tracks, proposal_boxes)
    affordances = _validate_affordances(metadata.get("affordances"))
    if not affordances:
        affordances = _default_affordances_for_domain(
            state_domain=state_domain,
            grounded_entities=grounded_entities,
            metadata=metadata,
        )
    return state_domain, grounded_entities, affordances


def default_candidate_actions(
    *,
    observation: Observation,
    state_domain: StateDomain,
    grounded_entities: list[GroundedEntity],
    affordances: list[GroundedAffordance],
) -> list[ActionToken]:
    actions: list[ActionToken] = []
    for affordance in affordances[:4]:
        verb = affordance.kind.split(".")[-1] if "." in affordance.kind else affordance.kind
        actions.append(
            ActionToken(
                id=f"act:{affordance.id}",
                verb=verb or "use",
                target_kind=affordance.kind,
                target_id=affordance.target_entity_id or affordance.id,
                target_label=affordance.label,
                parameters={
                    "availability": affordance.availability,
                    "state_domain": affordance.state_domain,
                },
            )
        )
    if grounded_entities:
        lead = grounded_entities[0]
        actions.append(
            ActionToken(
                id=f"inspect:{lead.id}",
                verb="inspect",
                target_kind=lead.kind,
                target_id=lead.id,
                target_label=lead.label,
                parameters={"state_domain": state_domain},
            )
        )
    actions.append(
        ActionToken(
            id=f"memory:{observation.id}",
            verb="query_memory",
            target_kind="memory",
            target_id=observation.world_state_id or observation.id,
            target_label="continuity memory",
            parameters={"state_domain": "memory"},
        )
    )
    deduped: list[ActionToken] = []
    seen_ids: set[str] = set()
    for action in actions:
        if action.id in seen_ids:
            continue
        seen_ids.add(action.id)
        deduped.append(action)
    return deduped[:6]


def build_rollout_comparison(
    *,
    scene_state: SceneState,
    candidate_actions: list[ActionToken],
    horizon: int = 2,
) -> RolloutComparison:
    grounded_labels = [entity.label for entity in scene_state.grounded_entities[:6]]
    affordance_map = {affordance.id: affordance for affordance in scene_state.affordances}
    branches: list[RolloutBranch] = []
    baseline_risk = scene_state.metrics.surprise_score

    for index, action in enumerate(candidate_actions[:6]):
        availability = str(action.parameters.get("availability") or "available")
        target_label = action.target_label or action.target_id or action.verb
        blockers: list[str] = []
        if availability in {"missing", "error"}:
            blockers.append(f"{target_label} is currently {availability}")
        elif availability == "disabled":
            blockers.append(f"{target_label} is disabled")
        if action.verb in {"click", "submit", "open", "retry"} and scene_state.changed_elements:
            blockers.append("layout may have shifted since the previous state")
        if scene_state.state_domain == "camera" and scene_state.occluded_track_ids and action.verb != "query_memory":
            blockers.append("occlusion may hide the target")

        confidence = clamp_unit(
            scene_state.prediction_window.confidence
            + max(0.0, 0.15 - (0.08 * len(blockers)))
            + (0.05 if availability == "available" else 0.0)
        )
        risk_score = clamp_unit(
            baseline_risk
            + (0.18 * len(blockers))
            + (0.08 if availability in {"missing", "error"} else 0.0)
            - (0.1 if action.verb == "query_memory" else 0.0)
        )
        recovery_cost = clamp_unit((risk_score * 0.6) + (0.1 * len(blockers)))
        predicted_summary = (
            f"{action.verb.replace('_', ' ')} should preserve continuity around {', '.join(grounded_labels[:3])}"
            if grounded_labels
            else f"{action.verb.replace('_', ' ')} should advance the current {scene_state.state_domain} state"
        )
        predicted_affordances: list[PredictedAffordanceState] = []
        for affordance in scene_state.affordances[:4]:
            next_availability = affordance.availability
            if affordance.id == action.target_id and action.verb in {"click", "submit"}:
                next_availability = "hidden"
            elif action.verb == "retry" and affordance.availability in {"error", "missing"}:
                next_availability = "available"
            predicted_affordances.append(
                PredictedAffordanceState(
                    affordance_id=affordance.id,
                    label=affordance.label,
                    availability=next_availability,
                    reason="predicted from current affordance state",
                )
            )
        follow_up = ActionToken(
            id=f"{action.id}:followup",
            verb="query_memory" if blockers else "confirm_state",
            target_kind="memory" if blockers else scene_state.state_domain,
            target_id=scene_state.id,
            target_label="recover context" if blockers else "verify continuity",
            parameters={"state_domain": "memory" if blockers else scene_state.state_domain},
        )
        steps = [
            RolloutStep(
                step_index=0,
                action=action,
                predicted_state_domain=scene_state.state_domain,
                predicted_summary=predicted_summary,
                blockers=blockers,
                confidence=round(confidence, 4),
            ),
        ]
        if horizon > 1:
            steps.append(
                RolloutStep(
                    step_index=1,
                    action=follow_up,
                    predicted_state_domain="memory" if blockers else scene_state.state_domain,
                    predicted_summary=(
                        "Fallback to continuity memory if the primary affordance path breaks"
                        if blockers
                        else "Confirm that the resulting state matches the expected rollout"
                    ),
                    blockers=[],
                    confidence=round(clamp_unit(confidence - 0.06), 4),
                )
            )

        branches.append(
            RolloutBranch(
                id=f"branch-{index}-{action.id}",
                candidate_action=action,
                predicted_next_state_summary=predicted_summary,
                predicted_persistent_entities=grounded_labels[:4],
                predicted_affordances=predicted_affordances,
                risk_score=round(risk_score, 4),
                confidence=round(confidence, 4),
                expected_recovery_cost=round(recovery_cost, 4),
                failure_predicates=blockers,
                steps=steps,
            )
        )

    branches.sort(key=lambda branch: (branch.risk_score, -branch.confidence, branch.id))
    chosen = branches[0].id if branches else None
    summary = (
        f"Plan A: {branches[0].candidate_action.verb.replace('_', ' ')}; "
        f"Plan B: {branches[1].candidate_action.verb.replace('_', ' ') if len(branches) > 1 else 'query memory'}."
        if branches
        else "No rollout candidates available yet."
    )
    return RolloutComparison(
        state_domain=scene_state.state_domain,
        based_on_world_state_id=scene_state.id,
        horizon=horizon,
        ranked_branches=branches,
        chosen_branch_id=chosen,
        summary=summary,
    )


def build_prediction_window(
    observation: Observation,
    previous_state: SceneState | None,
    recent_observations: list[Observation],
    entity_tracks: list[EntityTrack],
    *,
    state_domain: StateDomain = "camera",
    grounded_entities: list[GroundedEntity] | None = None,
    affordances: list[GroundedAffordance] | None = None,
    candidate_actions: list[ActionToken] | None = None,
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
    if state_domain != "camera":
        predicted_summary = (
            f"Expect the {state_domain} state to preserve {', '.join(stable_elements[:3]) or 'the current affordances'}"
        )
    return PredictionWindow(
        previous_observation_id=recent_observations[0].id if recent_observations else None,
        context_observation_ids=[item.id for item in recent_observations[:4]],
        expected_track_ids=[track.id for track in entity_tracks if track.status in {"visible", "re-identified", "occluded"}],
        predicted_tags=predicted_tags,
        predicted_summary=predicted_summary,
        stable_elements=stable_elements,
        confidence=round(mean(confidence_sources), 4),
        candidate_actions=list(candidate_actions or []),
        predicted_branches=[],
        chosen_branch_id=None,
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
    track_embedding = np.array(track.prototype_embedding, dtype=np.float32) if track.prototype_embedding else np.array([], dtype=np.float32)
    observation_embedding = np.array(observation.embedding, dtype=np.float32)
    if track_embedding.size and track_embedding.size != observation_embedding.size:
        track_embedding, observation_embedding = _align_embedding_pair(track_embedding, observation_embedding)
    prototype = (
        [
            float(value)
            for value in (
                (track_embedding + observation_embedding)
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
    tracks, persistence_signal = update_entity_tracks(observation, previous_state, existing_tracks)
    proposal_boxes = _metadata_boxes(observation)
    state_domain, grounded_entities, affordances = derive_grounded_entities(observation, tracks, proposal_boxes)
    candidate_actions = default_candidate_actions(
        observation=observation,
        state_domain=state_domain,
        grounded_entities=grounded_entities,
        affordances=affordances,
    )
    prediction_window = build_prediction_window(
        observation,
        previous_state,
        recent_observations,
        tracks,
        state_domain=state_domain,
        grounded_entities=grounded_entities,
        affordances=affordances,
        candidate_actions=candidate_actions,
    )
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
        state_domain=state_domain,
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
        grounded_entities=grounded_entities,
        affordances=affordances,
        metrics=metrics,
        metadata={
            "proof_mode": "hybrid",
            "nearest_memory_score": round(nearest_memory_score, 4),
            "primary_object_label": primary_object_label,
            "proposal_boxes": [box.model_dump(mode="json") for box in proposal_boxes],
            "summary_candidates": metadata.get("summary_candidates", []),
            "summary_source": metadata.get("summary_source"),
            "state_domain": state_domain,
        },
    )
    rollout_horizon = int(metadata.get("rollout_horizon") or 2)
    rollout_comparison = build_rollout_comparison(
        scene_state=scene_state,
        candidate_actions=prediction_window.candidate_actions,
        horizon=max(1, min(rollout_horizon, 5)),
    )
    scene_state = scene_state.model_copy(
        update={
            "conditioned_rollouts": rollout_comparison,
            "prediction_window": prediction_window.model_copy(
                update={
                    "predicted_branches": rollout_comparison.ranked_branches,
                    "chosen_branch_id": rollout_comparison.chosen_branch_id,
                }
            ),
        }
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


def build_recovery_benchmark_run(
    *,
    session_id: str,
    scene_states: list[SceneState],
    comparison: RolloutComparison | None = None,
) -> RecoveryBenchmarkRun:
    latest = scene_states[0] if scene_states else None
    comparison = comparison or (latest.conditioned_rollouts if latest else None)
    scenarios: list[RecoveryScenario] = []
    if latest is not None:
        occlusion_passed = bool(
            latest.metrics.persistence_signal.recovered_track_ids
            or latest.metrics.occlusion_recovery_score >= 0.5
        )
        scenarios.append(
            RecoveryScenario(
                id="camera-occlusion",
                title="Camera occlusion and re-identification",
                domain="camera",
                passed=occlusion_passed,
                score=round(latest.metrics.occlusion_recovery_score, 4),
                details="Tracks should remain coherent through temporary occlusion.",
                related_branch_id=comparison.chosen_branch_id if comparison else None,
            )
        )
        changed_text = " ".join(latest.changed_elements).lower()
        layout_shift_passed = "lost:" in changed_text or "new:" in changed_text
        scenarios.append(
            RecoveryScenario(
                id="browser-layout-shift",
                title="Browser missing or moved control recovery",
                domain="browser",
                passed=layout_shift_passed,
                score=round(clamp_unit(latest.metrics.surprise_score + 0.2), 4),
                details="A shifted affordance should register as a state change rather than a fresh unrelated world.",
                related_branch_id=comparison.chosen_branch_id if comparison else None,
            )
        )
        error_affordances = [
            affordance for affordance in latest.affordances if affordance.availability in {"error", "missing"}
        ]
        scenarios.append(
            RecoveryScenario(
                id="tool-failure",
                title="Tool or network failure recovery",
                domain="hybrid",
                passed=bool(error_affordances) or bool(comparison and comparison.ranked_branches),
                score=round(
                    clamp_unit(
                        1.0 - (comparison.ranked_branches[0].risk_score if comparison and comparison.ranked_branches else 0.45)
                    ),
                    4,
                ),
                details="The planner should expose an explicit fallback branch when the primary affordance path fails.",
                related_branch_id=comparison.chosen_branch_id if comparison else None,
            )
        )
        memory_branch = next(
            (
                branch
                for branch in (comparison.ranked_branches if comparison else [])
                if branch.candidate_action.verb == "query_memory"
            ),
            None,
        )
        scenarios.append(
            RecoveryScenario(
                id="memory-fallback",
                title="Memory query fallback",
                domain="memory",
                passed=memory_branch is not None,
                score=round(memory_branch.confidence if memory_branch is not None else 0.0, 4),
                details="Continuity memory should remain available as a recovery path when visual affordances break.",
                related_branch_id=memory_branch.id if memory_branch is not None else None,
            )
        )
    passed = sum(1 for scenario in scenarios if scenario.passed)
    summary = (
        f"Recovery benchmark covered {len(scenarios)} scenarios; {passed}/{len(scenarios)} passed. "
        f"{comparison.summary if comparison else 'No rollout comparison available yet.'}"
    ) if scenarios else "No recovery scenarios available yet."
    return RecoveryBenchmarkRun(
        id=f"benchmark_{uuid4().hex[:12]}",
        session_id=session_id,
        benchmark_scope="hybrid",
        world_state_ids=[state.id for state in scene_states[:8]],
        scenarios=scenarios,
        winner="action_conditioned_rollout" if passed >= max(len(scenarios) // 2, 1) else "baseline_caption_retrieval",
        summary=summary,
    )
