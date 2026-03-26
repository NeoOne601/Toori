from __future__ import annotations

import base64
import io
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .atlas import EpistemicAtlas
from .config import resolve_data_dir
from .events import EventBus
from .models import (
    AnalyzeRequest,
    AnalyzeResponse,
    Answer,
    ChallengeEvaluateRequest,
    ChallengeRun,
    JEPAForecastResponse,
    JEPATick,
    LivingLensTickRequest,
    LivingLensTickResponse,
    Observation,
    ObservationsResponse,
    ProofReportResponse,
    ProviderHealthResponse,
    QueryRequest,
    QueryResponse,
    ReasoningTraceEntry,
    RuntimeSettings,
    SceneState,
    TalkerEvent,
    WorldStateResponse,
)
from .providers import ProviderRegistry
from .storage import ObservationStore
from .talker import SelectiveTalker
from .world_model import build_baseline_comparison, build_challenge_run, build_object_summary, build_scene_state


class RuntimeContainer:
    def __init__(self, data_dir: str | Path | None = None) -> None:
        from cloud.jepa_service.engine import ImmersiveJEPAEngine, JEPAEngine
        
        self.data_dir = resolve_data_dir(data_dir)
        self.store = ObservationStore(self.data_dir)
        self.providers = ProviderRegistry()
        self.events = EventBus()
        self.engine = JEPAEngine()
        self._immersive_engine_factory = ImmersiveJEPAEngine
        self._immersive_engines: dict[str, ImmersiveJEPAEngine] = {}
        self._jepa_ticks_by_session: dict[str, deque[JEPATick]] = defaultdict(lambda: deque(maxlen=240))
        self._latest_proof_report: Optional[Path] = None
        self.talker = SelectiveTalker()
        self.atlas = EpistemicAtlas()
        self._previous_tracks: list = []

    def get_settings(self) -> RuntimeSettings:
        settings = self.store.load_settings()
        if settings is None:
            raise RuntimeError("Runtime settings are unavailable")
        return settings

    def update_settings(self, settings: RuntimeSettings) -> RuntimeSettings:
        saved = self.store.save_settings(settings)
        self.providers.reset_circuits()
        self.events.publish(
            "provider.changed",
            {"settings": saved.model_dump(mode="json")},
        )
        return saved

    def provider_health(self) -> ProviderHealthResponse:
        return ProviderHealthResponse(providers=self.providers.health_snapshot(self.get_settings()))

    def list_observations(self, *, session_id: Optional[str], limit: int = 50) -> ObservationsResponse:
        return ObservationsResponse(observations=self.store.list_observations(session_id=session_id, limit=limit))

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        response, _, _ = self._analyze_with_world_model(request)
        return response

    def living_lens_tick(self, request: LivingLensTickRequest) -> LivingLensTickResponse:
        response, scene_state, tracks = self._analyze_with_world_model(request, _is_tick=True)
        observations = list(reversed(self.store.recent_observations(session_id=request.session_id, limit=8)))
        history = list(reversed(self.store.recent_scene_states(session_id=request.session_id, limit=8)))
        baseline = build_baseline_comparison(observations, history) if request.proof_mode in {"both", "baseline"} else None
        frame = np.asarray(Image.open(response.observation.image_path).convert("RGB"), dtype=np.uint8)
        immersive_engine = self._engine_for_session(request.session_id)
        jepa_tick = immersive_engine.tick(
            frame,
            session_id=request.session_id,
            observation_id=response.observation.id,
        )
        self._jepa_ticks_by_session[request.session_id].append(jepa_tick)

        immersive_tracks = jepa_tick.entity_tracks[:12]
        self.store.save_entity_tracks(immersive_tracks)
        talker_event = self._talker_event_from_jepa(jepa_tick, immersive_tracks)
        self._previous_tracks = [track.model_copy() for track in immersive_tracks]

        self.atlas.update(
            entity_tracks=immersive_tracks,
            scene_state=scene_state,
            energy_map=jepa_tick.energy_map,
        )

        scene_state.metrics.surprise_score = round(float(min(jepa_tick.mean_energy, 1.0)), 4)
        scene_state.metrics.prediction_consistency = round(float(max(1.0 - jepa_tick.mean_energy, 0.0)), 4)

        threshold = float(getattr(immersive_engine, "_mu_E", 0.0) + (2.0 * jepa_tick.energy_std))
        should_talk = bool(jepa_tick.talker_event)
        self.events.publish(
            "jepa.energy_map",
            {
                "grid": [14, 14],
                "values": np.asarray(jepa_tick.energy_map, dtype=np.float32).ravel().tolist(),
                "mean_energy": jepa_tick.mean_energy,
                "threshold": threshold,
                "should_talk": should_talk,
                "sigreg_loss": jepa_tick.sigreg_loss,
            },
        )
        self.events.publish(
            "jepa_tick",
            {"payload": jepa_tick.to_payload().model_dump(mode="json")},
        )
        if talker_event is not None:
            self.events.publish("jepa.talker_event", talker_event.model_dump(mode="json"))

        return LivingLensTickResponse(
            **response.model_dump(),
            scene_state=scene_state,
            entity_tracks=immersive_tracks,
            baseline_comparison=baseline,
            talker_event=talker_event,
            jepa_tick=jepa_tick.to_payload(),
        )

    def get_world_state(self, session_id: str) -> WorldStateResponse:
        history = self.store.recent_scene_states(session_id=session_id, limit=24)
        tracks = self.store.list_entity_tracks(session_id=session_id, limit=32)
        challenges = self.store.recent_challenge_runs(session_id=session_id, limit=8)
        return WorldStateResponse(
            session_id=session_id,
            current=history[0] if history else None,
            history=history,
            entity_tracks=tracks,
            challenges=challenges,
            atlas=self.atlas.to_dict(),
        )

    def forecast_jepa(self, session_id: str, k: int) -> JEPAForecastResponse:
        engine = self._immersive_engines.get(session_id)
        if engine is None:
            return JEPAForecastResponse(session_id=session_id, k=k, prediction=[], forecast_error=None, ready=False)
        prediction, forecast_error = engine.forecast_last_state(k)
        return JEPAForecastResponse(
            session_id=session_id,
            k=k,
            prediction=np.asarray(prediction, dtype=np.float32).tolist(),
            forecast_error=forecast_error,
            ready=bool(prediction.size),
        )

    def generate_proof_report(self, session_id: str, chart_b64: str | None = None) -> ProofReportResponse:
        from .proof_report import generate_proof_report

        ticks = list(self._jepa_ticks_by_session.get(session_id, deque()))
        path = generate_proof_report(ticks=ticks, session_id=session_id, chart_b64=chart_b64)
        self._latest_proof_report = Path(path)
        return ProofReportResponse(session_id=session_id, path=str(path), generated=True)

    def latest_proof_report(self) -> Optional[Path]:
        return self._latest_proof_report

    def _engine_for_session(self, session_id: str):
        if session_id not in self._immersive_engines:
            self._immersive_engines[session_id] = self._immersive_engine_factory()
        return self._immersive_engines[session_id]

    def _talker_event_from_jepa(self, tick: JEPATick, entity_tracks: list) -> TalkerEvent | None:
        if not tick.talker_event:
            return None
        event_type = tick.talker_event
        names = [track.label for track in entity_tracks[:2]]
        name = names[0] if names else "entity"
        descriptions = {
            "ENTITY_APPEARED": "Something new just entered",
            "ENTITY_DISAPPEARED": f"{name} left - I'll remember it",
            "OCCLUSION_START": f"I'm still tracking {name} even though I can't see it",
            "OCCLUSION_END": f"{name} came back - exactly where I expected",
            "PREDICTION_VIOLATION": "That wasn't supposed to happen",
            "SCENE_STABLE": "Everything is as expected",
        }
        return TalkerEvent(
            event_type=event_type,  # type: ignore[arg-type]
            confidence=max(0.0, min(1.0, tick.mean_energy)),
            entity_ids=[track.id for track in entity_tracks[:4]],
            energy_summary=tick.mean_energy,
            description=descriptions.get(event_type, "Unexpected change detected"),
        )

    def _emit_reasoning_latency(self, trace: list[ReasoningTraceEntry]) -> None:
        for entry in trace:
            if entry.provider not in {"ollama", "mlx"}:
                continue
            if not entry.attempted or entry.latency_ms is None:
                continue
            self.events.publish(
                "llm_latency",
                {
                    "provider": entry.provider,
                    "ms": float(entry.latency_ms),
                    "success": entry.success,
                },
            )

    def evaluate_challenge(self, request: ChallengeEvaluateRequest) -> ChallengeRun:
        if request.observation_ids:
            observations = self.store.get_observations_by_ids(request.observation_ids)
        else:
            observations = list(
                reversed(self.store.recent_observations(session_id=request.session_id, limit=request.limit))
            )
        world_states = [
            self.store.get_scene_state(observation.world_state_id)
            for observation in observations
            if observation.world_state_id
        ]
        scene_states = [state for state in world_states if state is not None]
        if not scene_states:
            scene_states = list(
                reversed(self.store.recent_scene_states(session_id=request.session_id, limit=request.limit))
            )
        challenge = build_challenge_run(
            session_id=request.session_id,
            challenge_set=request.challenge_set,
            proof_mode=request.proof_mode,
            observations=observations,
            scene_states=scene_states,
        )
        self.store.save_challenge_run(challenge)
        self.events.publish(
            "challenge.updated",
            {"challenge": challenge.model_dump(mode="json")},
        )
        return challenge

    def query(self, request: QueryRequest) -> QueryResponse:
        settings = self.get_settings()
        answer: Optional[Answer] = None
        reasoning_trace: list[ReasoningTraceEntry] = []

        if request.image_base64 or request.file_path:
            image_bytes, image = self._load_image(request.image_base64, request.file_path)
            embedding, _, _, _ = self.providers.perceive(settings, image)
            hits = self.store.search_by_vector(
                embedding,
                top_k=request.top_k,
                session_id=request.session_id,
                time_window_s=request.time_window_s,
                exclude_id=None,
            )
            if request.query:
                context = [
                    self.store.get_observation(hit.observation_id)
                    for hit in hits
                    if self.store.get_observation(hit.observation_id) is not None
                ]
                outcome = self.providers.reason(
                    settings,
                    prompt=request.query,
                    image_bytes=image_bytes,
                    image_path=None,
                    context=[item for item in context if item is not None],
                )
                answer = outcome.answer
                reasoning_trace = outcome.trace
                self._emit_reasoning_latency(reasoning_trace)
        else:
            hits = self.store.search_by_text(
                request.query or "",
                top_k=request.top_k,
                session_id=request.session_id,
                time_window_s=request.time_window_s,
            )
            if request.query and hits:
                context = [
                    self.store.get_observation(hit.observation_id)
                    for hit in hits
                    if self.store.get_observation(hit.observation_id) is not None
                ]
                outcome = self.providers.reason(
                    settings,
                    prompt=request.query,
                    image_bytes=None,
                    image_path=None,
                    context=[item for item in context if item is not None],
                )
                answer = outcome.answer
                reasoning_trace = outcome.trace
                self._emit_reasoning_latency(reasoning_trace)

        return QueryResponse(
            hits=hits,
            answer=answer,
            provider_health=self.providers.health_snapshot(settings),
            reasoning_trace=reasoning_trace,
        )

    def _analyze_with_world_model(
        self,
        request: AnalyzeRequest,
        _is_tick: bool = False,
    ) -> tuple[AnalyzeResponse, "SceneState", list]:
        settings = self.get_settings()
        self.store.prune(settings.retention_days)

        image_bytes, image = self._load_image(request.image_base64, request.file_path)
        embedding, provider_name, confidence, provider_metadata = self.providers.perceive(settings, image)
        if provider_name != "basic":
            _, _, basic_metadata = self.providers.basic.perceive(image)
            provider_metadata = {**basic_metadata, **provider_metadata}

        previous = self.store.recent_observations(session_id=request.session_id, limit=1)
        novelty = 1.0
        if previous:
            novelty = max(0.0, 1.0 - self._cosine_similarity(embedding, previous[0].embedding))

        observation = self.store.create_observation(
            image=image,
            raw_bytes=image_bytes,
            embedding=embedding,
            session_id=request.session_id,
            confidence=confidence,
            novelty=novelty,
            source_query=request.query,
            tags=request.tags + self._fallback_tags(provider_metadata),
            providers=[provider_name],
            metadata={"perception": provider_metadata},
        )

        proposal_boxes = self.providers.object_proposals(
            settings,
            image,
            provider_name=provider_name,
        )

        hits = self.store.search_by_vector(
            embedding,
            top_k=request.top_k or settings.top_k,
            session_id=request.session_id,
            exclude_id=observation.id,
            time_window_s=request.time_window_s,
        )

        answer, reasoning_trace = self._maybe_reason(
            settings=settings,
            request=request,
            observation=observation,
            image_bytes=image_bytes,
            _is_tick=_is_tick,
        )
        summary_text, summary_metadata = build_object_summary(
            observation,
            provider_metadata,
            proposal_boxes,
            answer_text=answer.text if answer is not None else None,
            query=request.query,
        )
        if answer is not None:
            observation = self.store.update_observation(
                observation.id,
                summary=summary_text,
                providers=observation.providers + [answer.provider],
                metadata={
                    **observation.metadata,
                    "answer": answer.model_dump(mode="json"),
                    "reasoning_trace": [entry.model_dump(mode="json") for entry in reasoning_trace],
                    "object_proposals": [box.model_dump(mode="json") for box in proposal_boxes],
                    "proposal_boxes": [box.model_dump(mode="json") for box in proposal_boxes],
                    **summary_metadata,
                },
            )
        else:
            observation = self.store.update_observation(
                observation.id,
                summary=summary_text,
                metadata={
                    **observation.metadata,
                    "reasoning_trace": [entry.model_dump(mode="json") for entry in reasoning_trace],
                    "object_proposals": [box.model_dump(mode="json") for box in proposal_boxes],
                    "proposal_boxes": [box.model_dump(mode="json") for box in proposal_boxes],
                    **summary_metadata,
                },
            )

        previous_state = self.store.latest_scene_state(request.session_id)
        recent_observations = [
            item
            for item in self.store.recent_observations(session_id=request.session_id, limit=6)
            if item.id != observation.id
        ]
        existing_tracks = self.store.list_entity_tracks(session_id=request.session_id, limit=32)
        scene_state, entity_tracks = build_scene_state(
            observation=observation,
            hits=hits,
            previous_state=previous_state,
            recent_observations=recent_observations,
            existing_tracks=existing_tracks,
        )
        self.store.save_scene_state(scene_state)
        self.store.save_entity_tracks(entity_tracks)
        observation = self.store.update_observation(
            observation.id,
            world_state_id=scene_state.id,
            metadata={
                **observation.metadata,
                "world_model": scene_state.metrics.model_dump(mode="json"),
                "scene_state_id": scene_state.id,
                "proposal_boxes": [box.model_dump(mode="json") for box in proposal_boxes],
            },
        )

        self.events.publish(
            "observation.created",
            {"observation": observation.model_dump(mode="json")},
        )
        if answer is not None:
            self.events.publish(
                "answer.ready",
                {"observation_id": observation.id, "answer": answer.model_dump(mode="json")},
            )
        self.events.publish(
            "search.ready",
            {
                "observation_id": observation.id,
                "hits": [hit.model_dump(mode="json") for hit in hits],
            },
        )
        self.events.publish(
            "world_state.updated",
            {"scene_state": scene_state.model_dump(mode="json")},
        )
        for track in entity_tracks[:8]:
            self.events.publish(
                "entity_track.updated",
                {"track": track.model_dump(mode="json")},
            )

        response = AnalyzeResponse(
            observation=observation,
            hits=hits,
            answer=answer,
            provider_health=self.providers.health_snapshot(settings),
            reasoning_trace=reasoning_trace,
        )
        return response, scene_state, entity_tracks

    def _maybe_reason(
        self,
        *,
        settings: RuntimeSettings,
        request: AnalyzeRequest,
        observation: Observation,
        image_bytes: bytes,
        _is_tick: bool = False,
    ) -> tuple[Optional[Answer], list[ReasoningTraceEntry]]:
        # During living_lens_tick, suppress reasoning — talker handles output
        if _is_tick and not request.query and request.decode_mode != "force":
            return None, []
        should_reason = request.decode_mode == "force" or bool(request.query)
        should_reason = should_reason or (
            request.decode_mode == "auto" and observation.novelty >= settings.decode_auto_threshold
        )
        if not should_reason:
            return None, []
        context = self.store.recent_observations(session_id=observation.session_id, limit=5)
        prompt = request.query or "Describe the live scene, focus on actionable objects, activities, and visual changes."
        outcome = self.providers.reason(
            settings,
            prompt=prompt,
            image_bytes=image_bytes,
            image_path=observation.image_path,
            context=context,
        )
        self._emit_reasoning_latency(outcome.trace)
        return outcome.answer, outcome.trace

    def _load_image(self, image_base64: Optional[str], file_path: Optional[str]) -> tuple[bytes, Image.Image]:
        if image_base64:
            raw_bytes = base64.b64decode(image_base64)
        elif file_path:
            raw_bytes = Path(file_path).expanduser().read_bytes()
        else:  # pragma: no cover - validated by Pydantic
            raise ValueError("image input missing")
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return raw_bytes, image

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        left_vec = np.array(left, dtype=np.float32)
        right_vec = np.array(right, dtype=np.float32)
        denom = (np.linalg.norm(left_vec) or 1.0) * (np.linalg.norm(right_vec) or 1.0)
        return float(np.dot(left_vec, right_vec) / denom)

    def _fallback_summary(self, observation: Observation, metadata: dict, query: Optional[str]) -> str:
        top_label = str(metadata.get("top_label", "")).replace("_", " ").strip()
        dominant = metadata.get("dominant_color")
        brightness = metadata.get("brightness_label")
        edge = metadata.get("edge_label")
        parts = []
        if top_label:
            parts.append(top_label)
        if dominant:
            parts.append(f"{dominant} dominant")
        if brightness:
            parts.append(brightness)
        if edge:
            parts.append(edge)
        summary = "Local observation"
        if parts:
            summary = f"{' '.join(parts)} scene"
        if query:
            summary = f"{summary}; prompt: {query}"
        return summary

    def _fallback_tags(self, metadata: dict) -> list[str]:
        tags = []
        top_label = str(metadata.get("top_label", "")).replace("_", " ").strip()
        if top_label:
            tags.append(top_label)
        for key in ("dominant_color", "brightness_label", "edge_label"):
            value = metadata.get(key)
            if value:
                tags.append(str(value))
        return tags
