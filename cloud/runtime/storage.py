from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
from PIL import Image

from .config import default_settings
from .models import (
    ChallengeRun,
    EntityTrack,
    Observation,
    ProviderConfig,
    RuntimeSettings,
    SceneState,
    SearchHit,
)


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ObservationStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.frames_dir = self.data_dir / "frames"
        self.thumbs_dir = self.data_dir / "thumbs"
        self.db_path = self.data_dir / "runtime.sqlite3"
        self._lock = threading.Lock()
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()
        if self.load_settings() is None:
            self.save_settings(default_settings())

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    id TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS observations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    world_state_id TEXT,
                    image_path TEXT NOT NULL,
                    thumbnail_path TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    embedding_json TEXT NOT NULL,
                    summary TEXT,
                    source_query TEXT,
                    tags_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    novelty REAL NOT NULL,
                    providers_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS scene_states (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    observation_id TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS entity_tracks (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS challenge_runs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_observations_session_created
                    ON observations(session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_scene_states_session_created
                    ON scene_states(session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_entity_tracks_session_updated
                    ON entity_tracks(session_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_challenge_runs_session_created
                    ON challenge_runs(session_id, created_at DESC);
                """
            )
            self._migrate_observations(connection)
            connection.commit()

    def _migrate_observations(self, connection: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(observations)").fetchall()
        }
        if "world_state_id" not in columns:
            connection.execute("ALTER TABLE observations ADD COLUMN world_state_id TEXT")

    def save_settings(self, settings: RuntimeSettings) -> RuntimeSettings:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO settings (id, value_json)
                VALUES ('runtime', ?)
                ON CONFLICT(id) DO UPDATE SET value_json = excluded.value_json
                """,
                (settings.model_dump_json(),),
            )
            connection.commit()
        return settings

    def load_settings(self) -> Optional[RuntimeSettings]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value_json FROM settings WHERE id = 'runtime'"
            ).fetchone()
        if row is None:
            return None
        stored = RuntimeSettings.model_validate_json(row["value_json"])
        defaults = default_settings()
        merged = defaults.model_dump()
        stored_data = stored.model_dump()
        for key, value in stored_data.items():
            if key == "providers":
                continue
            merged[key] = value
        merged_providers: dict[str, dict] = {}
        for name in set(defaults.providers) | set(stored.providers):
            default_provider = defaults.providers.get(name, ProviderConfig(name=name))
            stored_provider = stored.providers.get(name)
            merged_provider = default_provider.model_dump()
            if stored_provider is not None:
                for key, value in stored_provider.model_dump().items():
                    if key == "metadata":
                        merged_provider["metadata"] = {
                            **merged_provider.get("metadata", {}),
                            **(value or {}),
                        }
                    elif isinstance(value, str):
                        merged_provider[key] = value or merged_provider.get(key)
                    elif value is not None:
                        merged_provider[key] = value
            merged_providers[name] = merged_provider
        merged["providers"] = merged_providers
        return RuntimeSettings.model_validate(merged)

    def prune(self, retention_days: int) -> None:
        cutoff = (_utc_now() - timedelta(days=retention_days)).isoformat()
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT image_path, thumbnail_path FROM observations WHERE created_at < ?",
                (cutoff,),
            ).fetchall()
            connection.execute(
                "DELETE FROM observations WHERE created_at < ?",
                (cutoff,),
            )
            connection.execute(
                "DELETE FROM scene_states WHERE created_at < ?",
                (cutoff,),
            )
            connection.execute(
                "DELETE FROM entity_tracks WHERE updated_at < ?",
                (cutoff,),
            )
            connection.execute(
                "DELETE FROM challenge_runs WHERE created_at < ?",
                (cutoff,),
            )
            connection.commit()
        for row in rows:
            for key in ("image_path", "thumbnail_path"):
                path = Path(row[key])
                if path.exists():
                    path.unlink()

    def create_observation(
        self,
        *,
        image: Image.Image,
        raw_bytes: bytes,
        embedding: list[float],
        session_id: str,
        confidence: float,
        novelty: float,
        source_query: Optional[str],
        tags: list[str],
        providers: list[str],
        metadata: dict,
        world_state_id: Optional[str] = None,
    ) -> Observation:
        digest = sha256(raw_bytes).hexdigest()[:16]
        observation_id = f"obs_{digest}_{uuid4().hex[:8]}"
        created_at = _utc_now()

        frame_path = self.frames_dir / f"{observation_id}.png"
        thumb_path = self.thumbs_dir / f"{observation_id}.png"

        image.save(frame_path, format="PNG")
        thumbnail = image.copy()
        thumbnail.thumbnail((320, 320))
        thumbnail.save(thumb_path, format="PNG")

        observation = Observation(
            id=observation_id,
            session_id=session_id,
            created_at=created_at,
            world_state_id=world_state_id,
            image_path=str(frame_path),
            thumbnail_path=str(thumb_path),
            width=image.width,
            height=image.height,
            embedding=embedding,
            summary=None,
            source_query=source_query,
            tags=tags,
            confidence=confidence,
            novelty=novelty,
            providers=providers,
            metadata=metadata,
        )

        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO observations (
                    id, session_id, created_at, world_state_id, image_path, thumbnail_path, width, height,
                    embedding_json, summary, source_query, tags_json, confidence, novelty,
                    providers_json, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.id,
                    observation.session_id,
                    observation.created_at.isoformat(),
                    observation.world_state_id,
                    observation.image_path,
                    observation.thumbnail_path,
                    observation.width,
                    observation.height,
                    json.dumps(observation.embedding),
                    observation.summary,
                    observation.source_query,
                    json.dumps(observation.tags),
                    observation.confidence,
                    observation.novelty,
                    json.dumps(observation.providers),
                    json.dumps(observation.metadata),
                ),
            )
            connection.commit()
        return observation

    def update_observation(
        self,
        observation_id: str,
        *,
        summary: Optional[str] = None,
        providers: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        world_state_id: Optional[str] = None,
    ) -> Observation:
        current = self.get_observation(observation_id)
        if current is None:
            raise KeyError(observation_id)
        updated = current.model_copy(
            update={
                "summary": summary if summary is not None else current.summary,
                "providers": providers if providers is not None else current.providers,
                "metadata": metadata if metadata is not None else current.metadata,
                "world_state_id": world_state_id if world_state_id is not None else current.world_state_id,
            }
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE observations
                SET summary = ?, providers_json = ?, metadata_json = ?, world_state_id = ?
                WHERE id = ?
                """,
                (
                    updated.summary,
                    json.dumps(updated.providers),
                    json.dumps(updated.metadata),
                    updated.world_state_id,
                    observation_id,
                ),
            )
            connection.commit()
        return updated

    def get_observation(self, observation_id: str) -> Optional[Observation]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM observations WHERE id = ?",
                (observation_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_observation(row)

    def get_observations_by_ids(self, observation_ids: list[str]) -> list[Observation]:
        if not observation_ids:
            return []
        placeholders = ", ".join("?" for _ in observation_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM observations WHERE id IN ({placeholders}) ORDER BY created_at ASC",
                tuple(observation_ids),
            ).fetchall()
        return [self._row_to_observation(row) for row in rows]

    def recent_observations(self, *, session_id: Optional[str] = None, limit: int = 10) -> list[Observation]:
        query = "SELECT * FROM observations"
        params: list[object] = []
        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [self._row_to_observation(row) for row in rows]

    def list_observations(self, *, session_id: Optional[str] = None, limit: int = 50) -> list[Observation]:
        return self.recent_observations(session_id=session_id, limit=limit)

    def search_by_vector(
        self,
        vector: list[float],
        *,
        top_k: int,
        session_id: Optional[str] = None,
        exclude_id: Optional[str] = None,
        time_window_s: Optional[int] = None,
    ) -> list[SearchHit]:
        observations = self._candidate_observations(
            session_id=session_id,
            exclude_id=exclude_id,
            time_window_s=time_window_s,
        )
        if not observations:
            return []
        target = np.array(vector, dtype=np.float32)
        target_norm = np.linalg.norm(target) or 1.0
        hits: list[tuple[float, Observation]] = []
        for observation in observations:
            current = np.array(observation.embedding, dtype=np.float32)
            denom = (np.linalg.norm(current) or 1.0) * target_norm
            similarity = float(np.dot(target, current) / denom)
            hits.append((similarity, observation))
        hits.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchHit(
                observation_id=observation.id,
                score=round(score, 6),
                summary=observation.summary,
                thumbnail_path=observation.thumbnail_path,
                session_id=observation.session_id,
                created_at=observation.created_at,
                tags=observation.tags,
            )
            for score, observation in hits[:top_k]
        ]

    def search_by_text(
        self,
        query: str,
        *,
        top_k: int,
        session_id: Optional[str] = None,
        time_window_s: Optional[int] = None,
    ) -> list[SearchHit]:
        normalized = query.lower().strip()
        terms = [term for term in normalized.split() if term]
        observations = self._candidate_observations(
            session_id=session_id,
            exclude_id=None,
            time_window_s=time_window_s,
        )
        scored: list[tuple[float, Observation]] = []
        for observation in observations:
            corpus = " ".join(
                filter(
                    None,
                    [
                        observation.summary or "",
                        observation.source_query or "",
                        " ".join(observation.tags),
                    ],
                )
            ).lower()
            if not corpus:
                continue
            coverage = sum(corpus.count(term) for term in terms)
            if coverage <= 0:
                continue
            age_hours = max((_utc_now() - observation.created_at).total_seconds() / 3600, 0.0)
            score = coverage + max(0.0, 1.0 - (age_hours / 48.0))
            scored.append((score, observation))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchHit(
                observation_id=observation.id,
                score=round(float(score), 6),
                summary=observation.summary,
                thumbnail_path=observation.thumbnail_path,
                session_id=observation.session_id,
                created_at=observation.created_at,
                tags=observation.tags,
            )
            for score, observation in scored[:top_k]
        ]

    def save_scene_state(self, scene_state: SceneState) -> SceneState:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO scene_states (id, session_id, created_at, observation_id, data_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    session_id = excluded.session_id,
                    created_at = excluded.created_at,
                    observation_id = excluded.observation_id,
                    data_json = excluded.data_json
                """,
                (
                    scene_state.id,
                    scene_state.session_id,
                    scene_state.created_at.isoformat(),
                    scene_state.observation_id,
                    scene_state.model_dump_json(),
                ),
            )
            connection.commit()
        return scene_state

    def get_scene_state(self, scene_state_id: str) -> Optional[SceneState]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT data_json FROM scene_states WHERE id = ?",
                (scene_state_id,),
            ).fetchone()
        if row is None:
            return None
        return SceneState.model_validate_json(row["data_json"])

    def latest_scene_state(self, session_id: str) -> Optional[SceneState]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT data_json
                FROM scene_states
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return SceneState.model_validate_json(row["data_json"])

    def recent_scene_states(self, *, session_id: str, limit: int = 12) -> list[SceneState]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT data_json
                FROM scene_states
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [SceneState.model_validate_json(row["data_json"]) for row in rows]

    def save_entity_tracks(self, tracks: list[EntityTrack]) -> list[EntityTrack]:
        if not tracks:
            return []
        with self._lock, self._connect() as connection:
            for track in tracks:
                connection.execute(
                    """
                    INSERT INTO entity_tracks (id, session_id, updated_at, data_json)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        session_id = excluded.session_id,
                        updated_at = excluded.updated_at,
                        data_json = excluded.data_json
                    """,
                    (
                        track.id,
                        track.session_id,
                        track.last_seen_at.isoformat(),
                        track.model_dump_json(),
                    ),
                )
            connection.commit()
        return tracks

    def list_entity_tracks(self, *, session_id: str, limit: int = 64) -> list[EntityTrack]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT data_json
                FROM entity_tracks
                WHERE session_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [EntityTrack.model_validate_json(row["data_json"]) for row in rows]

    def save_challenge_run(self, challenge: ChallengeRun) -> ChallengeRun:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO challenge_runs (id, session_id, created_at, data_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    session_id = excluded.session_id,
                    created_at = excluded.created_at,
                    data_json = excluded.data_json
                """,
                (
                    challenge.id,
                    challenge.session_id,
                    challenge.created_at.isoformat(),
                    challenge.model_dump_json(),
                ),
            )
            connection.commit()
        return challenge

    def recent_challenge_runs(self, *, session_id: str, limit: int = 12) -> list[ChallengeRun]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT data_json
                FROM challenge_runs
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [ChallengeRun.model_validate_json(row["data_json"]) for row in rows]

    def _candidate_observations(
        self,
        *,
        session_id: Optional[str],
        exclude_id: Optional[str],
        time_window_s: Optional[int],
    ) -> list[Observation]:
        query = "SELECT * FROM observations WHERE 1=1"
        params: list[object] = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if exclude_id:
            query += " AND id != ?"
            params.append(exclude_id)
        if time_window_s:
            cutoff = (_utc_now() - timedelta(seconds=time_window_s)).isoformat()
            query += " AND created_at >= ?"
            params.append(cutoff)
        query += " ORDER BY created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [self._row_to_observation(row) for row in rows]

    def _row_to_observation(self, row: sqlite3.Row) -> Observation:
        return Observation(
            id=row["id"],
            session_id=row["session_id"],
            created_at=_parse_dt(row["created_at"]),
            world_state_id=row["world_state_id"] if "world_state_id" in row.keys() else None,
            image_path=row["image_path"],
            thumbnail_path=row["thumbnail_path"],
            width=row["width"],
            height=row["height"],
            embedding=json.loads(row["embedding_json"]),
            summary=row["summary"],
            source_query=row["source_query"],
            tags=json.loads(row["tags_json"]),
            confidence=row["confidence"],
            novelty=row["novelty"],
            providers=json.loads(row["providers_json"]),
            metadata=json.loads(row["metadata_json"]),
        )
