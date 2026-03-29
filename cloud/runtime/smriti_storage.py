from __future__ import annotations

import contextlib
import dataclasses
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator, Optional
from uuid import uuid4

import numpy as np

from .error_types import SmritiSchemaError
from .observability import SchemaVersionManager
from .storage import ObservationStore, _parse_dt, _utc_now


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _dump_json(value: Any) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True)


def _load_json(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _parse_optional_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return _parse_dt(value)


def _normalize_vector(vector: np.ndarray | list[float] | tuple[float, ...], *, target_dim: int | None = None) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return np.zeros(target_dim or 1, dtype=np.float32)
    if target_dim is not None and array.size != target_dim:
        if array.size < target_dim:
            array = np.pad(array, (0, target_dim - array.size))
        else:
            bins = np.array_split(array, target_dim)
            array = np.array([float(chunk.mean()) for chunk in bins], dtype=np.float32)
    norm = float(np.linalg.norm(array)) or 1.0
    return (array / norm).astype(np.float32)


def _mean_embedding(vectors: list[np.ndarray], *, target_dim: int | None = None) -> np.ndarray:
    if not vectors:
        return np.zeros(target_dim or 1, dtype=np.float32)
    matrix = np.stack([_normalize_vector(vector, target_dim=target_dim) for vector in vectors], axis=0)
    return _normalize_vector(np.mean(matrix, axis=0), target_dim=target_dim)


DepthStrataMap = dict[str, float]


@dataclass(slots=True)
class AnchorMatch:
    name: str
    confidence: float = 0.0
    patch_indices: list[int] = field(default_factory=list)
    energy: float = 0.0


@dataclass(slots=True)
class SetuDescription:
    text: str
    confidence: float = 0.0
    anchor_basis: str = ""
    region_id: str | None = None


@dataclass(slots=True)
class SmritiMedia:
    id: str
    observation_id: str | None
    file_path: str
    file_hash: str
    media_type: str
    depth_strata: DepthStrataMap | None
    anchor_matches: list[AnchorMatch]
    setu_descriptions: list[SetuDescription]
    hallucination_risk: float
    ingestion_status: str
    visual_cluster_id: int | None
    location_id: str | None = None
    temporal_scene_id: int | None = None
    original_created_at: datetime | None = None
    ingested_at: datetime | None = None
    pipeline_trace: dict[str, Any] | None = None
    alignment_loss: float | None = None
    error_message: str | None = None
    faiss_index_id: int | None = None
    embedding: list[float] | None = None


@dataclass(slots=True)
class SmritiPerson:
    id: str
    name: str
    embedding_centroid: list[float]
    first_seen_media_id: str | None
    appearance_count: int
    confidence_threshold: float
    is_private: bool
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True)
class SmritiLocation:
    id: str
    name: str | None
    scene_context_embedding: list[float]
    gps_lat: float | None
    gps_lon: float | None
    media_count: int
    is_private: bool
    created_at: datetime


@dataclass(slots=True)
class SmritiCluster:
    id: int
    centroid: list[float]
    media_count: int
    label: str | None
    dominant_depth_stratum: str | None
    temporal_span_days: float | None
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True)
class RecallResult:
    media_id: str
    file_path: str
    thumbnail_path: str
    setu_score: float
    vector_score: float
    fts_score: float
    hybrid_score: float
    primary_description: str
    anchor_basis: str
    depth_stratum: str
    created_at: datetime
    person_names: list[str]
    location_name: str | None


class SmetiDB(ObservationStore):
    SCHEMA_VERSION = 1

    MIGRATIONS: list[tuple[int, str]] = [
        (
            1,
            """
            CREATE TABLE IF NOT EXISTS smriti_media (
                id TEXT PRIMARY KEY,
                observation_id TEXT REFERENCES observations(id),
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                media_type TEXT NOT NULL CHECK(media_type IN
                    ('image','video','burst','screenshot','live_photo')),
                original_created_at TEXT,
                ingested_at TEXT NOT NULL,
                depth_strata_json TEXT,
                anchor_matches_json TEXT,
                setu_descriptions_json TEXT,
                pipeline_trace_json TEXT,
                alignment_loss REAL,
                hallucination_risk REAL,
                visual_cluster_id INTEGER,
                temporal_scene_id INTEGER,
                ingestion_status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(ingestion_status IN
                    ('pending','processing','complete','failed')),
                error_message TEXT,
                faiss_index_id INTEGER,
                embedding_json TEXT,
                location_id TEXT,
                UNIQUE(file_hash)
            );
            CREATE INDEX IF NOT EXISTS idx_smriti_media_status
                ON smriti_media(ingestion_status);
            CREATE INDEX IF NOT EXISTS idx_smriti_media_cluster
                ON smriti_media(visual_cluster_id);
            CREATE INDEX IF NOT EXISTS idx_smriti_media_hash
                ON smriti_media(file_hash);
            CREATE INDEX IF NOT EXISTS idx_smriti_media_location
                ON smriti_media(location_id);
            CREATE INDEX IF NOT EXISTS idx_smriti_media_observation
                ON smriti_media(observation_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS smriti_descriptions_fts
                USING fts5(
                    media_id UNINDEXED,
                    description_text,
                    anchor_names,
                    tokenize='porter unicode61'
                );

            CREATE TABLE IF NOT EXISTS smriti_persons (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                embedding_centroid_json TEXT NOT NULL,
                first_seen_media_id TEXT,
                appearance_count INTEGER DEFAULT 0,
                confidence_threshold REAL DEFAULT 0.82,
                is_private INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_smriti_persons_name
                ON smriti_persons(name);

            CREATE TABLE IF NOT EXISTS smriti_person_media (
                person_id TEXT REFERENCES smriti_persons(id),
                media_id TEXT REFERENCES smriti_media(id),
                confidence REAL NOT NULL,
                confirmed_by_user INTEGER DEFAULT 0,
                PRIMARY KEY (person_id, media_id)
            );
            CREATE INDEX IF NOT EXISTS idx_smriti_person_media_media
                ON smriti_person_media(media_id);

            CREATE TABLE IF NOT EXISTS smriti_locations (
                id TEXT PRIMARY KEY,
                name TEXT,
                scene_context_embedding_json TEXT NOT NULL,
                gps_lat REAL,
                gps_lon REAL,
                media_count INTEGER DEFAULT 0,
                is_private INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_smriti_locations_name
                ON smriti_locations(name);

            CREATE TABLE IF NOT EXISTS smriti_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                centroid_json TEXT NOT NULL,
                media_count INTEGER DEFAULT 0,
                label TEXT,
                dominant_depth_stratum TEXT,
                temporal_span_days REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS smriti_schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            """,
        ),
    ]

    def __init__(self, data_dir: Path, **kwargs: Any) -> None:
        super().__init__(data_dir, **kwargs)
        self._faiss_lock = threading.RLock()
        self._schema_manager = SchemaVersionManager(str(self.db_path))
        self._faiss_index_path = self.data_dir / "smriti_faiss.index"
        self._faiss_media_ids: list[str] = []
        self._faiss_vectors: list[np.ndarray] = []
        self._faiss_index_ids: list[int] = []
        self._faiss_index_by_media_id: dict[str, int] = {}
        self._faiss_vector_dim: int | None = None
        self._faiss_next_index_id = 0
        self._mandala_cache: dict[str, Any] | None = None
        self._mandala_cache_at = 0.0
        self._apply_migrations()
        self._load_or_rebuild_faiss_index()

    @contextlib.contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        try:
            try:
                connection.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
            try:
                connection.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.DatabaseError:
                pass
            connection.execute("PRAGMA foreign_keys=ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def _apply_migrations(self) -> None:
        try:
            self._schema_manager.apply_pending(self.MIGRATIONS)
        except SmritiSchemaError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            raise SmritiSchemaError(str(exc)) from exc

    def _load_or_rebuild_faiss_index(self) -> None:
        with self._faiss_lock:
            if self._load_faiss_index_from_disk():
                return
            self.faiss_rebuild()

    def _load_faiss_index_from_disk(self) -> bool:
        if not self._faiss_index_path.exists():
            return False
        try:
            with self._faiss_index_path.open("rb") as handle:
                payload = np.load(handle, allow_pickle=True)
                vectors = np.asarray(payload["vectors"], dtype=np.float32)
                media_ids = [str(item) for item in payload["media_ids"].tolist()]
                index_ids = [int(item) for item in payload["index_ids"].tolist()]
                vector_dim = int(np.asarray(payload["vector_dim"]).reshape(-1)[0]) if "vector_dim" in payload.files else None
        except Exception:
            return False
        self._faiss_vectors = [_normalize_vector(vector, target_dim=vectors.shape[1] if vectors.ndim == 2 else None) for vector in vectors]
        self._faiss_media_ids = media_ids
        self._faiss_index_ids = index_ids
        self._faiss_index_by_media_id = {media_id: index_id for media_id, index_id in zip(media_ids, index_ids)}
        self._faiss_vector_dim = vector_dim or (vectors.shape[1] if vectors.ndim == 2 and vectors.size else None)
        self._faiss_next_index_id = (max(index_ids) + 1) if index_ids else 0
        return True

    def _persist_faiss_index(self) -> None:
        payload = {
            "vectors": np.asarray(self._faiss_vectors, dtype=np.float32),
            "media_ids": np.asarray(self._faiss_media_ids, dtype=object),
            "index_ids": np.asarray(self._faiss_index_ids, dtype=np.int64),
            "vector_dim": np.asarray([self._faiss_vector_dim or 0], dtype=np.int64),
        }
        with self._faiss_index_path.open("wb") as handle:
            np.savez(handle, **payload)

    def _serialize_depth_strata(self, depth_strata: DepthStrataMap | None) -> str | None:
        if depth_strata is None:
            return None
        return _dump_json(depth_strata)

    def _deserialize_depth_strata(self, raw: str | None) -> DepthStrataMap | None:
        data = _load_json(raw, None)
        if not isinstance(data, dict):
            return None
        return {str(key): float(value) for key, value in data.items()}

    def _serialize_anchor_matches(self, anchor_matches: list[AnchorMatch]) -> str:
        return _dump_json([dataclasses.asdict(item) for item in anchor_matches])

    def _serialize_setu_descriptions(self, descriptions: list[SetuDescription]) -> str:
        return _dump_json([dataclasses.asdict(item) for item in descriptions])

    def _deserialize_anchor_matches(self, raw: str | None) -> list[AnchorMatch]:
        data = _load_json(raw, [])
        if not isinstance(data, list):
            return []
        result: list[AnchorMatch] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            result.append(
                AnchorMatch(
                    name=str(item.get("name", "")),
                    confidence=float(item.get("confidence", 0.0)),
                    patch_indices=[int(index) for index in item.get("patch_indices", []) if isinstance(index, (int, float))],
                    energy=float(item.get("energy", 0.0)),
                )
            )
        return result

    def _deserialize_setu_descriptions(self, raw: str | None) -> list[SetuDescription]:
        data = _load_json(raw, [])
        if not isinstance(data, list):
            return []
        result: list[SetuDescription] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            result.append(
                SetuDescription(
                    text=str(item.get("text", "")),
                    confidence=float(item.get("confidence", 0.0)),
                    anchor_basis=str(item.get("anchor_basis", "")),
                    region_id=item.get("region_id"),
                )
            )
        return result

    @staticmethod
    def _safe_size(path: Path) -> int:
        try:
            return path.stat().st_size if path.exists() else 0
        except OSError:
            return 0

    def _owned_paths_for_media(self, media: SmritiMedia) -> list[Path]:
        owned_paths: list[Path] = []
        for candidate in (
            Path(media.file_path),
            self.frames_dir / f"{media.id}.png",
            self.frames_dir / f"{media.id}.jpg",
            self.thumbs_dir / f"{media.id}.png",
            self.thumbs_dir / f"{media.id}.jpg",
        ):
            try:
                resolved = candidate.expanduser().resolve()
            except OSError:
                continue
            for root in (self.frames_dir, self.thumbs_dir):
                try:
                    resolved.relative_to(root.resolve())
                    owned_paths.append(resolved)
                    break
                except ValueError:
                    continue
        deduped: list[Path] = []
        seen: set[Path] = set()
        for item in owned_paths:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def get_ingestion_stats(self) -> dict[str, int]:
        stats = {"total": 0, "pending": 0, "processing": 0, "complete": 0, "failed": 0}
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT ingestion_status, COUNT(*) AS count
                FROM smriti_media
                GROUP BY ingestion_status
                """
            ).fetchall()
        for row in rows:
            status = str(row["ingestion_status"])
            count = int(row["count"])
            stats["total"] += count
            stats[status] = count
        return stats

    def count_media_in_folder(self, folder_path: str) -> int:
        folder = Path(folder_path).expanduser().resolve()
        count = 0
        with self._connect() as connection:
            rows = connection.execute("SELECT file_path FROM smriti_media").fetchall()
        for row in rows:
            file_path = row["file_path"]
            if not file_path:
                continue
            try:
                Path(file_path).expanduser().resolve().relative_to(folder)
            except (OSError, ValueError):
                continue
            count += 1
        return count

    def _delete_media_records(self, media_ids: list[str]) -> tuple[int, int]:
        if not media_ids:
            return 0, 0
        placeholders = ", ".join("?" for _ in media_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM smriti_media WHERE id IN ({placeholders})",
                tuple(media_ids),
            ).fetchall()
        medias = [self._row_to_media(row) for row in rows]
        removed_bytes = 0
        for media in medias:
            for owned_path in self._owned_paths_for_media(media):
                removed_bytes += self._safe_size(owned_path)
                try:
                    owned_path.unlink(missing_ok=True)
                except OSError:
                    continue

        with self._lock, self._connect() as connection:
            connection.execute(
                f"DELETE FROM smriti_descriptions_fts WHERE media_id IN ({placeholders})",
                tuple(media_ids),
            )
            connection.execute(
                f"DELETE FROM smriti_person_media WHERE media_id IN ({placeholders})",
                tuple(media_ids),
            )
            connection.execute(
                f"DELETE FROM smriti_media WHERE id IN ({placeholders})",
                tuple(media_ids),
            )
            connection.commit()

        self.faiss_rebuild()
        self._mandala_cache = None
        return len(medias), removed_bytes

    def prune_older_than(self, older_than_days: int) -> tuple[int, int]:
        cutoff = (_utc_now() - timedelta(days=int(older_than_days))).isoformat()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id
                FROM smriti_media
                WHERE COALESCE(original_created_at, ingested_at) < ?
                """,
                (cutoff,),
            ).fetchall()
        media_ids = [str(row["id"]) for row in rows]
        return self._delete_media_records(media_ids)

    def prune_missing_files(self) -> tuple[int, int]:
        missing_ids: list[str] = []
        with self._connect() as connection:
            rows = connection.execute("SELECT id, file_path FROM smriti_media").fetchall()
        for row in rows:
            file_path = row["file_path"]
            if not file_path:
                missing_ids.append(str(row["id"]))
                continue
            try:
                exists = Path(file_path).expanduser().resolve().exists()
            except OSError:
                exists = False
            if not exists:
                missing_ids.append(str(row["id"]))
        return self._delete_media_records(missing_ids)

    def prune_failed(self) -> tuple[int, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM smriti_media WHERE ingestion_status = 'failed'"
            ).fetchall()
        media_ids = [str(row["id"]) for row in rows]
        return self._delete_media_records(media_ids)

    def clear_all_smriti_data(self) -> tuple[int, int]:
        with self._connect() as connection:
            total_records = int(
                connection.execute("SELECT COUNT(*) AS count FROM smriti_media").fetchone()["count"]
            )

        removed_bytes = 0
        for root in (self.frames_dir, self.thumbs_dir):
            if not root.exists():
                continue
            for item in root.rglob("*"):
                if not item.is_file():
                    continue
                removed_bytes += self._safe_size(item)
                try:
                    item.unlink()
                except OSError:
                    continue
        removed_bytes += self._safe_size(self._faiss_index_path)
        try:
            self._faiss_index_path.unlink(missing_ok=True)
        except OSError:
            pass

        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM smriti_person_media")
            connection.execute("DELETE FROM smriti_persons")
            connection.execute("DELETE FROM smriti_locations")
            connection.execute("DELETE FROM smriti_clusters")
            connection.execute("DELETE FROM smriti_descriptions_fts")
            connection.execute("DELETE FROM smriti_media")
            connection.commit()

        self._faiss_media_ids = []
        self._faiss_vectors = []
        self._faiss_index_ids = []
        self._faiss_index_by_media_id = {}
        self._faiss_next_index_id = 0
        self._mandala_cache = None
        return total_records, removed_bytes

    def _media_embedding_vector(self, media: SmritiMedia) -> np.ndarray | None:
        if media.embedding is not None:
            return _normalize_vector(media.embedding, target_dim=self._faiss_vector_dim)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT embedding_json
                FROM smriti_media
                WHERE id = ?
                """,
                (media.id,),
            ).fetchone()
        if row and row["embedding_json"]:
            return _normalize_vector(_load_json(row["embedding_json"], []), target_dim=self._faiss_vector_dim)
        if media.observation_id:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT embedding_json FROM observations WHERE id = ?",
                    (media.observation_id,),
                ).fetchone()
            if row and row["embedding_json"]:
                return _normalize_vector(_load_json(row["embedding_json"], []), target_dim=self._faiss_vector_dim)
        return None

    def _row_to_media(self, row: sqlite3.Row) -> SmritiMedia:
        return SmritiMedia(
            id=row["id"],
            observation_id=row["observation_id"],
            file_path=row["file_path"],
            file_hash=row["file_hash"],
            media_type=row["media_type"],
            depth_strata=self._deserialize_depth_strata(row["depth_strata_json"]),
            anchor_matches=self._deserialize_anchor_matches(row["anchor_matches_json"]),
            setu_descriptions=self._deserialize_setu_descriptions(row["setu_descriptions_json"]),
            hallucination_risk=float(row["hallucination_risk"] or 0.0),
            ingestion_status=row["ingestion_status"],
            visual_cluster_id=row["visual_cluster_id"],
            location_id=row["location_id"],
            temporal_scene_id=row["temporal_scene_id"],
            original_created_at=_parse_optional_dt(row["original_created_at"]),
            ingested_at=_parse_optional_dt(row["ingested_at"]),
            pipeline_trace=_load_json(row["pipeline_trace_json"], None),
            alignment_loss=row["alignment_loss"],
            error_message=row["error_message"],
            faiss_index_id=row["faiss_index_id"],
            embedding=_load_json(row["embedding_json"], None),
        )

    def _row_to_person(self, row: sqlite3.Row) -> SmritiPerson:
        return SmritiPerson(
            id=row["id"],
            name=row["name"],
            embedding_centroid=_load_json(row["embedding_centroid_json"], []),
            first_seen_media_id=row["first_seen_media_id"],
            appearance_count=int(row["appearance_count"] or 0),
            confidence_threshold=float(row["confidence_threshold"] or 0.82),
            is_private=bool(row["is_private"]),
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )

    def _row_to_location(self, row: sqlite3.Row) -> SmritiLocation:
        return SmritiLocation(
            id=row["id"],
            name=row["name"],
            scene_context_embedding=_load_json(row["scene_context_embedding_json"], []),
            gps_lat=row["gps_lat"],
            gps_lon=row["gps_lon"],
            media_count=int(row["media_count"] or 0),
            is_private=bool(row["is_private"]),
            created_at=_parse_dt(row["created_at"]),
        )

    def _row_to_cluster(self, row: sqlite3.Row) -> SmritiCluster:
        return SmritiCluster(
            id=int(row["id"]),
            centroid=_load_json(row["centroid_json"], []),
            media_count=int(row["media_count"] or 0),
            label=row["label"],
            dominant_depth_stratum=row["dominant_depth_stratum"],
            temporal_span_days=row["temporal_span_days"],
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )

    def _media_result_fields(self, media_id: str) -> tuple[str, str, str | None]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT file_path, original_created_at, ingested_at FROM smriti_media WHERE id = ?",
                (media_id,),
            ).fetchone()
        if row is None:
            return "", "", None
        created_at = row["original_created_at"] or row["ingested_at"]
        return row["file_path"], created_at, row["ingested_at"]

    def _media_thumbnail_path(self, media_id: str, fallback: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT o.thumbnail_path
                FROM smriti_media m
                LEFT JOIN observations o ON o.id = m.observation_id
                WHERE m.id = ?
                """,
                (media_id,),
            ).fetchone()
        if row and row["thumbnail_path"]:
            return str(row["thumbnail_path"])
        return fallback

    def _media_person_names(self, media_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.name
                FROM smriti_person_media pm
                JOIN smriti_persons p ON p.id = pm.person_id
                WHERE pm.media_id = ?
                ORDER BY p.name
                """,
                (media_id,),
            ).fetchall()
        return [str(row["name"]) for row in rows]

    def _media_location_name(self, media_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT l.name
                FROM smriti_media m
                LEFT JOIN smriti_locations l ON l.id = m.location_id
                WHERE m.id = ?
                """,
                (media_id,),
            ).fetchone()
        if row is None:
            return None
        return row["name"]

    def _depth_stratum_label(self, media: SmritiMedia) -> str:
        if not media.depth_strata:
            return "unknown"
        best = max(media.depth_strata.items(), key=lambda item: item[1], default=None)
        if best is None:
            return "unknown"
        return str(best[0])

    def _setu_description_summary(self, media: SmritiMedia) -> tuple[str, str]:
        if media.setu_descriptions:
            description = media.setu_descriptions[0]
            anchor_basis = description.anchor_basis or ", ".join(match.name for match in media.anchor_matches[:3] if match.name)
            return description.text or "media observation", anchor_basis
        if media.anchor_matches:
            names = [match.name for match in media.anchor_matches if match.name]
            return (", ".join(names) or "media observation", ", ".join(names))
        return "media observation", ""

    def _sync_fts_index_locked(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM smriti_descriptions_fts")
            rows = connection.execute(
                """
                SELECT id, anchor_matches_json, setu_descriptions_json
                FROM smriti_media
                WHERE ingestion_status = 'complete'
                ORDER BY ingested_at ASC
                """
            ).fetchall()
            for row in rows:
                anchor_matches = self._deserialize_anchor_matches(row["anchor_matches_json"])
                descriptions = self._deserialize_setu_descriptions(row["setu_descriptions_json"])
                description_text = " ".join(description.text for description in descriptions if description.text).strip()
                anchor_names = " ".join(match.name for match in anchor_matches if match.name).strip()
                if not description_text and not anchor_names:
                    continue
                connection.execute(
                    """
                    INSERT INTO smriti_descriptions_fts (media_id, description_text, anchor_names)
                    VALUES (?, ?, ?)
                    """,
                    (row["id"], description_text, anchor_names),
                )
            connection.commit()

    def _write_media_record(self, media: SmritiMedia) -> SmritiMedia:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO smriti_media (
                    id, observation_id, file_path, file_hash, media_type, original_created_at,
                    ingested_at, depth_strata_json, anchor_matches_json, setu_descriptions_json,
                    pipeline_trace_json, alignment_loss, hallucination_risk, visual_cluster_id,
                    temporal_scene_id, ingestion_status, error_message, faiss_index_id,
                    embedding_json, location_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    observation_id = excluded.observation_id,
                    file_path = excluded.file_path,
                    file_hash = excluded.file_hash,
                    media_type = excluded.media_type,
                    original_created_at = excluded.original_created_at,
                    ingested_at = excluded.ingested_at,
                    depth_strata_json = excluded.depth_strata_json,
                    anchor_matches_json = excluded.anchor_matches_json,
                    setu_descriptions_json = excluded.setu_descriptions_json,
                    pipeline_trace_json = excluded.pipeline_trace_json,
                    alignment_loss = excluded.alignment_loss,
                    hallucination_risk = excluded.hallucination_risk,
                    visual_cluster_id = excluded.visual_cluster_id,
                    temporal_scene_id = excluded.temporal_scene_id,
                    ingestion_status = excluded.ingestion_status,
                    error_message = excluded.error_message,
                    faiss_index_id = excluded.faiss_index_id,
                    embedding_json = excluded.embedding_json,
                    location_id = excluded.location_id
                """,
                (
                    media.id,
                    media.observation_id,
                    media.file_path,
                    media.file_hash,
                    media.media_type,
                    media.original_created_at.isoformat() if media.original_created_at else None,
                    media.ingested_at.isoformat() if media.ingested_at else _utc_now().isoformat(),
                    self._serialize_depth_strata(media.depth_strata),
                    self._serialize_anchor_matches(media.anchor_matches),
                    self._serialize_setu_descriptions(media.setu_descriptions),
                    _dump_json(media.pipeline_trace) if media.pipeline_trace is not None else None,
                    media.alignment_loss,
                    media.hallucination_risk,
                    media.visual_cluster_id,
                    media.temporal_scene_id,
                    media.ingestion_status,
                    media.error_message,
                    media.faiss_index_id,
                    _dump_json(media.embedding) if media.embedding is not None else None,
                    media.location_id,
                ),
            )
            connection.commit()
        if media.ingestion_status == "complete":
            with self._faiss_lock:
                embedding = self._media_embedding_vector(media)
                if embedding is not None:
                    self.faiss_add(media.id, embedding)
        with self._lock:
            self._sync_fts_index_locked()
            self._update_media_location_counts()
        return self.get_smriti_media(media.id) or media

    def _media_from_kwargs(self, **kwargs: Any) -> SmritiMedia:
        media_id = str(kwargs.get("id") or f"smriti_{uuid4().hex[:12]}")
        original_created_at = kwargs.get("original_created_at")
        ingested_at = kwargs.get("ingested_at") or _utc_now()
        depth_strata = kwargs.get("depth_strata")
        if depth_strata is not None and not isinstance(depth_strata, dict):
            raise TypeError("depth_strata must be a mapping or None")
        anchor_matches = kwargs.get("anchor_matches") or []
        setu_descriptions = kwargs.get("setu_descriptions") or []
        embedding = kwargs.get("embedding")
        if embedding is None and kwargs.get("embedding_json") is not None:
            embedding = _load_json(kwargs["embedding_json"], None)
        file_hash = str(kwargs.get("file_hash") or "")
        if not file_hash and kwargs.get("file_path"):
            try:
                file_hash = self._hash_path(Path(str(kwargs["file_path"])))
            except Exception:
                file_hash = f"missing-{media_id}"
        if not file_hash:
            raise ValueError("file_hash is required")
        if kwargs.get("ingestion_status") is None:
            ingestion_status = "pending"
        else:
            ingestion_status = str(kwargs.get("ingestion_status"))
        return SmritiMedia(
            id=media_id,
            observation_id=kwargs.get("observation_id"),
            file_path=str(kwargs.get("file_path") or ""),
            file_hash=file_hash,
            media_type=str(kwargs.get("media_type") or "image"),
            depth_strata=depth_strata,
            anchor_matches=[
                item if isinstance(item, AnchorMatch) else AnchorMatch(**item)
                for item in anchor_matches
                if isinstance(item, (AnchorMatch, dict))
            ],
            setu_descriptions=[
                item if isinstance(item, SetuDescription) else SetuDescription(**item)
                for item in setu_descriptions
                if isinstance(item, (SetuDescription, dict))
            ],
            hallucination_risk=float(kwargs.get("hallucination_risk") or 0.0),
            ingestion_status=ingestion_status,
            visual_cluster_id=kwargs.get("visual_cluster_id"),
            location_id=kwargs.get("location_id"),
            temporal_scene_id=kwargs.get("temporal_scene_id"),
            original_created_at=original_created_at if isinstance(original_created_at, datetime) else _parse_optional_dt(str(original_created_at)) if original_created_at else None,
            ingested_at=ingested_at if isinstance(ingested_at, datetime) else _parse_optional_dt(str(ingested_at)) or _utc_now(),
            pipeline_trace=kwargs.get("pipeline_trace"),
            alignment_loss=kwargs.get("alignment_loss"),
            error_message=kwargs.get("error_message"),
            faiss_index_id=kwargs.get("faiss_index_id"),
            embedding=_normalize_vector(embedding, target_dim=self._faiss_vector_dim).tolist() if embedding is not None else None,
        )

    def _hash_path(self, path: Path) -> str:
        from hashlib import sha256

        return sha256(path.read_bytes()).hexdigest()

    def create_smriti_media(self, **kwargs: Any) -> SmritiMedia:
        file_hash = str(kwargs.get("file_hash") or "")
        if file_hash:
            existing = self._get_smriti_media_by_hash(file_hash)
            if existing is not None:
                return existing
        media = self._media_from_kwargs(**kwargs)
        existing = self._get_smriti_media_by_hash(media.file_hash)
        if existing is not None:
            return existing
        return self._write_media_record(media)

    def _get_smriti_media_by_hash(self, file_hash: str) -> SmritiMedia | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM smriti_media WHERE file_hash = ? ORDER BY ingested_at DESC LIMIT 1",
                (file_hash,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_media(row)

    def get_smriti_media_by_hash(self, file_hash: str) -> SmritiMedia | None:
        return self._get_smriti_media_by_hash(file_hash)

    def get_smriti_media(self, media_id: str) -> SmritiMedia | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM smriti_media WHERE id = ?",
                (media_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_media(row)

    def update_smriti_media(self, media_id: str, **kwargs: Any) -> SmritiMedia:
        current = self.get_smriti_media(media_id)
        if current is None:
            raise KeyError(media_id)
        payload = dataclasses.asdict(current)
        payload.update(kwargs)
        payload["id"] = current.id
        payload["file_hash"] = kwargs.get("file_hash", current.file_hash)
        payload["embedding"] = kwargs.get("embedding", current.embedding)
        payload["ingested_at"] = kwargs.get("ingested_at", current.ingested_at or _utc_now())
        payload["original_created_at"] = kwargs.get("original_created_at", current.original_created_at)
        payload["depth_strata"] = kwargs.get("depth_strata", current.depth_strata)
        payload["anchor_matches"] = kwargs.get("anchor_matches", current.anchor_matches)
        payload["setu_descriptions"] = kwargs.get("setu_descriptions", current.setu_descriptions)
        updated = self._media_from_kwargs(**payload)
        return self._write_media_record(updated)

    def mark_smriti_media_failed(self, file_hash: str, error_message: str) -> SmritiMedia | None:
        media = self._get_smriti_media_by_hash(file_hash)
        if media is None:
            return None
        return self.update_smriti_media(
            media.id,
            ingestion_status="failed",
            error_message=error_message,
        )

    def get_person_by_name(self, name: str) -> SmritiPerson | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM smriti_persons WHERE name = ? ORDER BY updated_at DESC LIMIT 1",
                (name,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_person(row)

    def get_pending_media(self, limit: int = 10) -> list[SmritiMedia]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM smriti_media
                WHERE ingestion_status = 'pending'
                ORDER BY ingested_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_media(row) for row in rows]

    def create_person(self, name: str, embedding: np.ndarray) -> SmritiPerson:
        person_id = f"person_{uuid4().hex[:12]}"
        now = _utc_now()
        centroid = _normalize_vector(embedding, target_dim=self._faiss_vector_dim).tolist()
        person = SmritiPerson(
            id=person_id,
            name=name,
            embedding_centroid=centroid,
            first_seen_media_id=None,
            appearance_count=0,
            confidence_threshold=0.82,
            is_private=False,
            created_at=now,
            updated_at=now,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO smriti_persons (
                    id, name, embedding_centroid_json, first_seen_media_id, appearance_count,
                    confidence_threshold, is_private, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    person.id,
                    person.name,
                    _dump_json(person.embedding_centroid),
                    person.first_seen_media_id,
                    person.appearance_count,
                    person.confidence_threshold,
                    int(person.is_private),
                    person.created_at.isoformat(),
                    person.updated_at.isoformat(),
                ),
            )
            connection.commit()
        return person

    def get_person(self, person_id: str) -> SmritiPerson | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM smriti_persons WHERE id = ?",
                (person_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_person(row)

    def create_location(
        self,
        name: str | None,
        embedding: np.ndarray,
        *,
        gps_lat: float | None = None,
        gps_lon: float | None = None,
        is_private: bool = False,
    ) -> SmritiLocation:
        location_id = f"loc_{uuid4().hex[:12]}"
        now = _utc_now()
        location = SmritiLocation(
            id=location_id,
            name=name,
            scene_context_embedding=_normalize_vector(embedding, target_dim=self._faiss_vector_dim).tolist(),
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            media_count=0,
            is_private=is_private,
            created_at=now,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO smriti_locations (
                    id, name, scene_context_embedding_json, gps_lat, gps_lon,
                    media_count, is_private, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    location.id,
                    location.name,
                    _dump_json(location.scene_context_embedding),
                    location.gps_lat,
                    location.gps_lon,
                    location.media_count,
                    int(location.is_private),
                    location.created_at.isoformat(),
                ),
            )
            connection.commit()
        return location

    def get_location(self, location_id: str) -> SmritiLocation | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM smriti_locations WHERE id = ?",
                (location_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_location(row)

    def link_person_to_media(self, person_id: str, media_id: str, confidence: float) -> None:
        if self.get_person(person_id) is None:
            raise KeyError(person_id)
        if self.get_smriti_media(media_id) is None:
            raise KeyError(media_id)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO smriti_person_media (person_id, media_id, confidence, confirmed_by_user)
                VALUES (?, ?, ?, 0)
                ON CONFLICT(person_id, media_id) DO UPDATE SET
                    confidence = excluded.confidence,
                    confirmed_by_user = MAX(confirmed_by_user, excluded.confirmed_by_user)
                """,
                (person_id, media_id, float(confidence)),
            )
            appearance_count = connection.execute(
                "SELECT COUNT(*) AS count FROM smriti_person_media WHERE person_id = ?",
                (person_id,),
            ).fetchone()["count"]
            connection.execute(
                """
                UPDATE smriti_persons
                SET appearance_count = ?, updated_at = ?
                WHERE id = ?
                """,
                (int(appearance_count), _utc_now().isoformat(), person_id),
            )
            connection.commit()

    def propagate_person_tag(self, person_id: str, threshold: float = 0.82) -> int:
        person = self.get_person(person_id)
        if person is None:
            raise KeyError(person_id)
        centroid = _normalize_vector(person.embedding_centroid, target_dim=self._faiss_vector_dim)
        linked_count = 0
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT m.*
                FROM smriti_media m
                LEFT JOIN smriti_person_media pm
                    ON pm.media_id = m.id AND pm.person_id = ?
                WHERE pm.person_id IS NULL
                  AND m.ingestion_status = 'complete'
                """,
                (person_id,),
            ).fetchall()
        for row in rows:
            media = self._row_to_media(row)
            media_embedding = self._media_embedding_vector(media)
            if media_embedding is None:
                continue
            similarity = float(np.dot(centroid, media_embedding))
            if similarity < threshold:
                continue
            self.link_person_to_media(person_id, media.id, similarity)
            linked_count += 1
        with self._lock, self._connect() as connection:
            appearance_count = connection.execute(
                "SELECT COUNT(*) AS count FROM smriti_person_media WHERE person_id = ?",
                (person_id,),
            ).fetchone()["count"]
            connection.execute(
                """
                UPDATE smriti_persons
                SET appearance_count = ?, updated_at = ?
                WHERE id = ?
                """,
                (int(appearance_count), _utc_now().isoformat(), person_id),
            )
            connection.commit()
        return linked_count

    def _candidate_media_rows(
        self,
        *,
        person_filter: str | None = None,
        location_filter: str | None = None,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
    ) -> list[sqlite3.Row]:
        where: list[str] = ["1 = 1"]
        params: list[Any] = []
        if person_filter:
            where.append(
                """
                EXISTS (
                    SELECT 1
                    FROM smriti_person_media pm
                    JOIN smriti_persons p ON p.id = pm.person_id
                    WHERE pm.media_id = m.id
                      AND (p.id = ? OR lower(p.name) = lower(?))
                )
                """
            )
            params.extend([person_filter, person_filter])
        if location_filter:
            where.append(
                """
                EXISTS (
                    SELECT 1
                    FROM smriti_locations l
                    WHERE l.id = m.location_id
                      AND (l.id = ? OR lower(COALESCE(l.name, '')) = lower(?))
                )
                """
            )
            params.extend([location_filter, location_filter])
        if time_start is not None:
            where.append("COALESCE(m.original_created_at, m.ingested_at) >= ?")
            params.append(time_start.isoformat())
        if time_end is not None:
            where.append("COALESCE(m.original_created_at, m.ingested_at) <= ?")
            params.append(time_end.isoformat())
        query = f"""
            SELECT m.*
            FROM smriti_media m
            WHERE {' AND '.join(where)}
            ORDER BY COALESCE(m.original_created_at, m.ingested_at) DESC
        """
        with self._connect() as connection:
            return connection.execute(query, tuple(params)).fetchall()

    def _build_recall_result(self, media: SmritiMedia, *, vector_score: float, fts_score: float) -> RecallResult:
        primary_description, anchor_basis = self._setu_description_summary(media)
        depth_stratum = self._depth_stratum_label(media)
        hybrid_score = (vector_score * 0.7) + (fts_score * 0.3)
        created_at = media.original_created_at or media.ingested_at or _utc_now()
        return RecallResult(
            media_id=media.id,
            file_path=media.file_path,
            thumbnail_path=self._media_thumbnail_path(media.id, media.file_path),
            setu_score=round(max(0.0, 1.0 - hybrid_score), 6),
            vector_score=round(vector_score, 6),
            fts_score=round(fts_score, 6),
            hybrid_score=round(hybrid_score, 6),
            primary_description=primary_description,
            anchor_basis=anchor_basis,
            depth_stratum=depth_stratum,
            created_at=created_at,
            person_names=self._media_person_names(media.id),
            location_name=self._media_location_name(media.id),
        )

    def recall_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 50,
        person_filter: str | None = None,
        location_filter: str | None = None,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
    ) -> list[RecallResult]:
        candidates = self._candidate_media_rows(
            person_filter=person_filter,
            location_filter=location_filter,
            time_start=time_start,
            time_end=time_end,
        )
        if not candidates:
            return []
        query_vector = _normalize_vector(query_embedding, target_dim=self._faiss_vector_dim)
        scored: list[tuple[float, SmritiMedia]] = []
        for row in candidates:
            media = self._row_to_media(row)
            media_vector = self._media_embedding_vector(media)
            if media_vector is None:
                continue
            similarity = float(np.dot(query_vector, media_vector))
            scored.append((similarity, media))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            self._build_recall_result(media, vector_score=score, fts_score=0.0)
            for score, media in scored[:top_k]
        ]

    def _fts_score(self, query: str, media: SmritiMedia) -> float:
        query_terms = {token for token in query.lower().split() if token}
        if not query_terms:
            return 0.0
        descriptions = " ".join(description.text for description in media.setu_descriptions if description.text)
        anchors = " ".join(match.name for match in media.anchor_matches if match.name)
        corpus = f"{descriptions} {anchors}".lower()
        if not corpus.strip():
            return 0.0
        hits = sum(1 for term in query_terms if term in corpus)
        return round(min(hits / max(len(query_terms), 1), 1.0), 6)

    def fts_search(self, query: str, top_k: int = 20) -> list[RecallResult]:
        normalized = query.strip()
        if not normalized:
            return []
        with self._connect() as connection:
            try:
                rows = connection.execute(
                    """
                    SELECT media_id
                    FROM smriti_descriptions_fts
                    WHERE smriti_descriptions_fts MATCH ?
                    LIMIT ?
                    """,
                    (normalized, max(top_k * 4, top_k)),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        if rows:
            media_ids = [row["media_id"] for row in rows]
            placeholders = ", ".join("?" for _ in media_ids)
            with self._connect() as connection:
                media_rows = connection.execute(
                    f"SELECT * FROM smriti_media WHERE id IN ({placeholders})",
                    tuple(media_ids),
                ).fetchall()
        else:
            media_rows = self._candidate_media_rows()
        results: list[tuple[float, SmritiMedia]] = []
        for row in media_rows:
            media = self._row_to_media(row)
            fts_score = self._fts_score(normalized, media)
            if fts_score <= 0.0:
                continue
            results.append((fts_score, media))
        results.sort(key=lambda item: item[0], reverse=True)
        return [
            self._build_recall_result(media, vector_score=0.0, fts_score=score)
            for score, media in results[:top_k]
        ]

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        fts_query: str,
        top_k: int = 20,
        vector_weight: float = 0.7,
        fts_weight: float = 0.3,
    ) -> list[RecallResult]:
        candidates = self._candidate_media_rows()
        if not candidates:
            return []
        query_vector = _normalize_vector(query_embedding, target_dim=self._faiss_vector_dim)
        scored: list[tuple[float, SmritiMedia, float, float]] = []
        for row in candidates:
            media = self._row_to_media(row)
            media_vector = self._media_embedding_vector(media)
            if media_vector is None:
                continue
            vector_score = float(np.dot(query_vector, media_vector))
            fts_score = self._fts_score(fts_query, media) if fts_query.strip() else 0.0
            hybrid_score = (vector_score * vector_weight) + (fts_score * fts_weight)
            scored.append((hybrid_score, media, vector_score, fts_score))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            self._build_recall_result(media, vector_score=vector_score, fts_score=fts_score)
            for _, media, vector_score, fts_score in scored[:top_k]
        ]

    def faiss_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        filter_status: str = "complete",
    ) -> list[tuple[str, float]]:
        if not self._faiss_vectors:
            return []
        query_vector = _normalize_vector(query_embedding, target_dim=self._faiss_vector_dim)
        media_status: dict[str, str] = {}
        with self._connect() as connection:
            placeholders = ", ".join("?" for _ in self._faiss_media_ids)
            if placeholders:
                rows = connection.execute(
                    f"SELECT id, ingestion_status FROM smriti_media WHERE id IN ({placeholders})",
                    tuple(self._faiss_media_ids),
                ).fetchall()
                media_status = {str(row["id"]): str(row["ingestion_status"]) for row in rows}
        scored: list[tuple[str, float]] = []
        for media_id, vector in zip(self._faiss_media_ids, self._faiss_vectors):
            if filter_status and media_status.get(media_id, "") != filter_status:
                continue
            score = float(np.dot(query_vector, vector))
            scored.append((media_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(media_id, round(score, 6)) for media_id, score in scored[:top_k]]

    def faiss_add(self, media_id: str, embedding: np.ndarray) -> int:
        vector = _normalize_vector(embedding, target_dim=self._faiss_vector_dim)
        with self._faiss_lock:
            if self._faiss_vector_dim is None:
                self._faiss_vector_dim = int(vector.size)
            elif vector.size != self._faiss_vector_dim:
                vector = _normalize_vector(vector, target_dim=self._faiss_vector_dim)
            if media_id in self._faiss_index_by_media_id:
                index = self._faiss_index_by_media_id[media_id]
                position = self._faiss_index_ids.index(index)
                self._faiss_vectors[position] = vector
            else:
                index = self._faiss_next_index_id
                self._faiss_next_index_id += 1
                self._faiss_media_ids.append(media_id)
                self._faiss_vectors.append(vector)
                self._faiss_index_ids.append(index)
                self._faiss_index_by_media_id[media_id] = index
            self._persist_faiss_index()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE smriti_media
                SET faiss_index_id = ?, embedding_json = ?
                WHERE id = ?
                """,
                (index, _dump_json(vector.tolist()), media_id),
            )
            connection.commit()
        return index

    def faiss_rebuild(self) -> None:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT m.id, m.faiss_index_id, m.embedding_json, m.observation_id
                FROM smriti_media m
                WHERE m.ingestion_status = 'complete'
                ORDER BY COALESCE(m.faiss_index_id, m.ingested_at) ASC
                """
            ).fetchall()
        vectors: list[np.ndarray] = []
        media_ids: list[str] = []
        index_ids: list[int] = []
        next_index_id = 0
        for row in rows:
            raw_embedding = _load_json(row["embedding_json"], None)
            if raw_embedding is None and row["observation_id"]:
                with self._connect() as connection:
                    obs_row = connection.execute(
                        "SELECT embedding_json FROM observations WHERE id = ?",
                        (row["observation_id"],),
                    ).fetchone()
                raw_embedding = _load_json(obs_row["embedding_json"], None) if obs_row else None
            if raw_embedding is None:
                continue
            vector = _normalize_vector(raw_embedding, target_dim=self._faiss_vector_dim)
            vectors.append(vector)
            media_ids.append(str(row["id"]))
            index_id = int(row["faiss_index_id"]) if row["faiss_index_id"] is not None else next_index_id
            index_ids.append(index_id)
            next_index_id = max(next_index_id, index_id + 1)
        with self._faiss_lock:
            self._faiss_vectors = vectors
            self._faiss_media_ids = media_ids
            self._faiss_index_ids = index_ids
            self._faiss_index_by_media_id = {media_id: index_id for media_id, index_id in zip(media_ids, index_ids)}
            self._faiss_next_index_id = next_index_id
            if vectors:
                self._faiss_vector_dim = int(vectors[0].size)
            self._persist_faiss_index()
        with self._lock, self._connect() as connection:
            for media_id, index_id, vector in zip(media_ids, index_ids, vectors):
                connection.execute(
                    """
                    UPDATE smriti_media
                    SET faiss_index_id = ?, embedding_json = ?
                    WHERE id = ?
                    """,
                    (index_id, _dump_json(vector.tolist()), media_id),
                )
            connection.commit()

    def update_clusters(self, media_ids: list[str], embeddings: np.ndarray) -> None:
        media_ids = [str(media_id) for media_id in media_ids]
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(media_ids):
            raise ValueError("embeddings must match media_ids")
        normalized = np.stack([
            _normalize_vector(vector, target_dim=self._faiss_vector_dim)
            for vector in matrix
        ], axis=0)
        with self._lock, self._connect() as connection:
            cluster_rows = connection.execute("SELECT * FROM smriti_clusters").fetchall()
            clusters = [self._row_to_cluster(row) for row in cluster_rows]
            for media_id, vector in zip(media_ids, normalized):
                media_row = connection.execute("SELECT * FROM smriti_media WHERE id = ?", (media_id,)).fetchone()
                if media_row is None:
                    continue
                media = self._row_to_media(media_row)
                best_cluster: SmritiCluster | None = None
                best_similarity = -1.0
                for cluster in clusters:
                    centroid = _normalize_vector(cluster.centroid, target_dim=vector.size)
                    similarity = float(np.dot(centroid, vector))
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_cluster = cluster
                if best_cluster is None or best_similarity < 0.82:
                    now = _utc_now()
                    connection.execute(
                        """
                        INSERT INTO smriti_clusters (
                            centroid_json, media_count, label, dominant_depth_stratum,
                            temporal_span_days, created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            _dump_json(vector.tolist()),
                            1,
                            media.media_type,
                            self._depth_stratum_label(media),
                            0.0,
                            now.isoformat(),
                            now.isoformat(),
                        ),
                    )
                    cluster_id = int(connection.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])
                    connection.execute(
                        "UPDATE smriti_media SET visual_cluster_id = ? WHERE id = ?",
                        (cluster_id, media_id),
                    )
                    clusters.append(
                        SmritiCluster(
                            id=cluster_id,
                            centroid=vector.tolist(),
                            media_count=1,
                            label=media.media_type,
                            dominant_depth_stratum=self._depth_stratum_label(media),
                            temporal_span_days=0.0,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                    continue
                updated_media_count = best_cluster.media_count + 1
                updated_centroid = _mean_embedding(
                    [
                        np.asarray(best_cluster.centroid, dtype=np.float32),
                        vector,
                    ],
                    target_dim=vector.size,
                )
                now = _utc_now()
                connection.execute(
                    """
                    UPDATE smriti_clusters
                    SET centroid_json = ?, media_count = ?, label = ?, dominant_depth_stratum = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        _dump_json(updated_centroid.tolist()),
                        updated_media_count,
                        best_cluster.label or media.media_type,
                        self._depth_stratum_label(media) or best_cluster.dominant_depth_stratum,
                        now.isoformat(),
                        best_cluster.id,
                    ),
                )
                connection.execute(
                    "UPDATE smriti_media SET visual_cluster_id = ? WHERE id = ?",
                    (best_cluster.id, media_id),
                )
            connection.commit()
        with self._lock, self._connect() as connection:
            cluster_rows = connection.execute("SELECT * FROM smriti_clusters").fetchall()
            for row in cluster_rows:
                cluster_id = int(row["id"])
                member_rows = connection.execute(
                    """
                    SELECT COALESCE(m.original_created_at, m.ingested_at) AS created_at
                    FROM smriti_media m
                    WHERE m.visual_cluster_id = ?
                    """,
                    (cluster_id,),
                ).fetchall()
                created_times = [_parse_dt(str(member["created_at"])) for member in member_rows if member["created_at"]]
                span_days = 0.0
                if created_times:
                    span_days = max((max(created_times) - min(created_times)).total_seconds() / 86400.0, 0.0)
                dominant_depth = self._dominant_depth_label_for_cluster(connection, cluster_id)
                connection.execute(
                    """
                    UPDATE smriti_clusters
                    SET media_count = ?, temporal_span_days = ?, dominant_depth_stratum = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        len(member_rows),
                        span_days,
                        dominant_depth,
                        _utc_now().isoformat(),
                        cluster_id,
                    ),
                )
            connection.commit()
        self._invalidate_mandala_cache()

    def _dominant_depth_label_for_cluster(self, connection: sqlite3.Connection, cluster_id: int) -> str | None:
        rows = connection.execute(
            """
            SELECT depth_strata_json
            FROM smriti_media
            WHERE visual_cluster_id = ?
            """,
            (cluster_id,),
        ).fetchall()
        counts: dict[str, float] = {}
        for row in rows:
            depth = self._deserialize_depth_strata(row["depth_strata_json"])
            if not depth:
                continue
            label, score = max(depth.items(), key=lambda item: item[1])
            counts[label] = counts.get(label, 0.0) + float(score)
        if not counts:
            return None
        return max(counts, key=counts.get)

    def _invalidate_mandala_cache(self) -> None:
        self._mandala_cache = None
        self._mandala_cache_at = 0.0

    def _build_mandala_data(self) -> dict[str, Any]:
        with self._connect() as connection:
            cluster_rows = connection.execute(
                "SELECT * FROM smriti_clusters ORDER BY media_count DESC, id ASC"
            ).fetchall()
            clusters = [self._row_to_cluster(row) for row in cluster_rows]
        nodes = [
            {
                "id": cluster.id,
                "label": cluster.label or f"Cluster {cluster.id}",
                "media_count": cluster.media_count,
                "centroid": cluster.centroid,
                "dominant_depth_stratum": cluster.dominant_depth_stratum,
                "temporal_span_days": cluster.temporal_span_days,
            }
            for cluster in clusters
        ]
        edges: list[dict[str, Any]] = []
        for left_index, left in enumerate(clusters):
            left_vector = _normalize_vector(left.centroid, target_dim=self._faiss_vector_dim)
            for right in clusters[left_index + 1 :]:
                right_vector = _normalize_vector(right.centroid, target_dim=self._faiss_vector_dim)
                similarity = float(np.dot(left_vector, right_vector))
                if similarity < 0.55:
                    continue
                edges.append(
                    {
                        "source": left.id,
                        "target": right.id,
                        "similarity": round(similarity, 6),
                    }
                )
        return {
            "nodes": nodes,
            "edges": edges,
            "generated_at": _utc_now().isoformat(),
        }

    def get_mandala_data(self) -> dict[str, Any]:
        now = time.monotonic()
        if self._mandala_cache is not None and (now - self._mandala_cache_at) < 30.0:
            return json.loads(json.dumps(self._mandala_cache, default=_json_default))
        data = self._build_mandala_data()
        self._mandala_cache = data
        self._mandala_cache_at = now
        return json.loads(json.dumps(data, default=_json_default))

    def _candidate_media_by_ids(self, media_ids: list[str]) -> list[SmritiMedia]:
        if not media_ids:
            return []
        placeholders = ", ".join("?" for _ in media_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM smriti_media WHERE id IN ({placeholders})",
                tuple(media_ids),
            ).fetchall()
        row_map = {row["id"]: self._row_to_media(row) for row in rows}
        return [row_map[media_id] for media_id in media_ids if media_id in row_map]

    def _update_media_location_counts(self) -> None:
        with self._connect() as connection:
            rows = connection.execute("SELECT id FROM smriti_locations").fetchall()
            for row in rows:
                location_id = row["id"]
                count = connection.execute(
                    "SELECT COUNT(*) AS count FROM smriti_media WHERE location_id = ?",
                    (location_id,),
                ).fetchone()["count"]
                connection.execute(
                    """
                    UPDATE smriti_locations
                    SET media_count = ?
                    WHERE id = ?
                    """,
                    (int(count), location_id),
                )
            connection.commit()


__all__ = [
    "AnchorMatch",
    "DepthStrataMap",
    "RecallResult",
    "SetuDescription",
    "SmritiCluster",
    "SmritiLocation",
    "SmritiMedia",
    "SmritiPerson",
    "SmetiDB",
]
