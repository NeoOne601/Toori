"""
Smriti Data Migration Service.

Safely copies all Smriti data to a new directory without deleting the source.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Callable, Optional

from cloud.runtime.config import resolve_smriti_storage
from cloud.runtime.models import (
    SmritiMigrationProgress,
    SmritiMigrationRequest,
    SmritiMigrationResult,
    SmritiStorageConfig,
)
from cloud.runtime.observability import get_logger

log = get_logger("migration")


def _file_md5(path: Path, chunk_size: int = 65536) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _human(num_bytes: int) -> str:
    value = float(max(num_bytes, 0))
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def _dir_file_count_and_bytes(path: Path) -> tuple[int, int]:
    count = 0
    total = 0
    if not path.exists():
        return count, total
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            total += item.stat().st_size
            count += 1
        except OSError:
            continue
    return count, total


def migrate_smriti_data(
    request: SmritiMigrationRequest,
    current_config: SmritiStorageConfig,
    base_data_dir: str,
    on_progress: Optional[Callable[[SmritiMigrationProgress], None]] = None,
) -> SmritiMigrationResult:
    """
    Execute Smriti data migration.

    This is a synchronous copy/verify flow and should run in a worker thread.
    """

    resolved_src = current_config.resolve_paths(base_data_dir)

    src_data_dir = Path(resolved_src.data_dir or Path(base_data_dir) / "smriti").expanduser().resolve()
    src_db = src_data_dir / "smriti.sqlite3"
    src_faiss = src_data_dir / "smriti_faiss.index"
    src_frames = Path(resolved_src.frames_dir or src_data_dir / "frames").expanduser().resolve()
    src_thumbs = Path(resolved_src.thumbs_dir or src_data_dir / "thumbs").expanduser().resolve()
    src_templates = Path(resolved_src.templates_path or src_data_dir / "sag_templates.json").expanduser().resolve()

    tgt_data_dir = Path(request.target_data_dir).expanduser().resolve()
    target_config = resolve_smriti_storage(
        type("SettingsCarrier", (), {"smriti_storage": SmritiStorageConfig(
            data_dir=str(tgt_data_dir),
            frames_dir=request.target_frames_dir,
            thumbs_dir=request.target_thumbs_dir,
            templates_path=request.target_templates_path,
            max_storage_gb=current_config.max_storage_gb,
            watch_folders=list(current_config.watch_folders),
            store_full_frames=current_config.store_full_frames,
            thumbnail_max_dim=current_config.thumbnail_max_dim,
            auto_prune_missing=current_config.auto_prune_missing,
        )})(),
        base_data_dir,
    )
    tgt_data_dir = Path(target_config.data_dir or request.target_data_dir).expanduser().resolve()
    tgt_frames = Path(target_config.frames_dir or tgt_data_dir / "frames").expanduser().resolve()
    tgt_thumbs = Path(target_config.thumbs_dir or tgt_data_dir / "thumbs").expanduser().resolve()
    tgt_templates = Path(target_config.templates_path or tgt_data_dir / "sag_templates.json").expanduser().resolve()
    tgt_db = tgt_data_dir / "smriti.sqlite3"
    tgt_faiss = tgt_data_dir / "smriti_faiss.index"

    files_moved = 0
    bytes_moved = 0
    errors: list[str] = []

    def _report(status: str, *, current_file: str | None = None, error: str | None = None) -> None:
        progress = SmritiMigrationProgress(
            status=status,
            files_moved=files_moved,
            files_total=total_files,
            bytes_moved=bytes_moved,
            bytes_total=total_bytes,
            bytes_moved_human=_human(bytes_moved),
            bytes_total_human=_human(total_bytes),
            current_file=current_file,
            error=error,
            dry_run=request.dry_run,
        )
        log.info("migration_progress", **progress.model_dump())
        if on_progress is not None:
            on_progress(progress)

    source_paths = [src_db, src_db.with_name(f"{src_db.name}-wal"), src_db.with_name(f"{src_db.name}-shm"), src_faiss, src_templates]
    total_files = 0
    total_bytes = 0
    for path in source_paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            total_files += 1
            total_bytes += path.stat().st_size
        except OSError:
            continue
    for directory in (src_frames, src_thumbs):
        count, size = _dir_file_count_and_bytes(directory)
        total_files += count
        total_bytes += size

    _report("preparing")

    if request.dry_run:
        return SmritiMigrationResult(
            success=True,
            dry_run=True,
            files_moved=total_files,
            bytes_moved=total_bytes,
            bytes_moved_human=_human(total_bytes),
            new_data_dir=str(tgt_data_dir),
            errors=[],
            rollback_available=src_data_dir.exists(),
        )

    try:
        tgt_data_dir.mkdir(parents=True, exist_ok=True)
        tgt_frames.mkdir(parents=True, exist_ok=True)
        tgt_thumbs.mkdir(parents=True, exist_ok=True)
        tgt_templates.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return SmritiMigrationResult(
            success=False,
            dry_run=False,
            files_moved=0,
            bytes_moved=0,
            bytes_moved_human="0 B",
            new_data_dir=str(tgt_data_dir),
            errors=[f"Cannot write to target: {exc}"],
            rollback_available=src_data_dir.exists(),
        )

    def _copy_verified(src: Path, dst: Path, status: str) -> None:
        nonlocal files_moved, bytes_moved
        if not src.exists() or not src.is_file():
            return
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if _file_md5(src) != _file_md5(dst):
                raise IOError("checksum mismatch after copy")
            size = dst.stat().st_size
            files_moved += 1
            bytes_moved += size
            _report(status, current_file=str(src.name))
        except Exception as exc:  # pragma: no cover - exercised via result errors
            message = f"Failed to copy {src} -> {dst}: {exc}"
            errors.append(message)
            _report("failed", current_file=str(src.name), error=message)

    def _copy_tree(src_dir: Path, dst_dir: Path, status: str) -> None:
        if not src_dir.exists():
            return
        for src_file in src_dir.rglob("*"):
            if not src_file.is_file():
                continue
            relative = src_file.relative_to(src_dir)
            _copy_verified(src_file, dst_dir / relative, status)

    _report("migrating_db")
    _copy_verified(src_db, tgt_db, "migrating_db")
    _copy_verified(src_db.with_name(f"{src_db.name}-wal"), tgt_db.with_name(f"{tgt_db.name}-wal"), "migrating_db")
    _copy_verified(src_db.with_name(f"{src_db.name}-shm"), tgt_db.with_name(f"{tgt_db.name}-shm"), "migrating_db")

    _report("migrating_faiss")
    _copy_verified(src_faiss, tgt_faiss, "migrating_faiss")

    _report("migrating_frames")
    _copy_tree(src_frames, tgt_frames, "migrating_frames")

    _report("migrating_thumbs")
    _copy_tree(src_thumbs, tgt_thumbs, "migrating_thumbs")

    _report("migrating_templates")
    _copy_verified(src_templates, tgt_templates, "migrating_templates")

    success = not errors
    _report("complete" if success else "failed")

    return SmritiMigrationResult(
        success=success,
        dry_run=False,
        files_moved=files_moved,
        bytes_moved=bytes_moved,
        bytes_moved_human=_human(bytes_moved),
        new_data_dir=str(tgt_data_dir),
        errors=errors,
        rollback_available=src_data_dir.exists(),
    )
