# SMRITI BUILD SYSTEM — SPRINT 5: PRODUCTION POLISH
# Upstream checkpoint: Sprint 4 complete — 162 tests passing

## MISSION

Sprint 5 closes all known production gaps and delivers the complete
Smriti experience: interactive Mandala, wired Deepdive, full Person
Journal, Setu-2 personalization, data migration, and WCAG 2.1 AA
accessibility. Every component must be production-grade, not a
prototype.

Target hardware: M1 iMac, 8GB unified memory.
Target: ≥ 195 tests passing (162 baseline + 33 new).
Spawn SUB AGENTS to distribute the task for completing this sprint.
---

## MANDATORY FIRST ACTIONS
```bash
# 1. Read every file you will touch — no exceptions
cat cloud/api/tests/test_cwma_ecgd_setu2.py    # Where telescope test lives
cat cloud/runtime/setu2.py                      # W-matrix update exists?
cat cloud/runtime/models.py                     # SmritiStorageConfig, StorageUsageReport
cat cloud/runtime/service.py                    # All smriti_ methods
cat cloud/runtime/app.py                        # All routes
cat cloud/runtime/smriti_storage.py            # SmetiDB methods
cat cloud/runtime/smriti_ingestion.py          # Ingestion daemon
cat cloud/runtime/config.py                    # resolve_smriti_storage
cat cloud/jepa_service/engine.py               # JEPATick fields
cat cloud/jepa_service/anchor_graph.py         # AnchorMatch
cat cloud/jepa_service/depth_separator.py      # DepthStrataMap
cat cloud/runtime/atlas.py                     # EpistemicAtlas
cat desktop/electron/src/components/smriti/MandalaView.tsx
cat desktop/electron/src/components/smriti/DeepdiveView.tsx
cat desktop/electron/src/components/smriti/PersonJournal.tsx
cat desktop/electron/src/components/smriti/RecallSurface.tsx
cat desktop/electron/src/components/smriti/PerformanceHUD.tsx
cat desktop/electron/src/hooks/useSmritiState.ts
cat desktop/electron/src/types.ts
cat desktop/electron/src/styles.css
cat desktop/electron/src/constants.ts
cat .smriti_build/state.json
cat requirements.txt

# 2. Establish 162-test baseline — record BEFORE any change
pytest -q cloud/api/tests cloud/jepa_service/tests \
       cloud/search_service/tests cloud/monitoring/tests \
       tests/test_readme.py
# MUST show: 162 passed, 0 failed
# If different: STOP. Report to operator. Do not proceed.

# 3. Confirm test_smriti_production.py status
ls cloud/api/tests/test_smriti_production.py 2>&1
# If "No such file": agent production_gate must create it FIRST

# 4. Confirm Setu-2 feedback hook
grep -n "update_metric_w\|feedback\|positive_pairs" cloud/runtime/setu2.py
grep -n "update_metric_w\|feedback" cloud/runtime/service.py
# Confirm W-matrix update is NOT called from service.py yet

# 5. Check force worker existence
ls desktop/electron/src/components/smriti/mandala-force-worker.ts 2>&1
# If missing: mandala_enhancement agent must create it

# 6. Initialize sprint state
python3.11 -c "
import json, pathlib
s = json.loads(pathlib.Path('.smriti_build/state.json').read_text())
s.update({
    'version': '5.0',
    'sprint': '5',
    'build_complete': False,
    'baseline_tests_passing': 162,
    'target_tests_passing': 195,
    'agents': {
        'production_gate': 'pending',
        'data_migration': 'pending',
        'setu2_feedback': 'pending',
        'mandala_enhancement': 'pending',
        'deepdive_enhancement': 'pending',
        'person_journal_enhancement': 'pending',
        'accessibility_pass': 'pending',
        'smriti_styles_sprint5': 'pending',
        'sprint5_validation': 'pending',
    },
    'blocking_issues': [],
})
pathlib.Path('.smriti_build/state.json').write_text(json.dumps(s, indent=2))
print('State initialized')
"
```

---

## EXECUTION DAG
```
LAYER_0 — PRODUCTION INFRASTRUCTURE (no UI dependencies)
  production_gate          ← Creates test_smriti_production.py
  data_migration           ← Migration service + API
  setu2_feedback           ← W-matrix update from user interactions

LAYER_1 — UI ENHANCEMENTS (all depend on LAYER_0)
  mandala_enhancement      ← Web Worker + expansion + LOD
  deepdive_enhancement     ← Patch clicking + entity threads
  person_journal_enhancement ← Co-occurrence graph + timeline

LAYER_2 — POLISH (depends on LAYER_1)
  accessibility_pass       ← WCAG 2.1 AA for all Smriti components
  smriti_styles_sprint5    ← CSS additions for new UI states

LAYER_3 — VALIDATION
  sprint5_validation       ← Full suite + Lighthouse + perf
```

---

## FAILURE PROTOCOL

1. After EVERY agent: run `pytest -q` — count must be ≥ 162.
2. Any count drop → STOP immediately. Fix regression before continuing.
3. TypeScript errors → STOP. Fix before next agent.
4. NEVER mark an agent complete by skipping its success gate.
5. production_gate MUST be the first agent executed.
   If it fails, no other agent may proceed.

---

## AGENT SPECIFICATIONS

---

### AGENT: production_gate

**Reads**: `cloud/api/tests/test_cwma_ecgd_setu2.py`, all existing
test files, `cloud/jepa_service/engine.py`, `cloud/runtime/app.py`

**Objective**: Create `cloud/api/tests/test_smriti_production.py` —
the definitive production acceptance gate. This file must contain
ALL critical regression tests in one authoritative location.

**Success gate**:
```bash
pytest -q cloud/api/tests/test_smriti_production.py -v
# ALL tests must pass
pytest -q cloud/api/tests cloud/jepa_service/tests
# Count must be ≥ 162 (no regression from adding these tests)
```

**Create** `cloud/api/tests/test_smriti_production.py`:
```python
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
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cloud.runtime.app import create_app


# ── Helpers ────────────────────────────────────────────────────────────────

def _png_b64(color: tuple[int, int, int], size: int = 32) -> str:
    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ══════════════════════════════════════════════════════════════════
# SECTION 1: SEMANTIC PRECISION — The Core Promise
# ══════════════════════════════════════════════════════════════════

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
    from cloud.jepa_service.anchor_graph import SemanticAnchorGraph, AnchorMatch
    from cloud.jepa_service.depth_separator import DepthStrataMap
    from cloud.jepa_service.confidence_gate import EpistemicConfidenceGate
    from cloud.runtime.setu2 import Setu2Bridge

    # Build a custom depth strata simulating:
    # - Person: foreground (top-left 7×7 quadrant)
    # - Telescope: background (bottom-right quadrant, vertical linear arrangement)
    h, w = 14, 14
    fg = np.zeros((h, w), dtype=bool)
    bg = np.zeros((h, w), dtype=bool)
    fg[:7, :7] = True     # Person region
    bg[7:, 7:] = True     # Telescope/background region
    mid = ~fg & ~bg

    custom_strata = DepthStrataMap(
        depth_proxy=np.where(fg, 2.0, np.where(bg, 0.1, 0.7)).astype(np.float32),
        foreground_mask=fg,
        midground_mask=mid,
        background_mask=bg,
        confidence=0.88,
        strata_entropy=1.3,
    )

    # Telescope patches: vertical linear arrangement in background quadrant
    telescope_patches = [7 * 14 + 10, 8 * 14 + 10, 9 * 14 + 10,
                         10 * 14 + 10, 11 * 14 + 10]

    from cloud.jepa_service.anchor_graph import _rowcol_to_patch
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

    # STRICT BODY PART EXCLUSION
    body_part_words = [
        "shoulder", "arm", "torso", "wig", "hair", "neck",
        "head", "ear", "elbow", "wrist", "hand", "body",
        "seatbelt", "seat belt", "harness", "strap",
    ]
    desc_lower = desc.text.lower()
    for word in body_part_words:
        assert word not in desc_lower, (
            f"PRODUCTION FAILURE: Telescope described as body part.\n"
            f"  Description: '{desc.text}'\n"
            f"  Forbidden word found: '{word}'\n"
            f"  This indicates TPDS/SAG/CWMA/ECGD pipeline failure."
        )

    # MUST indicate background or cylindrical
    assert ("background" in desc_lower or "cylindrical" in desc_lower
            or "object" in desc_lower), (
        f"Description lacks expected spatial context: '{desc.text}'"
    )

    # DEPTH STRATUM must be background
    assert desc.depth_stratum == "background", (
        f"Telescope must be background, got: '{desc.depth_stratum}'"
    )

    # Hallucination risk must be bounded
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

    chair = next(t for t in BOOTSTRAP_TEMPLATES if t.name == "chair_seated")
    cyl = next(t for t in BOOTSTRAP_TEMPLATES if t.name == "cylindrical_object")

    # Chair must have ABOVE edge (seat above base) or BESIDE (armrest)
    chair_relations = {r for _, _, r in chair.edges}
    assert len(chair_relations) > 0, "Chair must have edge relations"

    # Chair must NOT prefer background — it's midground (near person)
    assert chair.depth_preference in ("midground", "any"), (
        f"Chair depth preference wrong: {chair.depth_preference}"
    )

    # Cylindrical must prefer background
    assert cyl.depth_preference == "background", (
        f"Cylindrical must prefer background, got: {cyl.depth_preference}"
    )


def test_uncertain_region_returns_uncertainty_not_wrong_description():
    """
    ECGD gate failure must yield uncertainty visualization, NOT a
    confident wrong answer. This is the core epistemic safety property.
    """
    from cloud.jepa_service.anchor_graph import AnchorMatch
    from cloud.jepa_service.depth_separator import DepthStrataMap
    from cloud.jepa_service.confidence_gate import EpistemicConfidenceGate
    from cloud.runtime.setu2 import Setu2Bridge

    # Deliberately low-confidence match
    low_conf_match = AnchorMatch(
        template_name="unknown",
        confidence=0.10,   # Far below TAU_ANCHOR = 0.55
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
        energy_history=[1.0, 0.5, 0.8, 1.2, 0.9],  # High variance
    )

    assert gate_result.passes is False, "Low-confidence region must fail gate"
    assert gate_result.safe_embedding is None, "Failed gate must not provide safe embedding"
    assert gate_result.uncertainty_map is not None

    bridge = Setu2Bridge()
    desc = bridge.describe_region(gate_result)

    assert desc.is_uncertain is True, "Failed gate must produce uncertain description"
    assert desc.uncertainty_map is not None, "Uncertain description must include uncertainty map"

    # Must NOT produce a confident-sounding description
    confident_words = [
        "person", "chair", "telescope", "desk", "screen",
        "definitely", "clearly", "obviously",
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
    frame[30:120, 30:120, 0] = 200  # Red foreground region
    frame[140:200, 140:200, 2] = 180  # Blue background region

    tick = engine.tick(frame, session_id="production_gate", observation_id="obs_pg")

    assert tick.depth_strata is not None, "depth_strata must be populated"
    assert tick.anchor_matches is not None, "anchor_matches must be populated"
    assert tick.setu_descriptions is not None, "setu_descriptions must be populated"

    descriptions = tick.setu_descriptions
    assert isinstance(descriptions, list)

    if descriptions:
        risks = [d.get("description", {}).get("hallucination_risk", 1.0)
                 for d in descriptions]
        assert all(r < 0.80 for r in risks), (
            f"Some descriptions have unacceptably high hallucination risk: {risks}"
        )
        assert any(r < 0.50 for r in risks), (
            f"No description has low hallucination risk: {risks}"
        )


# ══════════════════════════════════════════════════════════════════
# SECTION 2: PERFORMANCE — The No-Lag Promise
# ══════════════════════════════════════════════════════════════════

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
        resp = client.post("/v1/living-lens/tick", json={
            "image_base64": _png_b64(color, size=64),
            "session_id": "perf_gate",
            "decode_mode": "off",
            "proof_mode": "both",
        })
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        assert resp.status_code == 200

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
    resp = client.post("/v1/smriti/recall", json={
        "query": "red jacket Kolkata",
        "top_k": 10,
    })
    elapsed = (time.perf_counter() - start) * 1000

    assert resp.status_code == 200
    assert elapsed < 1000, f"Recall took {elapsed:.0f}ms, exceeds 1000ms SLA"


# ══════════════════════════════════════════════════════════════════
# SECTION 3: STORAGE — User Safety Guarantee
# ══════════════════════════════════════════════════════════════════

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
    container1.update_settings(settings1.model_copy(
        update={"smriti_storage": updated_storage}
    ))

    # Simulate restart: new container, same data_dir
    container2 = RuntimeContainer(data_dir=str(tmp_path))
    settings2 = container2.get_settings()

    assert settings2.smriti_storage.max_storage_gb == 42.5, (
        "Storage budget not persisted across restart"
    )
    assert settings2.smriti_storage.store_full_frames is False, (
        "store_full_frames not persisted across restart"
    )


def test_watch_folder_list_persists_across_restart(tmp_path):
    """Watch folders must persist in smriti_storage.watch_folders."""
    from cloud.runtime.service import RuntimeContainer
    from pathlib import Path

    watch_dir = tmp_path / "my_photos"
    watch_dir.mkdir()

    container1 = RuntimeContainer(data_dir=str(tmp_path))
    container1.add_watch_folder(str(watch_dir))

    container2 = RuntimeContainer(data_dir=str(tmp_path))
    settings2 = container2.get_settings()
    assert str(watch_dir) in settings2.smriti_storage.watch_folders, (
        "Watch folder not persisted across restart"
    )


def test_prune_clear_all_requires_exact_confirmation(tmp_path):
    """Clear-all prune must reject any confirmation string except exact match."""
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    for bad_confirm in ["yes", "confirm", "CONFIRM", "", "clear all", "CLEAR_ALL"]:
        resp = client.post("/v1/smriti/storage/prune", json={
            "clear_all": True,
            "confirm_clear_all": bad_confirm,
        })
        assert resp.status_code in (400, 422, 500), (
            f"Prune clear-all accepted bad confirmation: '{bad_confirm}'"
        )


# ══════════════════════════════════════════════════════════════════
# SECTION 4: REGRESSION — Nothing Broke
# ══════════════════════════════════════════════════════════════════

def test_existing_toori_functionality_not_broken():
    """
    Run the original Toori test suite as a subprocess.
    Smriti must not break any pre-existing functionality.
    """
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q",
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
        f"REGRESSION in original Toori tests:\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def test_no_torch_imports_outside_perception():
    """torch must never be imported outside cloud/perception/."""
    import importlib
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
    # Import each module in a fresh subprocess to avoid contamination
    for mod in modules_to_check:
        result = subprocess.run(
            [sys.executable, "-c",
             f"import sys; import {mod}; "
             f"assert 'torch' not in sys.modules, "
             f"'torch imported by {mod}'"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            violating_modules.append(mod)

    assert not violating_modules, (
        f"torch imported outside cloud/perception/ by: {violating_modules}"
    )


def test_api_routes_all_present(tmp_path):
    """All expected Smriti API routes must be registered."""
    app = create_app(data_dir=str(tmp_path))
    routes = {r.path for r in app.routes}

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

    missing = [r for r in required_routes if r not in routes]
    assert not missing, f"Missing API routes: {missing}"
```

---

### AGENT: data_migration

**Reads**: `cloud/runtime/service.py`, `cloud/runtime/smriti_storage.py`,
`cloud/runtime/config.py`, `cloud/runtime/models.py`,
`cloud/runtime/app.py`

**Objective**: Implement a data migration service that safely moves
SmetiDB, FAISS index, frames, thumbnails, and learned templates to
a new directory when the user changes `data_dir` in SmritiStorageConfig.

**Success gate**:
```bash
pytest -q cloud/api/tests/test_smriti_data_migration.py -v
# All 10 tests must pass
pytest -q cloud/api/tests cloud/jepa_service/tests
# Count must be ≥ 162 (no regression)
```

**Deliverable 1**: Add to `cloud/runtime/models.py` after `SmritiPruneResult`:
```python
class SmritiMigrationRequest(BaseModel):
    """Request to migrate Smriti data to a new directory."""
    target_data_dir: str
    target_frames_dir: Optional[str] = None
    target_thumbs_dir: Optional[str] = None
    target_templates_path: Optional[str] = None
    dry_run: bool = False   # If True, report what WOULD be moved without moving


class SmritiMigrationProgress(BaseModel):
    """Real-time migration progress (returned during migration)."""
    status: str  # "preparing"|"migrating_db"|"migrating_frames"|
                 # "migrating_thumbs"|"migrating_faiss"|
                 # "migrating_templates"|"updating_config"|"complete"|"failed"
    files_moved: int = 0
    files_total: int = 0
    bytes_moved: int = 0
    bytes_total: int = 0
    bytes_moved_human: str = "0 B"
    bytes_total_human: str = "0 B"
    current_file: Optional[str] = None
    error: Optional[str] = None
    dry_run: bool = False


class SmritiMigrationResult(BaseModel):
    """Final result of a completed migration."""
    success: bool
    dry_run: bool
    files_moved: int
    bytes_moved: int
    bytes_moved_human: str
    new_data_dir: str
    errors: list[str] = Field(default_factory=list)
    rollback_available: bool = False  # True if original data still exists
```

**Deliverable 2**: Create `cloud/runtime/smriti_migration.py`:
```python
"""
Smriti Data Migration Service.

Safely moves all Smriti data (SmetiDB, FAISS index, frames,
thumbnails, learned templates) to a new directory.

Safety guarantees:
  1. NEVER deletes source data until destination is verified
  2. Atomic file-by-file copy with verification
  3. Config update happens LAST (after all files verified)
  4. If any step fails, source data is preserved
  5. dry_run mode shows exactly what would happen without doing it

Migration sequence:
  1. Validate target directory is writable
  2. Copy SQLite database (WAL mode — safe hot copy)
  3. Copy FAISS index file
  4. Copy frames directory (recursive)
  5. Copy thumbnails directory (recursive)
  6. Copy learned templates JSON
  7. Verify all copies are readable
  8. Update RuntimeSettings.smriti_storage paths
  9. Report success (source data preserved for user to delete manually)
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Callable, Iterator, Optional

from cloud.runtime.models import (
    SmritiMigrationRequest,
    SmritiMigrationResult,
    SmritiMigrationProgress,
)
from cloud.runtime.observability import get_logger

log = get_logger("migration")


def _file_md5(path: Path, chunk_size: int = 65536) -> str:
    """Compute MD5 of a file for integrity verification."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _dir_file_count_and_bytes(path: Path) -> tuple[int, int]:
    """Count files and total bytes in a directory tree."""
    count, total = 0, 0
    if not path.exists():
        return 0, 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
                count += 1
            except OSError:
                pass
    return count, total


def _human(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b //= 1024
    return f"{b:.1f} PB"


def migrate_smriti_data(
    request: SmritiMigrationRequest,
    current_config: "SmritiStorageConfig",
    base_data_dir: str,
    on_progress: Optional[Callable[[SmritiMigrationProgress], None]] = None,
) -> SmritiMigrationResult:
    """
    Execute Smriti data migration.
    
    This function is synchronous and blocking — run it in
    asyncio.get_event_loop().run_in_executor() for the API endpoint.
    
    on_progress: called with SmritiMigrationProgress after each stage.
    """
    errors: list[str] = []
    files_moved = 0
    bytes_moved = 0

    def _report(status: str, **kwargs) -> None:
        if on_progress:
            on_progress(SmritiMigrationProgress(
                status=status,
                files_moved=files_moved,
                bytes_moved=bytes_moved,
                bytes_moved_human=_human(bytes_moved),
                dry_run=request.dry_run,
                **kwargs,
            ))
        log.info("migration_progress", status=status, **kwargs)

    # ── 1. Resolve source paths ────────────────────────────────────────────
    from cloud.runtime.config import resolve_smriti_storage
    from cloud.runtime.models import RuntimeSettings
    resolved_src = current_config.resolve_paths(base_data_dir)

    src_data_dir = Path(resolved_src.data_dir)
    src_db = src_data_dir / "smriti.sqlite3"
    src_faiss = src_data_dir / "smriti_faiss.index"
    src_frames = Path(resolved_src.frames_dir)
    src_thumbs = Path(resolved_src.thumbs_dir)
    src_templates = Path(resolved_src.templates_path)

    # ── 2. Resolve target paths ────────────────────────────────────────────
    tgt_data_dir = Path(request.target_data_dir).expanduser().resolve()
    tgt_frames = (
        Path(request.target_frames_dir).expanduser().resolve()
        if request.target_frames_dir
        else tgt_data_dir / "frames"
    )
    tgt_thumbs = (
        Path(request.target_thumbs_dir).expanduser().resolve()
        if request.target_thumbs_dir
        else tgt_data_dir / "thumbs"
    )
    tgt_templates = (
        Path(request.target_templates_path).expanduser().resolve()
        if request.target_templates_path
        else tgt_data_dir / "sag_templates.json"
    )
    tgt_db = tgt_data_dir / "smriti.sqlite3"
    tgt_faiss = tgt_data_dir / "smriti_faiss.index"

    # ── 3. Compute migration scope ─────────────────────────────────────────
    _report("preparing")

    total_files = 0
    total_bytes = 0

    def _count(src: Path) -> None:
        nonlocal total_files, total_bytes
        if src.is_file():
            try:
                total_bytes += src.stat().st_size
                total_files += 1
            except OSError:
                pass
        elif src.is_dir():
            c, b = _dir_file_count_and_bytes(src)
            total_files += c
            total_bytes += b

    for src in (src_db, src_faiss, src_frames, src_thumbs, src_templates):
        _count(src)

    if request.dry_run:
        return SmritiMigrationResult(
            success=True,
            dry_run=True,
            files_moved=total_files,
            bytes_moved=total_bytes,
            bytes_moved_human=_human(total_bytes),
            new_data_dir=str(tgt_data_dir),
            errors=[],
            rollback_available=True,
        )

    # ── 4. Validate target writability ─────────────────────────────────────
    try:
        tgt_data_dir.mkdir(parents=True, exist_ok=True)
        tgt_frames.mkdir(parents=True, exist_ok=True)
        tgt_thumbs.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        return SmritiMigrationResult(
            success=False,
            dry_run=False,
            files_moved=0,
            bytes_moved=0,
            bytes_moved_human="0 B",
            new_data_dir=str(tgt_data_dir),
            errors=[f"Cannot write to target: {exc}"],
        )

    def _copy_file(src: Path, dst: Path) -> bool:
        nonlocal files_moved, bytes_moved
        if not src.exists():
            return True  # Nothing to copy — not an error
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(str(src), str(dst))
            size = dst.stat().st_size
            files_moved += 1
            bytes_moved += size
            return True
        except Exception as exc:
            errors.append(f"Failed to copy {src} → {dst}: {exc}")
            return False

    def _copy_dir(src: Path, dst: Path) -> None:
        if not src.exists():
            return
        for item in src.rglob("*"):
            if item.is_file():
                relative = item.relative_to(src)
                _copy_file(item, dst / relative)
                _report(
                    "migrating",
                    current_file=str(item.name),
                    files_total=total_files,
                    bytes_total=total_bytes,
                )

    # ── 5. Execute migration stages ────────────────────────────────────────
    _report("migrating_db", files_total=total_files, bytes_total=total_bytes)
    _copy_file(src_db, tgt_db)
    # Copy WAL and SHM files if they exist
    for ext in ("-wal", "-shm"):
        _copy_file(src_db.with_suffix(src_db.suffix + ext),
                   tgt_db.with_suffix(tgt_db.suffix + ext))

    _report("migrating_faiss")
    _copy_file(src_faiss, tgt_faiss)

    _report("migrating_frames")
    _copy_dir(src_frames, tgt_frames)

    _report("migrating_thumbs")
    _copy_dir(src_thumbs, tgt_thumbs)

    _report("migrating_templates")
    _copy_file(src_templates, tgt_templates)

    # ── 6. Verify destination DB is readable ───────────────────────────────
    if tgt_db.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(tgt_db))
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
            conn.close()
        except Exception as exc:
            errors.append(f"Destination DB verification failed: {exc}")

    success = len(errors) == 0

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
```

**Deliverable 3**: Add to `cloud/runtime/service.py`:
```python
def migrate_smriti_data(
    self,
    request: "SmritiMigrationRequest",
) -> "SmritiMigrationResult":
    """
    Migrate Smriti data to a new directory.
    Config is updated AFTER successful migration.
    Original data is preserved (user deletes manually).
    """
    from cloud.runtime.smriti_migration import migrate_smriti_data as _migrate
    from cloud.runtime.config import resolve_smriti_storage

    settings = self.get_settings()
    current_config = settings.smriti_storage

    def _progress(p: "SmritiMigrationProgress") -> None:
        self.events.publish("smriti.migration_progress", p.model_dump())

    result = _migrate(
        request=request,
        current_config=current_config,
        base_data_dir=str(self.data_dir),
        on_progress=_progress,
    )

    # Update config to point to new location (only if success + not dry_run)
    if result.success and not result.dry_run:
        from cloud.runtime.models import SmritiStorageConfig
        new_storage = current_config.model_copy(update={
            "data_dir": request.target_data_dir,
            "frames_dir": request.target_frames_dir,
            "thumbs_dir": request.target_thumbs_dir,
            "templates_path": request.target_templates_path,
        })
        self.update_settings(settings.model_copy(
            update={"smriti_storage": new_storage}
        ))

    return result
```

**Deliverable 4**: Add to `cloud/runtime/app.py` (inside `create_app()`):
```python
    @app.post("/v1/smriti/storage/migrate", dependencies=[Depends(require_auth)])
    async def migrate_smriti_storage(
        payload: SmritiMigrationRequest
    ) -> SmritiMigrationResult:
        """
        Migrate Smriti data to a new directory.
        Use dry_run=True first to see what will be moved.
        Original data is preserved after migration.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            app.state.runtime.migrate_smriti_data,
            payload,
        )
```

**Deliverable 5**: Create `cloud/api/tests/test_smriti_data_migration.py`:
```python
"""Tests for Smriti data migration service."""
import shutil
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from cloud.runtime.app import create_app
from cloud.runtime.models import SmritiMigrationRequest


def _make_fake_smriti_data(data_dir: Path) -> None:
    """Create minimal fake Smriti data structure for migration tests."""
    smriti = data_dir / "smriti"
    smriti.mkdir(parents=True, exist_ok=True)
    (smriti / "frames").mkdir(exist_ok=True)
    (smriti / "thumbs").mkdir(exist_ok=True)
    (smriti / "smriti.sqlite3").write_bytes(b"SQLite format 3\x00" + b"\x00" * 100)
    (smriti / "sag_templates.json").write_text("[]")
    (smriti / "frames" / "obs_001.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
    (smriti / "thumbs" / "obs_001.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)


def test_migration_dry_run_reports_what_would_be_moved(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    _make_fake_smriti_data(tmp_path)
    client = TestClient(app)
    target = str(tmp_path / "new_location")
    resp = client.post("/v1/smriti/storage/migrate", json={
        "target_data_dir": target,
        "dry_run": True,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["success"] is True
    assert not Path(target).exists(), "Dry run must not create directories"


def test_migration_copies_files_to_target(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    dst = tmp_path / "destination"
    _make_fake_smriti_data(src)
    app = create_app(data_dir=str(src))
    client = TestClient(app)
    resp = client.post("/v1/smriti/storage/migrate", json={
        "target_data_dir": str(dst),
        "dry_run": False,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["files_moved"] > 0
    assert dst.exists()


def test_migration_preserves_source_data(tmp_path):
    """Source data must survive migration (user deletes manually)."""
    src = tmp_path / "source"
    src.mkdir()
    dst = tmp_path / "destination"
    _make_fake_smriti_data(src)
    smriti_src = src / "smriti"

    app = create_app(data_dir=str(src))
    client = TestClient(app)
    client.post("/v1/smriti/storage/migrate", json={
        "target_data_dir": str(dst),
        "dry_run": False,
    })

    assert smriti_src.exists(), "Source data must be preserved after migration"
    assert (smriti_src / "smriti.sqlite3").exists()


def test_migration_updates_config_on_success(tmp_path):
    """Config must point to new dir after successful migration."""
    src = tmp_path / "source"
    src.mkdir()
    dst = tmp_path / "destination"
    _make_fake_smriti_data(src)

    app = create_app(data_dir=str(src))
    client = TestClient(app)
    resp = client.post("/v1/smriti/storage/migrate", json={
        "target_data_dir": str(dst),
        "dry_run": False,
    })
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    storage_resp = client.get("/v1/smriti/storage")
    config = storage_resp.json()
    assert str(dst) in (config.get("data_dir") or "")


def test_migration_to_nonwritable_directory_fails(tmp_path):
    """Migration to a non-writable location must fail gracefully."""
    import os
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    resp = client.post("/v1/smriti/storage/migrate", json={
        "target_data_dir": "/root/no_permission_here_12345",
        "dry_run": False,
    })
    body = resp.json()
    # Should either fail with 500 or succeed=False
    if resp.status_code == 200:
        assert body["success"] is False
        assert len(body["errors"]) > 0


def test_migration_reports_bytes_moved(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    dst = tmp_path / "destination"
    _make_fake_smriti_data(src)

    app = create_app(data_dir=str(src))
    client = TestClient(app)
    resp = client.post("/v1/smriti/storage/migrate", json={
        "target_data_dir": str(dst),
        "dry_run": False,
    })
    body = resp.json()
    assert body["bytes_moved"] > 0
    assert "B" in body["bytes_moved_human"]


def test_migration_does_not_break_existing_functionality(tmp_path):
    """After migration, existing Toori analyze endpoint still works."""
    import base64, io
    src = tmp_path / "source"
    src.mkdir()
    dst = tmp_path / "destination"

    app = create_app(data_dir=str(src))
    client = TestClient(app)
    client.post("/v1/smriti/storage/migrate", json={
        "target_data_dir": str(dst), "dry_run": False
    })

    img = __import__("PIL.Image", fromlist=["Image"]).Image
    im = img.new("RGB", (32, 32), color=(100, 150, 200))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    resp = client.post("/v1/analyze", json={
        "image_base64": b64,
        "session_id": "post_migration",
        "decode_mode": "off",
    })
    assert resp.status_code == 200


def test_migration_endpoint_registered(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    routes = {r.path for r in app.routes}
    assert "/v1/smriti/storage/migrate" in routes


def test_migration_result_has_rollback_available_flag(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    dst = tmp_path / "destination"
    _make_fake_smriti_data(src)

    app = create_app(data_dir=str(src))
    client = TestClient(app)
    resp = client.post("/v1/smriti/storage/migrate", json={
        "target_data_dir": str(dst), "dry_run": False
    })
    assert "rollback_available" in resp.json()


def test_config_not_updated_on_dry_run(tmp_path):
    """Config must NOT be modified during dry run."""
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    config_before = client.get("/v1/smriti/storage").json()
    client.post("/v1/smriti/storage/migrate", json={
        "target_data_dir": str(tmp_path / "fake_target"),
        "dry_run": True,
    })
    config_after = client.get("/v1/smriti/storage").json()

    assert config_before.get("data_dir") == config_after.get("data_dir"), (
        "Config must not change during dry run"
    )
```

Also add migration UI to `SmritiStorageSettings.tsx` — after the
"Storage Maintenance" section, add a "Data Migration" section:
```tsx
      {/* ── Data Migration ── */}
      <article className="panel">
        <div className="panel-head">
          <div>
            <p className="eyebrow">Data Migration</p>
            <h3>Move Smriti data to a new location</h3>
          </div>
        </div>
        <p className="muted" style={{ fontSize: "0.88rem", marginBottom: "1rem" }}>
          Move all indexed data (database, FAISS index, frames, thumbnails)
          to a new directory. Original data is preserved — you can delete it
          manually after verifying the migration succeeded.
          Use Dry Run first to see what will be moved.
        </p>
        <MigrationPanel onStatusChange={onStatusChange} />
      </article>
```

Create `MigrationPanel` as an inner component within `SmritiStorageSettings.tsx`:
```tsx
function MigrationPanel({ onStatusChange }: { onStatusChange?: (msg: string) => void }) {
  const [targetDir, setTargetDir] = useState("");
  const [migrating, setMigrating] = useState(false);
  const [lastResult, setLastResult] = useState<{
    success: boolean;
    dry_run: boolean;
    files_moved: number;
    bytes_moved_human: string;
    errors: string[];
    rollback_available: boolean;
  } | null>(null);

  const browse = async () => {
    const path = await pickFolderPath();
    if (path) setTargetDir(path);
  };

  const runMigration = async (dryRun: boolean) => {
    if (!targetDir) return;
    setMigrating(true);
    try {
      const result = await runtimeRequest<typeof lastResult>(
        "/v1/smriti/storage/migrate",
        "POST",
        { target_data_dir: targetDir, dry_run: dryRun }
      );
      setLastResult(result);
      if (result?.success) {
        onStatusChange?.(
          dryRun
            ? `Dry run: would move ${result.files_moved} files (${result.bytes_moved_human})`
            : `Migration complete: ${result.files_moved} files moved (${result.bytes_moved_human})`
        );
      } else {
        onStatusChange?.(`Migration failed: ${result?.errors?.join("; ")}`);
      }
    } catch (err) {
      onStatusChange?.(`Migration error: ${(err as Error).message}`);
    } finally {
      setMigrating(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.85rem" }}>
      <label className="field">
        <span>Target Directory</span>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <input
            value={targetDir}
            onChange={(e) => setTargetDir(e.target.value)}
            placeholder="/Volumes/ExternalDrive/TooriSmriti"
            style={{ flex: 1 }}
          />
          <button onClick={browse} style={{ flexShrink: 0, borderRadius: 12 }}>
            Browse…
          </button>
        </div>
      </label>
      <div style={{ display: "flex", gap: "0.65rem" }}>
        <button
          onClick={() => runMigration(true)}
          disabled={!targetDir || migrating}
        >
          Dry Run (preview only)
        </button>
        <button
          className="primary"
          onClick={() => runMigration(false)}
          disabled={!targetDir || migrating}
        >
          {migrating ? "Migrating…" : "Run Migration"}
        </button>
      </div>
      {lastResult && (
        <div style={{
          border: `1px solid ${lastResult.success ? "rgba(67,216,201,0.3)" : "rgba(230,57,70,0.3)"}`,
          borderRadius: 16,
          padding: "0.85rem 1rem",
          background: lastResult.success ? "rgba(67,216,201,0.05)" : "rgba(230,57,70,0.05)",
        }}>
          <strong style={{ color: lastResult.success ? "var(--kpi-healthy)" : "var(--kpi-danger)" }}>
            {lastResult.dry_run ? "Dry Run Result" : lastResult.success ? "Migration Complete" : "Migration Failed"}
          </strong>
          <p style={{ margin: "0.4rem 0 0", fontSize: "0.85rem" }}>
            {lastResult.files_moved} files · {lastResult.bytes_moved_human}
          </p>
          {lastResult.rollback_available && !lastResult.dry_run && (
            <p className="muted" style={{ fontSize: "0.8rem", marginTop: "0.3rem" }}>
              Original data preserved. Delete it manually after verifying.
            </p>
          )}
          {lastResult.errors?.map((e, i) => (
            <p key={i} style={{ color: "var(--kpi-danger)", fontSize: "0.8rem" }}>{e}</p>
          ))}
        </div>
      )}
    </div>
  );
}
```

---

### AGENT: setu2_feedback

**Reads**: `cloud/runtime/setu2.py`, `cloud/runtime/service.py`,
`cloud/runtime/app.py`, `cloud/runtime/models.py`,
`desktop/electron/src/hooks/useSmritiState.ts`,
`desktop/electron/src/components/smriti/RecallSurface.tsx`

**Objective**: Wire the Setu-2 W-matrix feedback loop so that user
confirmations and rejections during recall actually improve
personalization. Add `POST /v1/smriti/recall/feedback` endpoint.
Wire the frontend to call it when user confirms a result.

**Success gate**:
```bash
pytest -q cloud/api/tests/test_setu2_feedback.py -v
# All 7 tests pass
```

**Deliverable 1**: Add to `cloud/runtime/models.py`:
```python
class SmritiRecallFeedback(BaseModel):
    """User feedback on a recall result — used to update Setu-2 W-matrix."""
    query: str                # The original query string
    media_id: str             # The media item being confirmed or rejected
    confirmed: bool           # True=positive pair, False=negative pair
    session_id: str = "default"


class SmritiRecallFeedbackResult(BaseModel):
    updated: bool
    w_mean: float             # Mean of W diagonal after update
    message: str
```

**Deliverable 2**: Add to `cloud/runtime/service.py`:
```python
def smriti_recall_feedback(
    self,
    feedback: "SmritiRecallFeedback",
) -> "SmritiRecallFeedbackResult":
    """
    Update Setu-2 W-matrix based on user confirmation/rejection.
    
    Confirmed results → positive pair (pull query embedding closer
    to media embedding in the JEPA metric space).
    Rejected results → negative pair (push apart).
    
    The W-matrix is persisted in the next settings save cycle.
    """
    from cloud.runtime.setu2 import Setu2Bridge
    import numpy as np

    # Get the query embedding (encode the query string to QUERY_DIM)
    # We use a simple hash-based deterministic embedding for now
    # In production Sprint 6: integrate sentence-transformers
    query_bytes = feedback.query.encode("utf-8")
    rng = np.random.default_rng(int.from_bytes(query_bytes[:8], "little"))
    query_embedding = rng.standard_normal(384).astype(np.float32)
    query_embedding /= np.linalg.norm(query_embedding) + 1e-8

    # Get the media embedding from SmetiDB
    media_embedding: np.ndarray | None = None
    if hasattr(self, "smriti_db") and self.smriti_db is not None:
        try:
            media = self.smriti_db.get_smriti_media(feedback.media_id)
            if media and media.embedding is not None:
                media_embedding = np.array(media.embedding, dtype=np.float32)
        except Exception:
            pass

    if media_embedding is None:
        return SmritiRecallFeedbackResult(
            updated=False,
            w_mean=0.0,
            message="Media embedding not found — feedback ignored",
        )

    # Get or create Setu-2 bridge from container
    if not hasattr(self, "_setu2_bridge"):
        self._setu2_bridge = Setu2Bridge()

    bridge: Setu2Bridge = self._setu2_bridge

    if feedback.confirmed:
        bridge.update_metric_w(
            positive_pairs=[(query_embedding, media_embedding)],
            negative_pairs=[],
            learning_rate=0.005,
        )
    else:
        bridge.update_metric_w(
            positive_pairs=[],
            negative_pairs=[(query_embedding, media_embedding)],
            learning_rate=0.005,
        )

    return SmritiRecallFeedbackResult(
        updated=True,
        w_mean=float(bridge._W.mean()),
        message=(
            f"W-matrix updated. Mean metric weight: {bridge._W.mean():.4f}. "
            f"{'Confirmed' if feedback.confirmed else 'Rejected'} result "
            f"will influence future recall ordering."
        ),
    )
```

**Deliverable 3**: Add to `cloud/runtime/app.py` (inside `create_app()`):
```python
    @app.post("/v1/smriti/recall/feedback", dependencies=[Depends(require_auth)])
    async def recall_feedback(
        payload: SmritiRecallFeedback
    ) -> SmritiRecallFeedbackResult:
        """
        Submit user feedback on a recall result.
        Confirmed results improve Setu-2 precision for similar queries.
        """
        return app.state.runtime.smriti_recall_feedback(payload)
```

**Deliverable 4**: Update `RecallSurface.tsx` — add thumbs-up/thumbs-down
buttons to each recall result card. When clicked, call the feedback endpoint:
```tsx
// Inside the result card rendering in RecallSurface.tsx,
// add this after the existing card content:

const [feedbackSent, setFeedbackSent] = useState<Record<string, boolean | null>>({});

async function sendFeedback(mediaId: string, confirmed: boolean) {
  try {
    await runtimeRequest("/v1/smriti/recall/feedback", "POST", {
      query: query,
      media_id: mediaId,
      confirmed,
      session_id: "smriti",
    });
    setFeedbackSent((prev) => ({ ...prev, [mediaId]: confirmed }));
  } catch {
    // Feedback is best-effort — don't disrupt UX on failure
  }
}

// In the card JSX, add feedback buttons:
// (Only show when not yet submitted)
{feedbackSent[result.media_id] === undefined && (
  <div style={{ display: "flex", gap: "0.35rem", marginTop: "0.4rem" }}>
    <button
      onClick={(e) => { e.stopPropagation(); sendFeedback(result.media_id, true); }}
      style={{ padding: "0.2rem 0.5rem", fontSize: "0.75rem", borderRadius: 999 }}
      title="This result is correct"
      aria-label="Confirm this result is relevant"
    >
      ✓
    </button>
    <button
      onClick={(e) => { e.stopPropagation(); sendFeedback(result.media_id, false); }}
      style={{ padding: "0.2rem 0.5rem", fontSize: "0.75rem", borderRadius: 999,
               color: "var(--text-muted)" }}
      title="This result is not relevant"
      aria-label="Mark this result as not relevant"
    >
      ✗
    </button>
  </div>
)}
{feedbackSent[result.media_id] === true && (
  <span style={{ fontSize: "0.75rem", color: "var(--kpi-healthy)" }}>
    ✓ Confirmed
  </span>
)}
{feedbackSent[result.media_id] === false && (
  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
    Noted
  </span>
)}
```

**Deliverable 5**: Create `cloud/api/tests/test_setu2_feedback.py`:
```python
"""Tests for Setu-2 feedback loop."""
import pytest
from fastapi.testclient import TestClient
from cloud.runtime.app import create_app


def test_feedback_endpoint_registered(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    routes = {r.path for r in app.routes}
    assert "/v1/smriti/recall/feedback" in routes


def test_feedback_confirmed_returns_updated_true_when_media_found(tmp_path):
    """With a real media in DB, confirmed feedback updates W."""
    import base64, io
    from PIL import Image
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    # Create an observation first
    img = Image.new("RGB", (32, 32), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    analyze = client.post("/v1/analyze", json={
        "image_base64": b64, "session_id": "fb_test", "decode_mode": "off"
    })
    obs_id = analyze.json()["observation"]["id"]

    resp = client.post("/v1/smriti/recall/feedback", json={
        "query": "person in office",
        "media_id": obs_id,
        "confirmed": True,
        "session_id": "fb_test",
    })
    assert resp.status_code == 200


def test_feedback_rejected_returns_result(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    resp = client.post("/v1/smriti/recall/feedback", json={
        "query": "test query",
        "media_id": "nonexistent_obs",
        "confirmed": False,
        "session_id": "test",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "updated" in body
    assert "message" in body


def test_feedback_missing_media_id_returns_not_updated(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    resp = client.post("/v1/smriti/recall/feedback", json={
        "query": "anything",
        "media_id": "obs_this_does_not_exist_xyz",
        "confirmed": True,
    })
    assert resp.status_code == 200
    assert resp.json()["updated"] is False


def test_feedback_requires_query_field(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    resp = client.post("/v1/smriti/recall/feedback", json={
        "media_id": "obs_xyz",
        "confirmed": True,
    })
    assert resp.status_code == 422


def test_feedback_result_has_w_mean_field(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    resp = client.post("/v1/smriti/recall/feedback", json={
        "query": "q", "media_id": "obs_fake", "confirmed": True,
    })
    body = resp.json()
    assert "w_mean" in body
    assert isinstance(body["w_mean"], (int, float))


def test_multiple_feedbacks_do_not_crash(tmp_path):
    """Rapid feedback submissions must not cause errors."""
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    for i in range(10):
        resp = client.post("/v1/smriti/recall/feedback", json={
            "query": f"query {i}", "media_id": f"obs_{i}", "confirmed": i % 2 == 0,
        })
        assert resp.status_code == 200
```

---

### AGENT: mandala_enhancement

**Reads**: `desktop/electron/src/components/smriti/MandalaView.tsx`,
`desktop/electron/src/hooks/useSmritiState.ts`,
`desktop/electron/src/types.ts`,
`desktop/electron/src/styles.css`

**Objective**: Enhance MandalaView with:
1. Web Worker for force simulation (never blocks render thread)
2. Animated cluster expansion with spring physics
3. Semantic zoom (LOD: far=superclusters, close=individual clusters)
4. Click handler for cluster expansion

**M1 CONSTRAINT**: Use Canvas 2D API only. No WebGL. No THREE.js.
The Canvas 2D path avoids Electron compositor issues on M1 8GB.

**Success gate**:
```bash
cd desktop/electron && npm run typecheck && npm run build
# Both must exit 0
```

**Deliverable 1**: Create `desktop/electron/src/components/smriti/mandala-force-worker.ts`:
```typescript
/**
 * Mandala Force Simulation Web Worker
 *
 * Runs D3-style force simulation off the main thread.
 * Posts updated cluster positions at ~30fps.
 * Main thread receives positions and re-renders Canvas 2D.
 *
 * Messages received:
 *   { type: "init", clusters: ClusterNode[], width: number, height: number }
 *   { type: "resize", width: number, height: number }
 *   { type: "tick_pause" } | { type: "tick_resume" }
 *   { type: "pin", clusterId: number, x: number, y: number }
 *   { type: "release", clusterId: number }
 *
 * Messages posted:
 *   { type: "positions", positions: Array<[number, number, number]> }
 *   i.e., [clusterId, x, y] tuples
 */

interface WorkerCluster {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  pinned: boolean;
  px?: number;
  py?: number;
}

let clusters: WorkerCluster[] = [];
let width = 800;
let height = 600;
let animating = true;
let rafHandle: ReturnType<typeof setTimeout> | null = null;
const ALPHA_DECAY = 0.0228;
const VELOCITY_DECAY = 0.4;
let alpha = 1.0;

function repulsion(a: WorkerCluster, b: WorkerCluster): void {
  const dx = a.x - b.x || Math.random() * 0.01;
  const dy = a.y - b.y || Math.random() * 0.01;
  const dist = Math.sqrt(dx * dx + dy * dy);
  const minDist = a.radius + b.radius + 8;
  if (dist < minDist && dist > 0) {
    const force = (minDist - dist) / dist * 0.5 * alpha;
    a.vx += dx * force;
    a.vy += dy * force;
    b.vx -= dx * force;
    b.vy -= dy * force;
  }
}

function centerForce(): void {
  const cx = width / 2;
  const cy = height / 2;
  const strength = 0.05 * alpha;
  for (const c of clusters) {
    if (!c.pinned) {
      c.vx += (cx - c.x) * strength;
      c.vy += (cy - c.y) * strength;
    }
  }
}

function applyVelocities(): void {
  for (const c of clusters) {
    if (c.pinned) {
      c.vx = 0;
      c.vy = 0;
      continue;
    }
    c.vx *= 1 - VELOCITY_DECAY;
    c.vy *= 1 - VELOCITY_DECAY;
    c.x += c.vx;
    c.y += c.vy;
    // Boundary
    const r = c.radius;
    c.x = Math.max(r, Math.min(width - r, c.x));
    c.y = Math.max(r, Math.min(height - r, c.y));
  }
}

function tick(): void {
  if (!animating || clusters.length === 0) return;
  alpha *= 1 - ALPHA_DECAY;
  if (alpha < 0.001) alpha = 0.001;

  centerForce();
  for (let i = 0; i < clusters.length; i++) {
    for (let j = i + 1; j < clusters.length; j++) {
      repulsion(clusters[i], clusters[j]);
    }
  }
  applyVelocities();

  const positions: [number, number, number][] = clusters.map(
    (c) => [c.id, c.x, c.y]
  );
  self.postMessage({ type: "positions", positions });
  rafHandle = setTimeout(tick, 33); // ~30fps
}

self.onmessage = (event: MessageEvent) => {
  const msg = event.data;
  switch (msg.type) {
    case "init": {
      width = msg.width;
      height = msg.height;
      alpha = 1.0;
      clusters = (msg.clusters as Array<{
        id: number; mediaCount: number
      }>).map((c, i) => {
        const angle = (i / msg.clusters.length) * Math.PI * 2;
        const r = Math.min(width, height) * 0.35;
        return {
          id: c.id,
          x: width / 2 + r * Math.cos(angle) + (Math.random() - 0.5) * 20,
          y: height / 2 + r * Math.sin(angle) + (Math.random() - 0.5) * 20,
          vx: 0,
          vy: 0,
          radius: Math.max(12, Math.min(60, Math.sqrt(c.mediaCount) * 3)),
          pinned: false,
        };
      });
      if (rafHandle) clearTimeout(rafHandle);
      tick();
      break;
    }
    case "resize": {
      width = msg.width;
      height = msg.height;
      break;
    }
    case "tick_pause": {
      animating = false;
      if (rafHandle) { clearTimeout(rafHandle); rafHandle = null; }
      break;
    }
    case "tick_resume": {
      animating = true;
      alpha = 0.3; // Gentle re-warm
      tick();
      break;
    }
    case "pin": {
      const c = clusters.find((cl) => cl.id === msg.clusterId);
      if (c) { c.pinned = true; c.x = msg.x; c.y = msg.y; }
      break;
    }
    case "release": {
      const c = clusters.find((cl) => cl.id === msg.clusterId);
      if (c) { c.pinned = false; }
      break;
    }
  }
};
```

**Deliverable 2**: Rewrite `MandalaView.tsx` to use the worker and
animate cluster expansion. Replace the entire file:
```tsx
/**
 * MandalaView — Smriti's primary browse surface.
 *
 * Architecture:
 * - Force simulation runs in a Web Worker (mandala-force-worker.ts)
 * - Rendering uses Canvas 2D API (M1-safe, no WebGL compositor issues)
 * - Cluster positions streamed from worker at ~30fps
 * - Expansion animation uses requestAnimationFrame on main thread
 * - Semantic zoom: scroll changes visible detail level
 * - Keyboard navigation: Tab through clusters, Enter to expand
 *
 * Performance budget:
 * - Worker tick: ~2ms (off main thread)
 * - Canvas draw: ~4ms per frame
 * - Total main thread: < 6ms per frame (leaves 10ms budget at 60fps)
 */
import {
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
  type CSSProperties,
} from "react";

export type ClusterNode = {
  id: number;
  label: string;
  mediaCount: number;
  dominantEnergy: number;   // 0–1
  depthStratum: "foreground" | "midground" | "background" | "mixed";
  thumbnailUrls?: string[];
};

type MandalaViewProps = {
  clusters: ClusterNode[];
  selectedClusterId: number | null;
  onClusterSelect: (id: number) => void;
  onClusterExpand: (id: number) => void;
  className?: string;
  style?: CSSProperties;
};

// Energy → color ramp (matches design system)
function energyColor(energy: number, alpha = 1): string {
  const r = Math.round(200 + energy * 55);
  const g = Math.round(200 - energy * 150);
  const b = Math.round(255 - energy * 200);
  return `rgba(${r},${g},${b},${alpha})`;
}

function stratumColor(s: string): string {
  switch (s) {
    case "foreground": return "rgba(255,140,66,0.85)";
    case "background": return "rgba(67,216,201,0.85)";
    case "midground":  return "rgba(130,171,255,0.85)";
    default:           return "rgba(200,200,220,0.85)";
  }
}

export default function MandalaView({
  clusters,
  selectedClusterId,
  onClusterSelect,
  onClusterExpand,
  className,
  style,
}: MandalaViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const positionsRef = useRef<Map<number, { x: number; y: number }>>(new Map());
  const rafRef = useRef<number>(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredId, setHoveredId] = useState<number | null>(null);
  const [zoom, setZoom] = useState(1.0);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [expandProgress, setExpandProgress] = useState(0); // 0–1 spring
  const clusterMap = useMemo(
    () => new Map(clusters.map((c) => [c.id, c])),
    [clusters]
  );

  // ── Initialize/update worker ──────────────────────────────────────────
  useEffect(() => {
    if (clusters.length === 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const w = canvas.width;
    const h = canvas.height;

    if (!workerRef.current) {
      workerRef.current = new Worker(
        new URL("./mandala-force-worker.ts", import.meta.url),
        { type: "module" }
      );
      workerRef.current.onmessage = (e) => {
        if (e.data.type === "positions") {
          const map = new Map<number, { x: number; y: number }>();
          for (const [id, x, y] of e.data.positions as [number, number, number][]) {
            map.set(id, { x, y });
          }
          positionsRef.current = map;
        }
      };
    }

    workerRef.current.postMessage({
      type: "init",
      clusters: clusters.map((c) => ({ id: c.id, mediaCount: c.mediaCount })),
      width: w,
      height: h,
    });
  }, [clusters]);

  // ── Canvas render loop ────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      ctx.save();
      ctx.translate(panOffset.x, panOffset.y);
      ctx.scale(zoom, zoom);

      for (const cluster of clusters) {
        const pos = positionsRef.current.get(cluster.id);
        if (!pos) continue;

        const r = Math.max(10, Math.min(55, Math.sqrt(cluster.mediaCount) * 3));
        const isHovered = cluster.id === hoveredId;
        const isSelected = cluster.id === selectedClusterId;
        const isExpanded = cluster.id === expandedId;

        // Glow for selected/hovered
        if (isSelected || isHovered) {
          ctx.shadowColor = stratumColor(cluster.depthStratum);
          ctx.shadowBlur = isSelected ? 24 : 14;
        } else {
          ctx.shadowBlur = 0;
        }

        // Cluster circle
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, r * (isExpanded ? 1 + expandProgress * 0.3 : 1), 0, Math.PI * 2);
        const grad = ctx.createRadialGradient(pos.x - r * 0.3, pos.y - r * 0.3, 0, pos.x, pos.y, r);
        grad.addColorStop(0, energyColor(cluster.dominantEnergy, 0.8));
        grad.addColorStop(1, energyColor(cluster.dominantEnergy, 0.35));
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.strokeStyle = isSelected
          ? stratumColor(cluster.depthStratum)
          : "rgba(255,255,255,0.12)";
        ctx.lineWidth = isSelected ? 2 : 1;
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Label (only when zoomed in enough)
        if (zoom > 0.5 || isHovered) {
          ctx.fillStyle = "rgba(236,243,251,0.92)";
          ctx.font = `${Math.max(9, 11 / zoom)}px Inter, system-ui, sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          const label = cluster.label.length > 16
            ? cluster.label.slice(0, 14) + "…"
            : cluster.label;
          ctx.fillText(label, pos.x, pos.y + r + 14 / zoom);

          // Count badge
          if (isHovered || isSelected) {
            const countText = `${cluster.mediaCount}`;
            ctx.font = `${Math.max(8, 10 / zoom)}px Inter, system-ui, sans-serif`;
            ctx.fillStyle = "rgba(157,177,198,0.88)";
            ctx.fillText(countText, pos.x, pos.y + r + 26 / zoom);
          }
        }
      }

      ctx.restore();
      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [clusters, hoveredId, selectedClusterId, zoom, panOffset, expandedId, expandProgress]);

  // ── Expand animation ──────────────────────────────────────────────────
  useEffect(() => {
    if (expandedId === null) { setExpandProgress(0); return; }
    let frame: number;
    let progress = 0;
    const animate = () => {
      progress = Math.min(progress + 0.08, 1);
      setExpandProgress(progress);
      if (progress < 1) frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [expandedId]);

  // ── Mouse events ─────────────────────────────────────────────────────
  const hitTest = useCallback((clientX: number, clientY: number): number | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const cx = (clientX - rect.left - panOffset.x) / zoom;
    const cy = (clientY - rect.top - panOffset.y) / zoom;
    for (const [id, pos] of positionsRef.current) {
      const cluster = clusterMap.get(id);
      if (!cluster) continue;
      const r = Math.max(10, Math.min(55, Math.sqrt(cluster.mediaCount) * 3));
      const dx = cx - pos.x;
      const dy = cy - pos.y;
      if (dx * dx + dy * dy <= r * r) return id;
    }
    return null;
  }, [zoom, panOffset, clusterMap]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const id = hitTest(e.clientX, e.clientY);
    setHoveredId(id);
    if (canvasRef.current) {
      canvasRef.current.style.cursor = id ? "pointer" : "default";
    }
  }, [hitTest]);

  const handleClick = useCallback((e: React.MouseEvent) => {
    const id = hitTest(e.clientX, e.clientY);
    if (id === null) return;
    onClusterSelect(id);
    setExpandedId(id);
    setExpandProgress(0);
  }, [hitTest, onClusterSelect]);

  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    const id = hitTest(e.clientX, e.clientY);
    if (id !== null) onClusterExpand(id);
  }, [hitTest, onClusterExpand]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setZoom((z) => Math.max(0.2, Math.min(4.0, z - e.deltaY * 0.001)));
  }, []);

  // ── Resize canvas to container ────────────────────────────────────────
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;
    const obs = new ResizeObserver((entries) => {
      for (const entry of entries) {
        canvas.width = entry.contentRect.width * devicePixelRatio;
        canvas.height = entry.contentRect.height * devicePixelRatio;
        canvas.style.width = `${entry.contentRect.width}px`;
        canvas.style.height = `${entry.contentRect.height}px`;
        const ctx = canvas.getContext("2d");
        if (ctx) ctx.scale(devicePixelRatio, devicePixelRatio);
        workerRef.current?.postMessage({
          type: "resize",
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });
    obs.observe(container);
    return () => obs.disconnect();
  }, []);

  // ── Keyboard navigation ───────────────────────────────────────────────
  const [focusedIdx, setFocusedIdx] = useState(0);
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (clusters.length === 0) return;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") {
      const next = (focusedIdx + 1) % clusters.length;
      setFocusedIdx(next);
      onClusterSelect(clusters[next].id);
    } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
      const prev = (focusedIdx - 1 + clusters.length) % clusters.length;
      setFocusedIdx(prev);
      onClusterSelect(clusters[prev].id);
    } else if (e.key === "Enter" || e.key === " ") {
      if (clusters[focusedIdx]) {
        onClusterExpand(clusters[focusedIdx].id);
      }
    } else if (e.key === "+" || e.key === "=") {
      setZoom((z) => Math.min(z + 0.2, 4.0));
    } else if (e.key === "-") {
      setZoom((z) => Math.max(z - 0.2, 0.2));
    } else if (e.key === "0") {
      setZoom(1.0);
      setPanOffset({ x: 0, y: 0 });
    }
  }, [clusters, focusedIdx, onClusterSelect, onClusterExpand]);

  // ── Empty/loading state ───────────────────────────────────────────────
  if (clusters.length === 0) {
    return (
      <div
        ref={containerRef}
        className={`mandala-container ${className || ""}`}
        style={style}
        role="region"
        aria-label="Memory map — no clusters yet"
      >
        <div className="mandala-empty">
          <div style={{ textAlign: "center" }}>
            <p style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>◎</p>
            <p className="muted">No memory clusters yet.</p>
            <p className="muted" style={{ fontSize: "0.85rem" }}>
              Add a watch folder in Settings → Smriti Storage
              to start indexing your media.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={`mandala-container ${className || ""}`}
      style={style}
      role="application"
      aria-label={`Memory map with ${clusters.length} clusters. Use arrow keys to navigate, Enter to expand.`}
      tabIndex={0}
      onKeyDown={handleKeyDown}
    >
      <canvas
        ref={canvasRef}
        aria-hidden="true"
        onMouseMove={handleMouseMove}
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
        onWheel={handleWheel}
        style={{ display: "block", width: "100%", height: "100%" }}
      />

      {/* Zoom controls */}
      <div
        style={{
          position: "absolute", bottom: "1rem", left: "1rem",
          display: "flex", flexDirection: "column", gap: "0.35rem",
          zIndex: 5,
        }}
        aria-label="Map controls"
      >
        {[
          { label: "+", action: () => setZoom((z) => Math.min(z + 0.25, 4.0)), aria: "Zoom in" },
          { label: "⊙", action: () => { setZoom(1); setPanOffset({ x: 0, y: 0 }); }, aria: "Reset view" },
          { label: "−", action: () => setZoom((z) => Math.max(z - 0.25, 0.2)), aria: "Zoom out" },
        ].map(({ label, action, aria }) => (
          <button
            key={label}
            onClick={action}
            aria-label={aria}
            style={{
              width: 36, height: 36, borderRadius: "50%",
              display: "flex", alignItems: "center", justifyContent: "center",
              background: "rgba(14,22,36,0.82)", border: "1px solid rgba(255,255,255,0.12)",
              fontSize: "1rem", cursor: "pointer",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Zoom indicator */}
      <div
        style={{
          position: "absolute", bottom: "1rem", right: "1rem",
          fontSize: "0.72rem", color: "var(--text-muted)",
          background: "rgba(14,22,36,0.7)",
          padding: "0.25rem 0.5rem", borderRadius: 8,
        }}
        aria-live="polite"
        aria-label={`Zoom level ${Math.round(zoom * 100)}%`}
      >
        {Math.round(zoom * 100)}%
      </div>

      {/* Screen-reader-only cluster list */}
      <ul
        style={{
          position: "absolute", width: 1, height: 1,
          overflow: "hidden", clip: "rect(0,0,0,0)",
          margin: 0, padding: 0, border: 0,
        }}
        aria-label="Memory clusters list"
      >
        {clusters.map((c, i) => (
          <li key={c.id}>
            <button
              onClick={() => { onClusterSelect(c.id); setFocusedIdx(i); }}
              onDoubleClick={() => onClusterExpand(c.id)}
              aria-pressed={selectedClusterId === c.id}
            >
              {c.label} — {c.mediaCount} memories
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

---

### AGENT: deepdive_enhancement

**Reads**: `desktop/electron/src/components/smriti/DeepdiveView.tsx`,
`desktop/electron/src/hooks/useSmritiState.ts`,
`desktop/electron/src/types.ts`,
`cloud/runtime/app.py`

**Objective**: Enhance DeepdiveView with:
1. Clickable JEPA energy patches — click any patch to see its AnchorMatch
2. Entity thread panel wired to real EpistemicAtlas data
3. Semantic neighbors wired to real Setu-2 recall data
4. Keyboard navigation (←/→ temporal, ↑/↓ semantic, E toggle energy,
   Escape closes, F fullscreen)

**Success gate**:
```bash
cd desktop/electron && npm run typecheck && npm run build
```

**Deliverable 1**: Add to `cloud/runtime/app.py`:
```python
    @app.get("/v1/smriti/media/{media_id}/neighbors",
             dependencies=[Depends(require_auth)])
    async def smriti_neighbors(
        media_id: str,
        top_k: int = Query(default=6, ge=1, le=20)
    ) -> dict:
        """Get semantic neighbors for a media item using Setu-2 EBM."""
        runtime = app.state.runtime
        if not hasattr(runtime, "smriti_db") or runtime.smriti_db is None:
            return {"neighbors": []}
        try:
            media = runtime.smriti_db.get_smriti_media(media_id)
            if media is None:
                return {"neighbors": []}
            embedding = media.embedding if hasattr(media, "embedding") else None
            if embedding is None:
                return {"neighbors": []}
            import numpy as np
            emb = np.array(embedding, dtype=np.float32)
            # Use Setu-2 for energy-ranked retrieval
            if hasattr(runtime, "_setu2_bridge"):
                corpus = runtime.smriti_db.get_all_embeddings(limit=500)
                if corpus["embeddings"].shape[0] > 1:
                    energies = runtime._setu2_bridge.project_query(
                        emb, corpus["embeddings"]
                    )
                    top_indices = np.argsort(energies)[:top_k + 1]
                    neighbors = []
                    for idx in top_indices:
                        mid = corpus["media_ids"][idx]
                        if mid != media_id:
                            neighbors.append({
                                "media_id": mid,
                                "setu_score": float(energies[idx]),
                                "thumbnail_path": corpus.get("thumbnails", {}).get(mid, ""),
                            })
                    return {"neighbors": neighbors[:top_k]}
        except Exception:
            pass
        return {"neighbors": []}
```

**Deliverable 2**: Rewrite `DeepdiveView.tsx` with full interactivity.

The key additions to the existing DeepdiveView structure:
- Patch grid overlay (14×14 transparent buttons over the image)
- Each patch button shows a popover with its AnchorMatch when clicked
- Entity threads panel fetches from `/v1/world-state?session_id=...`
- Semantic neighbors fetches from `/v1/smriti/media/{id}/neighbors`
- Keyboard event listeners attached to the modal container
```tsx
// Add these additions to the existing DeepdiveView.tsx:

// 1. State additions:
const [selectedPatch, setSelectedPatch] = useState<number | null>(null);
const [patchDetail, setPatchDetail] = useState<{
  anchor: string;
  confidence: number;
  stratum: string;
  description: string;
  hallucinationRisk: number;
} | null>(null);
const [neighbors, setNeighbors] = useState<Array<{
  media_id: string; setu_score: number; thumbnail_path: string
}>>([]);
const [energyVisible, setEnergyVisible] = useState(false);
const modalRef = useRef<HTMLDivElement>(null);

// 2. Load neighbors on open:
useEffect(() => {
  if (!result?.media_id) return;
  runtimeRequest<{ neighbors: typeof neighbors }>(
    `/v1/smriti/media/${result.media_id}/neighbors?top_k=6`
  ).then((r) => setNeighbors(r.neighbors)).catch(() => {});
}, [result?.media_id]);

// 3. Keyboard handler:
useEffect(() => {
  const handleKey = (e: KeyboardEvent) => {
    if (e.key === "Escape") { onClose(); return; }
    if (e.key === "e" || e.key === "E") setEnergyVisible((v) => !v);
    if (e.key === "f" || e.key === "F") {
      if (document.fullscreenElement) document.exitFullscreen();
      else modalRef.current?.requestFullscreen();
    }
  };
  window.addEventListener("keydown", handleKey);
  return () => window.removeEventListener("keydown", handleKey);
}, [onClose]);

// 4. Patch click handler:
function handlePatchClick(patchIdx: number) {
  setSelectedPatch(patchIdx === selectedPatch ? null : patchIdx);
  if (!result?.anchor_matches) { setPatchDetail(null); return; }
  const match = (result.anchor_matches as Array<{
    patch_indices: number[];
    template_name: string;
    confidence: number;
    depth_stratum: string;
  }>).find((m) => m.patch_indices?.includes(patchIdx));
  if (match) {
    const desc = (result.setu_descriptions as Array<{
      gate: { anchor_name: string };
      description: { text: string; hallucination_risk: number };
    }>)?.find((d) => d.gate?.anchor_name === match.template_name);
    setPatchDetail({
      anchor: match.template_name,
      confidence: match.confidence,
      stratum: match.depth_stratum,
      description: desc?.description?.text || match.template_name,
      hallucinationRisk: desc?.description?.hallucination_risk ?? 0.5,
    });
  } else {
    setPatchDetail({ anchor: "unknown", confidence: 0, stratum: "unknown",
                     description: "No anchor match for this patch",
                     hallucinationRisk: 1.0 });
  }
}

// 5. In the JSX, add the patch grid overlay over the image:
// (14×14 = 196 transparent patch buttons)
{energyVisible && (
  <div
    aria-label="JEPA patch selection grid"
    style={{
      position: "absolute", inset: 0,
      display: "grid",
      gridTemplateColumns: "repeat(14, 1fr)",
      gridTemplateRows: "repeat(14, 1fr)",
      pointerEvents: "all",
    }}
  >
    {Array.from({ length: 196 }, (_, i) => {
      const depthMap = result?.depth_strata;
      const row = Math.floor(i / 14);
      const col = i % 14;
      const isFg = depthMap?.foreground_mask?.[row]?.[col];
      const isBg = depthMap?.background_mask?.[row]?.[col];
      const energy = (result?.energy_map as number[][] | undefined)?.[row]?.[col] ?? 0;
      const isSelected = selectedPatch === i;
      return (
        <button
          key={i}
          onClick={() => handlePatchClick(i)}
          aria-label={`Patch ${i} — ${isFg ? "foreground" : isBg ? "background" : "midground"}`}
          style={{
            background: isSelected
              ? "rgba(255,255,255,0.3)"
              : isFg ? `rgba(255,140,66,${energy * 0.35})`
              : isBg ? `rgba(67,216,201,${energy * 0.35})`
              : `rgba(130,171,255,${energy * 0.25})`,
            border: isSelected ? "2px solid white" : "1px solid rgba(255,255,255,0.06)",
            cursor: "pointer",
            transition: "background 120ms ease",
          }}
        />
      );
    })}
  </div>
)}

// 6. Patch detail popover:
{selectedPatch !== null && patchDetail && energyVisible && (
  <div
    style={{
      position: "absolute",
      top: "1rem", right: "1rem",
      background: "rgba(14,22,36,0.94)",
      border: "1px solid rgba(255,255,255,0.12)",
      backdropFilter: "blur(20px)",
      borderRadius: 16,
      padding: "0.85rem 1rem",
      width: 240,
      zIndex: 10,
    }}
    role="tooltip"
    aria-live="polite"
  >
    <p className="eyebrow" style={{ marginBottom: "0.3rem" }}>
      Patch {selectedPatch}
    </p>
    <strong>{patchDetail.anchor}</strong>
    <p className="muted" style={{ fontSize: "0.82rem", margin: "0.25rem 0" }}>
      {patchDetail.description}
    </p>
    <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap", marginTop: "0.5rem" }}>
      <span className="chips chips--stable" style={{ fontSize: "0.75rem" }}>
        {patchDetail.stratum}
      </span>
      <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
        conf {(patchDetail.confidence * 100).toFixed(0)}%
      </span>
      <span style={{
        fontSize: "0.75rem",
        color: patchDetail.hallucinationRisk < 0.3
          ? "var(--kpi-healthy)"
          : patchDetail.hallucinationRisk < 0.6
          ? "var(--kpi-watch)"
          : "var(--kpi-danger)",
      }}>
        risk {(patchDetail.hallucinationRisk * 100).toFixed(0)}%
      </span>
    </div>
  </div>
)}

// 7. Semantic neighbors panel:
{neighbors.length > 0 && (
  <div style={{ marginTop: "1rem" }}>
    <p className="eyebrow" style={{ marginBottom: "0.5rem" }}>
      Semantic Neighbors
    </p>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.5rem" }}>
      {neighbors.slice(0, 6).map((n) => (
        <button
          key={n.media_id}
          onClick={() => {/* open neighbor in deepdive */}}
          style={{
            background: "rgba(255,255,255,0.04)",
            border: "1px solid var(--line)",
            borderRadius: 12, padding: "0.4rem",
            cursor: "pointer", aspectRatio: "1",
            overflow: "hidden",
          }}
          aria-label={`Similar memory — Setu score ${n.setu_score.toFixed(2)}`}
          title={`Setu score: ${n.setu_score.toFixed(3)}`}
        >
          <div style={{
            width: "100%", height: "100%",
            background: "rgba(67,216,201,0.1)",
            borderRadius: 8,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "0.7rem", color: "var(--text-muted)",
          }}>
            {n.setu_score.toFixed(2)}
          </div>
        </button>
      ))}
    </div>
  </div>
)}

// 8. Keyboard shortcut legend:
<div style={{
  display: "flex", gap: "1rem", flexWrap: "wrap",
  marginTop: "0.5rem", fontSize: "0.72rem", color: "var(--text-muted)",
}}>
  {[
    { key: "E", action: "Toggle energy" },
    { key: "F", action: "Fullscreen" },
    { key: "Esc", action: "Close" },
  ].map(({ key, action }) => (
    <span key={key}>
      <kbd style={{
        border: "1px solid rgba(255,255,255,0.15)",
        borderRadius: 4, padding: "0.1rem 0.3rem",
        fontFamily: "monospace", fontSize: "0.7rem",
        marginRight: "0.3rem",
      }}>
        {key}
      </kbd>
      {action}
    </span>
  ))}
</div>
```

---

### AGENT: person_journal_enhancement

**Reads**: `desktop/electron/src/components/smriti/PersonJournal.tsx`,
`desktop/electron/src/hooks/useSmritiState.ts`,
`desktop/electron/src/types.ts`

**Objective**: Enhance PersonJournal with a Canvas 2D co-occurrence graph
showing people who frequently appear with the target person,
connected by weighted edges from EpistemicAtlas data.

**Success gate**:
```bash
cd desktop/electron && npm run typecheck && npm run build
```

**Add co-occurrence graph to PersonJournal.tsx**:
```tsx
// Add Co-occurrence graph component inside PersonJournal.tsx:

type CoNode = { id: string; label: string; count: number; x?: number; y?: number };
type CoEdge = { source: string; target: string; weight: number };

function CoOccurrenceGraph({
  personName,
  journal,
}: {
  personName: string;
  journal: PersonJournalData | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Extract nodes and edges from atlas data in journal
  const { nodes, edges } = useMemo(() => {
    if (!journal?.atlas) return { nodes: [] as CoNode[], edges: [] as CoEdge[] };
    const atlasNodes = (journal.atlas as any)?.nodes || [];
    const atlasEdges = (journal.atlas as any)?.edges || [];
    const nodes: CoNode[] = atlasNodes.map((n: any) => ({
      id: String(n.entity_id || n.id),
      label: String(n.label || n.entity_id || "?"),
      count: Number(n.track_length || 1),
    }));
    const edges: CoEdge[] = atlasEdges.map((e: any) => ({
      source: String(e.source_id),
      target: String(e.target_id),
      weight: Number(e.spatial_proximity || 0.5),
    }));
    return { nodes, edges };
  }, [journal?.atlas]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || nodes.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width = canvas.offsetWidth * devicePixelRatio;
    const H = canvas.height = canvas.offsetHeight * devicePixelRatio;
    canvas.style.width = `${canvas.offsetWidth}px`;
    canvas.style.height = `${canvas.offsetHeight}px`;
    ctx.scale(devicePixelRatio, devicePixelRatio);

    const cW = canvas.offsetWidth;
    const cH = canvas.offsetHeight;
    const cx = cW / 2;
    const cy = cH / 2;

    // Simple circular layout
    const positioned = nodes.map((n, i) => {
      const angle = (i / nodes.length) * Math.PI * 2 - Math.PI / 2;
      const r = Math.min(cW, cH) * 0.35;
      return {
        ...n,
        x: n.label === personName ? cx : cx + r * Math.cos(angle),
        y: n.label === personName ? cy : cy + r * Math.sin(angle),
      };
    });

    const nodeMap = new Map(positioned.map((n) => [n.id, n]));

    ctx.clearRect(0, 0, cW, cH);

    // Draw edges
    for (const edge of edges) {
      const a = nodeMap.get(edge.source);
      const b = nodeMap.get(edge.target);
      if (!a || !b) continue;
      ctx.beginPath();
      ctx.moveTo(a.x!, a.y!);
      ctx.lineTo(b.x!, b.y!);
      ctx.strokeStyle = `rgba(157,177,198,${0.15 + edge.weight * 0.35})`;
      ctx.lineWidth = Math.max(1, edge.weight * 3);
      ctx.stroke();
    }

    // Draw nodes
    for (const node of positioned) {
      const r = Math.max(12, Math.min(30, Math.sqrt(node.count) * 4));
      const isPrimary = node.label === personName;

      ctx.beginPath();
      ctx.arc(node.x!, node.y!, r, 0, Math.PI * 2);
      ctx.fillStyle = isPrimary
        ? "rgba(255,140,66,0.8)"
        : "rgba(67,216,201,0.6)";
      ctx.fill();
      ctx.strokeStyle = isPrimary
        ? "rgba(255,200,100,0.9)"
        : "rgba(255,255,255,0.15)";
      ctx.lineWidth = isPrimary ? 2 : 1;
      ctx.stroke();

      ctx.fillStyle = "rgba(236,243,251,0.92)";
      ctx.font = `${isPrimary ? 11 : 9}px Inter, system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const label = node.label.length > 12
        ? node.label.slice(0, 10) + "…"
        : node.label;
      ctx.fillText(label, node.x!, node.y! + r + 11);
    }
  }, [nodes, edges, personName]);

  if (nodes.length === 0) return null;

  return (
    <div style={{ marginTop: "1rem" }}>
      <p className="eyebrow" style={{ marginBottom: "0.5rem" }}>
        Co-occurrence Graph
      </p>
      <div
        style={{
          border: "1px solid var(--line)",
          borderRadius: 18,
          overflow: "hidden",
          height: 220,
          background: "rgba(255,255,255,0.02)",
          position: "relative",
        }}
        role="img"
        aria-label={`Co-occurrence graph for ${personName} showing ${nodes.length} connected entities`}
      >
        <canvas
          ref={canvasRef}
          style={{ width: "100%", height: "100%", display: "block" }}
          aria-hidden="true"
        />
      </div>
    </div>
  );
}

// Add CoOccurrenceGraph inside PersonJournal render:
// After the existing timeline/appearance data, before the end:
<CoOccurrenceGraph personName={personName} journal={journal} />
```

---

### AGENT: accessibility_pass

**Reads**: All `desktop/electron/src/components/smriti/*.tsx`,
`desktop/electron/src/styles.css`

**Objective**: Ensure WCAG 2.1 AA compliance across all Smriti components.

**Specific requirements**:

1. **Focus management**: When DeepdiveView opens, focus moves to the modal.
   When it closes, focus returns to the triggering element.

2. **ARIA roles**: All interactive elements have correct ARIA labels.
   Canvas elements are `aria-hidden`. Screen-reader alternatives provided.

3. **Color contrast**: All text meets 4.5:1 contrast ratio.
   `var(--text-muted)` on `var(--bg)` must be verified and adjusted if needed.

4. **Keyboard traps**: Modal (Deepdive) must trap focus within it.
   Escape key closes it.

5. **Reduced motion**: All CSS animations conditional on
   `prefers-reduced-motion: no-preference`.

6. **Screen reader announcements**: Recall results and ingestion progress
   announced via `aria-live="polite"` regions.

**Deliverables**:

**Add to `desktop/electron/src/styles.css`** (append, do not replace):
```css
/* ─── Accessibility — WCAG 2.1 AA ──────────────────────────────────────── */

/* Focus ring — visible in all contexts */
:focus-visible {
  outline: 3px solid var(--accent-2);
  outline-offset: 3px;
  border-radius: 4px;
}

/* Ensure muted text meets 4.5:1 on dark background */
[data-theme="dark"] .muted,
:root .muted {
  color: #9db1c6; /* Verified 4.6:1 on #07111b */
}

[data-theme="light"] .muted {
  color: #4a5e70; /* Verified 4.7:1 on #eef4f8 */
}

/* Skip link for keyboard users */
.skip-link {
  position: absolute;
  top: -100%;
  left: 0;
  z-index: 9999;
  background: var(--accent-2);
  color: #000;
  padding: 0.5rem 1rem;
  font-weight: 600;
  text-decoration: none;
  border-radius: 0 0 8px 0;
}

.skip-link:focus {
  top: 0;
}

/* Reduced motion — disable all CSS animations */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}

/* High contrast adjustments */
@media (prefers-contrast: more) {
  .hallucination-badge.verified {
    border-width: 2px;
    font-weight: 700;
  }

  .recall-card {
    border-color: rgba(255, 255, 255, 0.3);
  }

  .deepdive-overlay {
    background: rgba(0, 0, 0, 0.98);
  }
}

/* Live region for announcements */
.sr-live-region {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
```

**Add focus trap to DeepdiveView** — when modal opens, call:
```tsx
// In DeepdiveView, add focus management:
useEffect(() => {
  if (!isOpen) return;
  const modal = modalRef.current;
  if (!modal) return;

  // Save currently focused element
  const previouslyFocused = document.activeElement as HTMLElement;

  // Focus the modal
  modal.focus();

  // Focus trap
  const focusableSelectors = [
    'button', '[href]', 'input', 'select', 'textarea',
    '[tabindex]:not([tabindex="-1"])'
  ].join(', ');

  const focusable = Array.from(
    modal.querySelectorAll<HTMLElement>(focusableSelectors)
  ).filter((el) => !el.hasAttribute('disabled'));

  const first = focusable[0];
  const last = focusable[focusable.length - 1];

  const handleTab = (e: KeyboardEvent) => {
    if (e.key !== 'Tab') return;
    if (e.shiftKey) {
      if (document.activeElement === first) {
        e.preventDefault();
        last?.focus();
      }
    } else {
      if (document.activeElement === last) {
        e.preventDefault();
        first?.focus();
      }
    }
  };

  modal.addEventListener('keydown', handleTab);

  return () => {
    modal.removeEventListener('keydown', handleTab);
    previouslyFocused?.focus?.(); // Restore focus on close
  };
}, [isOpen]);
```

**Add live region to SmritiTab.tsx**:
```tsx
// Add at top of SmritiTab render:
<div className="sr-live-region" aria-live="polite" aria-atomic="true">
  {smriti.status}
</div>
```

---

### AGENT: smriti_styles_sprint5

**Context files**: `desktop/electron/src/styles.css`

**Objective**: Append Sprint 5 CSS. Do NOT modify existing CSS.
Append to end of file only.

**APPEND to `desktop/electron/src/styles.css`**:
```css
/* ─── Sprint 5 additions ──────────────────────────────────────────────── */

/* Migration panel */
.migration-result {
  border-radius: 16px;
  padding: 0.85rem 1rem;
  margin-top: 0.75rem;
}

.migration-result.success {
  border: 1px solid rgba(67, 216, 201, 0.3);
  background: rgba(67, 216, 201, 0.05);
}

.migration-result.failed {
  border: 1px solid rgba(230, 57, 70, 0.3);
  background: rgba(230, 57, 70, 0.05);
}

/* Feedback buttons on recall cards */
.recall-feedback-row {
  display: flex;
  gap: 0.35rem;
  margin-top: 0.4rem;
}

.recall-feedback-btn {
  padding: 0.2rem 0.5rem;
  font-size: 0.75rem;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: transparent;
  cursor: pointer;
  transition: background 120ms ease, color 120ms ease;
}

.recall-feedback-btn:hover {
  background: rgba(255, 255, 255, 0.07);
}

.recall-feedback-btn.confirmed {
  color: var(--kpi-healthy);
  border-color: rgba(67, 216, 201, 0.35);
}

.recall-feedback-btn.rejected {
  color: var(--text-muted);
  border-color: rgba(255, 255, 255, 0.1);
}

/* Deepdive patch grid overlay */
.deepdive-patch-grid {
  position: absolute;
  inset: 0;
  display: grid;
  grid-template-columns: repeat(14, 1fr);
  grid-template-rows: repeat(14, 1fr);
  pointer-events: all;
}

.deepdive-patch-cell {
  border: 1px solid rgba(255, 255, 255, 0.05);
  cursor: crosshair;
  transition: background 80ms ease;
}

.deepdive-patch-cell:hover {
  background: rgba(255, 255, 255, 0.18) !important;
}

.deepdive-patch-cell[aria-pressed="true"] {
  border: 2px solid rgba(255, 255, 255, 0.85);
}

/* Patch detail popover */
.patch-detail-popover {
  position: absolute;
  top: 1rem;
  right: 1rem;
  width: 240px;
  background: rgba(14, 22, 36, 0.94);
  border: 1px solid rgba(255, 255, 255, 0.12);
  backdrop-filter: blur(20px);
  border-radius: 16px;
  padding: 0.85rem 1rem;
  z-index: 10;
  animation: popover-in 120ms ease;
}

@keyframes popover-in {
  from { opacity: 0; transform: translateY(-6px); }
  to   { opacity: 1; transform: translateY(0); }
}

@media (prefers-reduced-motion: reduce) {
  .patch-detail-popover {
    animation: none;
  }
}

/* Co-occurrence graph container */
.co-occurrence-graph {
  border: 1px solid var(--line);
  border-radius: 18px;
  overflow: hidden;
  height: 220px;
  background: rgba(255, 255, 255, 0.02);
  position: relative;
}

/* Semantic neighbors grid */
.semantic-neighbors-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.5rem;
  margin-top: 0.5rem;
}

.neighbor-thumb {
  aspect-ratio: 1;
  border: 1px solid var(--line);
  border-radius: 12px;
  overflow: hidden;
  cursor: pointer;
  background: rgba(67, 216, 201, 0.06);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.7rem;
  color: var(--text-muted);
  transition: border-color 140ms ease, background 140ms ease;
}

.neighbor-thumb:hover {
  border-color: rgba(67, 216, 201, 0.4);
  background: rgba(67, 216, 201, 0.1);
}

/* Keyboard shortcut legend */
.kbd-legend {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  font-size: 0.72rem;
  color: var(--text-muted);
  margin-top: 0.5rem;
}

kbd {
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 4px;
  padding: 0.1rem 0.35rem;
  font-family: var(--smriti-font-mono, "JetBrains Mono", "Fira Code", monospace);
  font-size: 0.7rem;
  margin-right: 0.3rem;
  background: rgba(255, 255, 255, 0.04);
}

/* Migration panel */
.migration-panel {
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
}

.migration-result {
  border-radius: 16px;
  padding: 0.85rem 1rem;
  margin-top: 0.25rem;
  animation: popover-in 140ms ease;
}

.migration-result.success {
  border: 1px solid rgba(67, 216, 201, 0.3);
  background: rgba(67, 216, 201, 0.05);
}

.migration-result.failed {
  border: 1px solid rgba(230, 57, 70, 0.3);
  background: rgba(230, 57, 70, 0.05);
}

/* Mandala zoom controls */
.mandala-controls {
  position: absolute;
  bottom: 1rem;
  left: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  z-index: 5;
}

.mandala-ctrl-btn {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(14, 22, 36, 0.82);
  border: 1px solid rgba(255, 255, 255, 0.12);
  font-size: 1rem;
  cursor: pointer;
  transition: background 120ms ease, border-color 120ms ease;
}

.mandala-ctrl-btn:hover {
  background: rgba(67, 216, 201, 0.12);
  border-color: rgba(67, 216, 201, 0.35);
}

.mandala-zoom-label {
  position: absolute;
  bottom: 1rem;
  right: 1rem;
  font-size: 0.72rem;
  color: var(--text-muted);
  background: rgba(14, 22, 36, 0.7);
  padding: 0.25rem 0.5rem;
  border-radius: 8px;
  pointer-events: none;
  user-select: none;
}

/* Person journal timeline */
.journal-timeline {
  display: flex;
  flex-direction: column;
  gap: 0;
  position: relative;
  padding-left: 1.5rem;
}

.journal-timeline::before {
  content: "";
  position: absolute;
  left: 0.55rem;
  top: 0.5rem;
  bottom: 0.5rem;
  width: 2px;
  background: linear-gradient(
    to bottom,
    rgba(67, 216, 201, 0.6),
    rgba(67, 216, 201, 0.08)
  );
  border-radius: 2px;
}

.journal-event {
  position: relative;
  padding: 0.55rem 0;
}

.journal-event::before {
  content: "";
  position: absolute;
  left: -1.05rem;
  top: 1.05rem;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: rgba(67, 216, 201, 0.8);
  border: 2px solid rgba(14, 22, 36, 0.95);
}

/* Sprint 5 responsive overrides */
@media (max-width: 900px) {
  .semantic-neighbors-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .patch-detail-popover {
    top: auto;
    bottom: 4rem;
    right: 0.5rem;
    width: 200px;
  }
}
/* ── End Sprint 5 ─────────────────────────────────────────────────────── */
AGENT: sprint5_validation
Reads: ALL modified files. State after all 8 prior agents complete.
Objective: Execute the complete validation suite. Every gate must
pass. Update state.json with honest measured counts.
Fail loudly if any gate does not pass — never fabricate results.
Success gates (ALL must pass in sequence):
# ══ GATE 1: Full backend test suite ══════════════════════════════════════
pytest -q \
  cloud/api/tests \
  cloud/jepa_service/tests \
  cloud/search_service/tests \
  cloud/monitoring/tests \
  tests/test_readme.py
# Required: ≥ 195 passed, 0 failed
# If < 195: list which agents produced fewer tests than specified.
# Do NOT mark build_complete = true unless this passes.

# ══ GATE 2: Production gate — all 11 tests ════════════════════════════════
pytest -v cloud/api/tests/test_smriti_production.py
# ALL 11 tests must pass. Any failure = build rejected.

# ══ GATE 3: Storage config — all 16 tests ════════════════════════════════
pytest -v cloud/api/tests/test_smriti_storage_config.py
# All 16 must pass.

# ══ GATE 4: Data migration — all 10 tests ════════════════════════════════
pytest -v cloud/api/tests/test_smriti_data_migration.py
# All 10 must pass.

# ══ GATE 5: Setu-2 feedback — all 7 tests ════════════════════════════════
pytest -v cloud/api/tests/test_setu2_feedback.py
# All 7 must pass.

# ══ GATE 6: Telescope contract — primary contract ════════════════════════
pytest -v \
  cloud/api/tests/test_smriti_production.py::test_telescope_behind_person_not_described_as_body_part
# MUST pass. This gate cannot be waived.

# ══ GATE 7: TypeScript compile ═══════════════════════════════════════════
cd desktop/electron && npm run typecheck
# Must exit 0 with no errors.

# ══ GATE 8: Frontend production build ════════════════════════════════════
cd desktop/electron && npm run build
# Must exit 0. dist/ directory must exist.

# ══ GATE 9: Deprecation clean ════════════════════════════════════════════
python3.11 -W error::DeprecationWarning -c "
from cloud.runtime.app import create_app
import tempfile, pathlib
with tempfile.TemporaryDirectory() as d:
    app = create_app(data_dir=d)
print('CLEAN — 0 DeprecationWarnings')
"
# Must print CLEAN.

# ══ GATE 10: Force worker file exists ════════════════════════════════════
test -f desktop/electron/src/components/smriti/mandala-force-worker.ts \
  && echo "WORKER EXISTS" || echo "MISSING: mandala-force-worker.ts"
# Must print WORKER EXISTS.

# ══ GATE 11: Migration endpoint registered ═══════════════════════════════
python3.11 -c "
import tempfile
from cloud.runtime.app import create_app
with tempfile.TemporaryDirectory() as d:
    app = create_app(data_dir=d)
    routes = {r.path for r in app.routes}
    required = [
        '/v1/smriti/storage/migrate',
        '/v1/smriti/recall/feedback',
        '/v1/smriti/media/{media_id}/neighbors',
    ]
    missing = [r for r in required if r not in routes]
    if missing:
        print('MISSING ROUTES:', missing)
        exit(1)
    print('ALL SPRINT 5 ROUTES PRESENT')
"

# ══ GATE 12: Torch isolation ═════════════════════════════════════════════
python3.11 -c "
import subprocess, sys
modules = [
    'cloud.jepa_service.engine',
    'cloud.jepa_service.depth_separator',
    'cloud.jepa_service.anchor_graph',
    'cloud.runtime.setu2',
    'cloud.runtime.smriti_storage',
    'cloud.runtime.smriti_ingestion',
    'cloud.runtime.smriti_migration',
]
violations = []
for mod in modules:
    r = subprocess.run(
        [sys.executable, '-c',
         f'import {mod}; import sys; '
         f'assert \"torch\" not in sys.modules, f\"VIOLATION: torch in {mod}\"'],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        violations.append(mod)
if violations:
    print('TORCH ISOLATION VIOLATIONS:', violations)
    exit(1)
print('TORCH ISOLATION CLEAN')
"

# ══ GATE 13: Accessibility CSS present ═══════════════════════════════════
grep -c "prefers-reduced-motion" desktop/electron/src/styles.css
# Must return ≥ 1

grep -c "skip-link" desktop/electron/src/styles.css
# Must return ≥ 1

grep -c "sr-live-region" desktop/electron/src/styles.css
# Must return ≥ 1

# ══ GATE 14: Smriti routes all present (complete set) ═════════════════════
python3.11 -c "
import tempfile
from cloud.runtime.app import create_app
with tempfile.TemporaryDirectory() as d:
    app = create_app(data_dir=d)
    routes = {r.path for r in app.routes}
    sprint_4_routes = [
        '/v1/smriti/ingest',
        '/v1/smriti/recall',
        '/v1/smriti/status',
        '/v1/smriti/clusters',
        '/v1/smriti/metrics',
        '/v1/smriti/storage',
        '/v1/smriti/storage/usage',
        '/v1/smriti/watch-folders',
        '/v1/smriti/storage/prune',
        '/v1/smriti/tag/person',
    ]
    sprint_5_routes = [
        '/v1/smriti/storage/migrate',
        '/v1/smriti/recall/feedback',
        '/v1/smriti/media/{media_id}/neighbors',
    ]
    all_required = sprint_4_routes + sprint_5_routes
    missing = [r for r in all_required if r not in routes]
    if missing:
        print('MISSING ROUTES:', missing)
        exit(1)
    print(f'ALL {len(all_required)} SMRITI ROUTES REGISTERED')
"
On ALL gates passing, update .smriti_build/state.json:

python3.11 -c "
import json, pathlib, subprocess, sys

# Get actual test count
result = subprocess.run(
    [sys.executable, '-m', 'pytest', '-q',
     'cloud/api/tests',
     'cloud/jepa_service/tests',
     'cloud/search_service/tests',
     'cloud/monitoring/tests',
     'tests/test_readme.py',
     '--tb=no'],
    capture_output=True, text=True
)
# Parse last line: '195 passed in X.Xs'
last = [l for l in result.stdout.strip().split('\n') if 'passed' in l]
count = int(last[-1].split()[0]) if last else -1

state = {
    'version': '5.0',
    'sprint': '5',
    'build_complete': True,
    'baseline_tests_passing': 162,
    'total_tests_passing': count,  # HONEST measured count
    'target_tests_passing': 195,
    'met_target': count >= 195,
    'telescope_test': 'PASSED',
    'deprecation_warnings': 0,
    'torch_isolation': 'CLEAN',
    'accessibility': 'WCAG_2_1_AA',
    'agents': {
        'production_gate': 'complete',
        'data_migration': 'complete',
        'setu2_feedback': 'complete',
        'mandala_enhancement': 'complete',
        'deepdive_enhancement': 'complete',
        'person_journal_enhancement': 'complete',
        'accessibility_pass': 'complete',
        'smriti_styles_sprint5': 'complete',
        'sprint5_validation': 'complete',
    },
    'new_routes': [
        '/v1/smriti/storage/migrate',
        '/v1/smriti/recall/feedback',
        '/v1/smriti/media/{media_id}/neighbors',
    ],
    'new_files': [
        'cloud/api/tests/test_smriti_production.py',
        'cloud/runtime/smriti_migration.py',
        'cloud/api/tests/test_smriti_data_migration.py',
        'cloud/api/tests/test_setu2_feedback.py',
        'desktop/electron/src/components/smriti/mandala-force-worker.ts',
    ],
    'blocking_issues': [],
    'm1_verified': True,
}
pathlib.Path('.smriti_build/state.json').write_text(json.dumps(state, indent=2))
print(f'Sprint 5 state written. Tests: {count}. Target met: {count >= 195}')
"
```

**If count < 195**, write the state with `"build_complete": false` and
`"met_target": false`, and print which agents delivered fewer tests than
their specification required. NEVER write `"build_complete": true` unless
count ≥ 195 and all 14 gates pass.

---

## M1-SPECIFIC CONSTRAINTS (applies to ALL agents)

These are non-negotiable hardware constraints for M1 iMac 8GB:
```
1. NO CUDA anywhere. All numpy paths are float32. MPS only when
   TOORI_DINOV2_DEVICE=mps is explicitly set in environment.

2. 8GB unified memory ceiling:
   - JEPAWorkerPool: num_workers=1 default. Hard cap at 2.
   - Any new worker spawned in tests must have a memory budget comment.
   - smriti_migration runs in run_in_executor — not in JEPAWorkerPool.

3. PyAV on Apple Silicon:
   pip install av  (pre-built arm64 wheel — never compile from source)

4. Canvas 2D only in MandalaView and PersonJournal:
   - NO WebGL context anywhere in Smriti UI
   - NO THREE.js import in any Smriti component
   - canvas.getContext("2d") is the only allowed 2D context

5. Web Worker for mandala-force-worker.ts:
   - The worker must import ONLY built-in types — no npm imports
   - No SharedArrayBuffer (not available in all Electron contexts)
   - Communication via postMessage only

6. No localStorage in any frontend component.
   Use React state + API calls only.

7. smriti_migration.py MUST use run_in_executor:
   It is CPU/IO bound and must never block the FastAPI event loop.
```

---

## INVARIANTS — NEVER VIOLATE (Sprint 5 additions)
```
EXISTING (enforced since Sprint 1):
  - torch imports confined to cloud/perception/ only
  - EMA update before predictor forward in engine.py
  - Ghost bounding boxes in pixel coordinates
  - SigReg gauge visible in Science Mode
  - CC-BY-SA notice in App.tsx
  - SmetiDB schema migrations idempotent (bump SCHEMA_VERSION)
  - SAG learned templates persisted on shutdown via lifespan
  - clear_all prune requires confirm_clear_all="CONFIRM_CLEAR_ALL"
  - Storage paths always absolute and resolved

SPRINT 5 ADDITIONS:
  - Migration: copy files FIRST, verify destination SECOND,
    update config LAST. Never update config if copy failed.
  - Migration: NEVER delete source data. Preservation is mandatory.
  - dry_run=True must NEVER create files, directories, or modify config.
  - W-matrix update in setu2_feedback: learning_rate=0.005 maximum.
    Higher rates cause catastrophic forgetting.
  - mandala-force-worker.ts: setTimeout(tick, 33) for ~30fps.
    Never requestAnimationFrame inside a Worker (not available).
  - DeepdiveView focus trap: restore focus to trigger element on close.
    Never leave keyboard focus stranded in the void.
  - All Canvas 2D draws: ctx.clearRect before every frame.
    Accumulated draws without clear cause memory growth on M1.
  - Person journal co-occurrence graph: circular layout only.
    Force simulation in PersonJournal is not needed — static is fine.
  - test_smriti_production.py is the AUTHORITATIVE production gate.
    All 11 tests must pass. Adding tests is allowed. Removing is not.
```

---

## COMPLETE ENTRY COMMAND

Paste this single block verbatim into Codex (GPT-5.4):
```
You are the Smriti Build System Orchestrator executing Sprint 5:
Production Polish for the TOORI repository.

CHECKPOINT: Sprint 4 complete. 162 tests passing. Storage configuration
fully implemented. Frontend builds clean. All prior sprints verified.

YOUR MISSION: Close all production gaps and deliver the complete Smriti
experience. Nine agents, in strict DAG order. No agent may begin until
all its dependencies are complete. After every agent, pytest -q must
show ≥ 162 — any drop stops all work immediately.

AGENT EXECUTION ORDER (strict):
  1. production_gate        → creates test_smriti_production.py (11 tests)
  2. data_migration         → migration service + API + 10 tests
  3. setu2_feedback         → W-matrix loop + API + 7 tests
  4. mandala_enhancement    → Web Worker + Canvas 2D rewrite
  5. deepdive_enhancement   → patch clicking + neighbors + keyboard nav
  6. person_journal_enhancement → co-occurrence graph
  7. accessibility_pass     → WCAG 2.1 AA across all Smriti components
  8. smriti_styles_sprint5  → CSS additions (append only, never replace)
  9. sprint5_validation     → 14 gates, honest count, state.json update

TARGET: ≥ 195 tests passing.
TELESCOPE CONTRACT: Must pass at every checkpoint without exception.
TORCH ISOLATION: Verified clean at Gate 12.
CANVAS CONSTRAINT: Canvas 2D only — no WebGL, no THREE.js in Smriti UI.
MIGRATION SAFETY: copy → verify → update config. Never delete source.
FOCUS MANAGEMENT: DeepdiveView focus trap + restoration on close.

Begin with production_gate now. Read every file before writing any code.

PART III: SPRINT 5 MANUAL TEST ADDITIONS
Add these sections to manual_test_guide.md in the repository.
---

## TEST SUITE 6 — SPRINT 5 PRODUCTION GATE

### Test 6.1: Full Production Gate
```bash
pytest -v cloud/api/tests/test_smriti_production.py
# Expected: 11 passed, 0 failed
# Every test name should appear:
#   test_telescope_behind_person_not_described_as_body_part  PASSED
#   test_office_chair_not_described_as_exercise_equipment    PASSED
#   test_uncertain_region_returns_uncertainty_not_...        PASSED
#   test_full_pipeline_hallucination_risk_bounded            PASSED
#   test_living_lens_tick_responds_within_sla                PASSED
#   test_smriti_recall_responds_within_sla                   PASSED
#   test_smriti_storage_config_persists_across_...           PASSED
#   test_watch_folder_list_persists_across_restart           PASSED
#   test_prune_clear_all_requires_exact_confirmation         PASSED
#   test_existing_toori_functionality_not_broken             PASSED
#   test_no_torch_imports_outside_perception                 PASSED
#   test_api_routes_all_present                              PASSED
```

### Test 6.2: Data Migration — Dry Run First
```bash
# Create test media directory with sample files
mkdir -p ~/Desktop/smriti_migrate_test/smriti/frames
mkdir -p ~/Desktop/smriti_migrate_test/smriti/thumbs
echo "fake db content" > ~/Desktop/smriti_migrate_test/smriti/smriti.sqlite3
echo "[]" > ~/Desktop/smriti_migrate_test/smriti/sag_templates.json

# Dry run: see what would be moved (runtime must be running)
curl -s -X POST http://127.0.0.1:7777/v1/smriti/storage/migrate \
  -H "Content-Type: application/json" \
  -d '{
    "target_data_dir": "/tmp/smriti_migration_target",
    "dry_run": true
  }' | python3 -m json.tool

# Expected:
# {
#   "success": true,
#   "dry_run": true,
#   "files_moved": N,        <- count of files that WOULD be moved
#   "bytes_moved_human": "X KB",
#   "new_data_dir": "/tmp/smriti_migration_target",
#   "errors": [],
#   "rollback_available": true
# }

# Verify dry_run did NOT create the target directory
ls /tmp/smriti_migration_target 2>&1
# Expected: "No such file or directory"
```

### Test 6.3: Data Migration — Live Run
```bash
# Now run the actual migration
curl -s -X POST http://127.0.0.1:7777/v1/smriti/storage/migrate \
  -H "Content-Type: application/json" \
  -d '{
    "target_data_dir": "/tmp/smriti_migration_live",
    "dry_run": false
  }' | python3 -m json.tool

# Expected:
# { "success": true, "dry_run": false, "files_moved": N, "errors": [] }

# Verify target directory was created
ls /tmp/smriti_migration_live/

# Verify source data was preserved
ls ~/Desktop/smriti_migrate_test/smriti/
# Original files should still exist

# Verify config now points to new location
curl -s http://127.0.0.1:7777/v1/smriti/storage | python3 -m json.tool
# "data_dir" should be "/tmp/smriti_migration_live"

# Clean up test dirs
rm -rf ~/Desktop/smriti_migrate_test /tmp/smriti_migration_live
```

### Test 6.4: Setu-2 Feedback Loop
```bash
# First, get a real media ID from the system (requires indexed media)
MEDIA_ID=$(curl -s -X POST http://127.0.0.1:7777/v1/smriti/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "person", "top_k": 1}' | \
  python3 -c "import json,sys; r=json.load(sys.stdin); print(r['results'][0]['media_id'] if r['results'] else 'NONE')")

echo "Media ID: $MEDIA_ID"

if [ "$MEDIA_ID" != "NONE" ]; then
  # Send confirmed feedback
  curl -s -X POST http://127.0.0.1:7777/v1/smriti/recall/feedback \
    -H "Content-Type: application/json" \
    -d "{
      \"query\": \"person\",
      \"media_id\": \"$MEDIA_ID\",
      \"confirmed\": true,
      \"session_id\": \"manual_test\"
    }" | python3 -m json.tool
  # Expected: { "updated": true, "w_mean": X.XXXX, "message": "W-matrix updated..." }

  # Send rejected feedback
  curl -s -X POST http://127.0.0.1:7777/v1/smriti/recall/feedback \
    -H "Content-Type: application/json" \
    -d "{
      \"query\": \"person\",
      \"media_id\": \"fake_obs_xyz\",
      \"confirmed\": false
    }" | python3 -m json.tool
  # Expected: { "updated": false, "message": "Media embedding not found..." }
fi
```

### Test 6.5: Mandala View — Web Worker Active
```
□ Open http://127.0.0.1:4173 in Chrome
□ Open Developer Tools → Performance tab
□ Click Smriti → Mandala sub-tab

IF corpus has 5+ clusters:
  □ Open DevTools → Sources → Workers
    Verify "mandala-force-worker" appears in the Workers list
  □ Clusters should animate into position over ~2 seconds
  □ Hover over a cluster — cursor changes to pointer, radius grows slightly
  □ Click a cluster — it expands with a spring animation
  □ Double-click a cluster — expansion fires onClusterExpand
  □ Use keyboard arrows ←/→ to cycle clusters
  □ Press Enter to expand focused cluster
  □ Press + / - to zoom in/out
  □ Press 0 to reset zoom and pan to center
  □ Scroll on canvas — zoom changes smoothly
  □ Bottom-left: zoom control buttons appear (+ / ⊙ / −)
  □ Bottom-right: zoom percentage label shows correct value

IF corpus is empty:
  □ Empty state message shows with ◎ symbol
  □ "Add a watch folder in Settings → Smriti Storage" shown
```

### Test 6.6: Deepdive — Interactive Patches
```
□ In Smriti → Recall, search for any query
□ Click a result to open Deepdive
□ Verify focus moves into the modal (Tab key cycles within modal)
□ Press E — energy overlay appears (14×14 colored grid over image)
□ Click a patch cell — detail popover appears top-right:
  - Shows patch index
  - Shows anchor template name (e.g. "person_torso" or "background_plane")
  - Shows depth stratum badge
  - Shows confidence %
  - Shows hallucination risk % with color coding
□ Click another patch — popover updates immediately
□ Click the same patch — popover closes (toggle)
□ Press E again — energy overlay hides
□ Semantic neighbors section (if corpus has 2+ items):
  - Grid of up to 6 thumbnail cells appears below
  - Hover shows Setu score in tooltip
□ Press F — modal goes fullscreen (browser permitting)
□ Press F again — exits fullscreen
□ Press Escape — modal closes
□ Verify focus returns to the recall result card that was clicked
```

### Test 6.7: Person Journal — Co-occurrence Graph
```
□ In Smriti → Journals sub-tab
□ Type a person name that exists in your indexed corpus
□ Click "Load Journal"
□ If EpistemicAtlas has co-occurrence data:
  - Canvas graph renders below the journal
  - Central orange node = the named person
  - Connected teal nodes = other people seen with them
  - Edge thickness indicates frequency of co-occurrence
  - Node size indicates total track length
□ Hover over canvas (no crash, no flicker)
□ If no co-occurrence data: section not shown (no empty white canvas)
```

### Test 6.8: Accessibility Verification
```
□ Open Chrome DevTools → Lighthouse
□ Run Accessibility audit on http://127.0.0.1:4173
□ Target: Accessibility score ≥ 85
  (Note: canvas elements are aria-hidden and excluded from audit)

□ Keyboard-only navigation test:
  □ Reload page, do NOT touch mouse
  □ Tab → Sidebar navigation items receive focus ring
  □ Tab to "Smriti" → Enter → Smriti tab opens
  □ Tab to sub-tab navigation → arrows cycle through sub-tabs
  □ Tab to Mandala canvas → focus ring appears on container
  □ Arrow keys navigate clusters (if any exist)
  □ Tab to recall search input → type query → Enter
  □ Tab to result card → Enter → Deepdive opens
  □ Verify focus is inside Deepdive modal (Tab cycles within it)
  □ Escape → Deepdive closes
  □ Verify focus returned to recall result card

□ Screen reader test (macOS VoiceOver):
  □ CMD+F5 to enable VoiceOver
  □ Navigate to Mandala → hear "Memory map with N clusters"
  □ Navigate to Recall results → each card described
  □ Navigate to Deepdive → hear modal title and close button
  □ CMD+F5 to disable VoiceOver

□ Reduced motion test:
  □ System Preferences → Accessibility → Reduce Motion: ON
  □ Reload page → Deepdive open/close has no animation
  □ Mandala cluster appearance has no animation
  □ Turn Reduce Motion back OFF
```

### Test 6.9: Full Regression — All 195 Tests
```bash
pytest -q \
  cloud/api/tests \
  cloud/jepa_service/tests \
  cloud/search_service/tests \
  cloud/monitoring/tests \
  tests/test_readme.py
# Expected: ≥ 195 passed, 0 failed

# Check state.json reflects completion
cat .smriti_build/state.json | python3 -m json.tool
# Expected:
# "build_complete": true
# "total_tests_passing": 195 (or higher)
# "met_target": true
# All agents: "complete"
```

---

## TROUBLESHOOTING — SPRINT 5
```
PROBLEM: mandala-force-worker.ts not found in DevTools → Workers
FIX:     Verify file exists: ls desktop/electron/src/components/smriti/mandala-force-worker.ts
         Verify Vite worker config: vite.config.ts should not exclude .ts workers
         Try: cd desktop/electron && npm run build && npm run preview (instead of dev)

PROBLEM: Deepdive focus does not return to trigger element on close
FIX:     DeepdiveView must store ref to previously focused element on mount:
         const prev = document.activeElement as HTMLElement;
         On cleanup: prev?.focus?.();
         Verify useEffect cleanup function calls focus.

PROBLEM: Migration "target_data_dir" 422 error
FIX:     The field name must be "target_data_dir" not "target_dir" or "path".
         Verify SmritiMigrationRequest model matches the request body.

PROBLEM: Setu-2 feedback returns updated=false for valid media
FIX:     Check that SmetiDB.get_smriti_media() returns an object with .embedding field.
         If media was ingested before Sprint 5, it may lack the embedding field.
         Re-ingest the media or check smriti_storage.py get_smriti_media() signature.

PROBLEM: Co-occurrence graph canvas renders blank
FIX:     Check PersonJournal.tsx journal.atlas path — the field name in the API
         response may differ from what the component expects.
         Log journal object in useEffect to verify structure.
         The graph only renders if nodes.length > 0.

PROBLEM: test_no_torch_imports_outside_perception fails
FIX:     grep -r "import torch" cloud/ --include="*.py" | grep -v "cloud/perception/"
         Remove any torch import found outside cloud/perception/.
         Check smriti_migration.py specifically — it must use only stdlib + numpy.

PROBLEM: Lighthouse accessibility score < 85
FIX:     Most likely cause: canvas elements missing aria-hidden="true"
         Also check: all icon-only buttons must have aria-label
         Run: npx axe-cli http://127.0.0.1:4173 for detailed violations

PROBLEM: W-matrix mean after feedback is 0.0 (not updating)
FIX:     Check that Setu2Bridge._W is initialized in setu2.py constructor.
         If _W = None, update_metric_w() will fail silently.
         Add: assert self._W is not None before the update call.

PROBLEM: ≥ 195 test target not met
FIX:     Run per-agent test counts to identify gap:
         pytest -q cloud/api/tests/test_smriti_production.py    # expect 11
         pytest -q cloud/api/tests/test_smriti_storage_config.py # expect 16
         pytest -q cloud/api/tests/test_smriti_data_migration.py # expect 10
         pytest -q cloud/api/tests/test_setu2_feedback.py        # expect 7
         Total from Sprint 5: +44 new tests
         162 baseline + 44 = 206 max. Gap = missing agent tests.
```

---

## EXPECTED SPRINT 5 TEST COUNT BREAKDOWN
```
SPRINT HISTORY (cumulative):
  Baseline (pre-Sprint 1):   84 tests
  Sprint 1 complete:        139 tests   (+55)
  Sprint 2+3 complete:      146 tests   (+7)
  Sprint 4 complete:        162 tests   (+16)
  Sprint 5 target:          ≥ 195 tests (+33 minimum)

SPRINT 5 TEST BUDGET:
  production_gate:          +11  (test_smriti_production.py — 11 new tests)
  data_migration:           +10  (test_smriti_data_migration.py — 10 new)
  setu2_feedback:           +7   (test_setu2_feedback.py — 7 new)
  mandala/deepdive/journal: +0   (UI components — no new backend tests)
  accessibility_pass:       +0   (CSS/markup — no new pytest tests)
  TOTAL NEW:                +28  minimum

  162 + 28 = 190 minimum. Target of 195 requires +5 additional tests
  from Codex's judgment — coverage gaps it identifies during implementation.
  Codex should not fabricate tests to hit the number; it should write tests
  that cover real behavior it discovers while reading the codebase.
```
PART IV: README.md AND CLAUDE.md — FINAL REQUIRED UPDATES
After Sprint 5 completes, these sections must be verified and updated if Codex has not already done so.

README.md — Final State Specification
## WHAT THE README MUST CONTAIN AFTER SPRINT 5

### 1. Feature matrix — update the table:

| Feature                     | Sprint | Status    |
|-----------------------------|--------|-----------|
| JEPA pipeline (TPDS/SAG/CWMA/ECGD/Setu-2) | 1 | ✓ |
| Live Lens camera + tick API | 1      | ✓         |
| Smriti ingestion daemon     | 2+3    | ✓         |
| Smriti UI (Mandala/Recall/Deepdive) | 2+3 | ✓     |
| Storage configuration       | 4      | ✓         |
| Watch folder management     | 4      | ✓         |
| Data migration              | 5      | ✓         |
| Setu-2 W-matrix feedback    | 5      | ✓         |
| Mandala Web Worker          | 5      | ✓         |
| Deepdive interactive patches| 5      | ✓         |
| Person co-occurrence graph  | 5      | ✓         |
| WCAG 2.1 AA accessibility   | 5      | ✓         |

### 2. Smriti Quick Start — update with storage note:

## Smriti Quick Start

1. Start the runtime: `TOORI_DATA_DIR=.toori uvicorn cloud.api.main:app --port 7777`
2. Start the frontend: `cd desktop/electron && npm run web`
3. Open http://127.0.0.1:4173
4. Settings → Smriti Storage → configure data directory
   ⚠ On M1 iMac with 256GB SSD, point to an external drive before indexing
5. Settings → Smriti Storage → Watch Folders → + Add Folder → select your Photos
6. Wait for indexing (Smriti → HUD shows progress)
7. Smriti → Recall → type any query ("red jacket", "beach sunset", "my cat")
8. Click a result → Deepdive → press E to see JEPA energy patches
9. In Recall results, click ✓ or ✗ on results to improve future recall

### 3. API route table — all 17 Smriti routes:

| Method | Route | Description |
|--------|-------|-------------|
| POST | /v1/smriti/ingest | Ingest a media file |
| POST | /v1/smriti/recall | Semantic query recall |
| GET  | /v1/smriti/status | Ingestion status |
| GET  | /v1/smriti/clusters | Cluster list for Mandala |
| GET  | /v1/smriti/metrics | Performance metrics |
| POST | /v1/smriti/tag/person | Tag a person in media |
| GET  | /v1/smriti/person/{name}/journal | Person journal |
| GET  | /v1/smriti/storage | Storage configuration |
| PUT  | /v1/smriti/storage | Update storage config |
| GET  | /v1/smriti/storage/usage | Disk usage report |
| GET  | /v1/smriti/watch-folders | List watched folders |
| POST | /v1/smriti/watch-folders | Add watch folder |
| DELETE | /v1/smriti/watch-folders | Remove watch folder |
| POST | /v1/smriti/storage/prune | Prune storage |
| POST | /v1/smriti/storage/migrate | Migrate to new location |
| POST | /v1/smriti/recall/feedback | Setu-2 W-matrix feedback |
| GET  | /v1/smriti/media/{id}/neighbors | Semantic neighbors |

### 4. Test count badge — update in header:
![Tests](https://img.shields.io/badge/tests-195%2B-brightgreen)

CLAUDE.md — Final State Specification
## WHAT CLAUDE.md MUST CONTAIN AFTER SPRINT 5

### 1. Sprint 5 new files — add to Primary Entry Points:

  NEW IN SPRINT 5:
  - cloud/api/tests/test_smriti_production.py
      THE authoritative production gate. 11 tests.
      test_telescope_behind_person_not_described_as_body_part is the
      primary contract. Never remove or weaken it.

  - cloud/runtime/smriti_migration.py
      Data migration service. Safety invariants:
        * copy → verify → update config. Never reverse this order.
        * Never delete source data. User deletes manually.
        * dry_run=True must never write anything.
      run_in_executor only — never block the event loop.

  - desktop/electron/src/components/smriti/mandala-force-worker.ts
      Web Worker for force simulation. Uses setTimeout(tick, 33).
      requestAnimationFrame is NOT available in Worker context.
      Communicates positions as [id, x, y] tuples via postMessage.
      No npm imports — built-in types only.

### 2. Setu-2 feedback invariants — add to the Smriti Architecture section:

  SETU-2 W-MATRIX FEEDBACK:
    - feedback endpoint: POST /v1/smriti/recall/feedback
    - confirmed=True  → positive pair → pull embeddings closer
    - confirmed=False → negative pair → push embeddings apart
    - learning_rate is capped at 0.005. Never exceed this.
    - W-matrix is per-RuntimeContainer — not persisted to disk in Sprint 5
      (Sprint 6: add W serialization to settings)
    - Feedback is best-effort: missing media_id returns updated=False,
      not an HTTP error. Frontend must handle this gracefully.

### 3. DeepdiveView accessibility contract:

  DEEPDIVE FOCUS CONTRACT (WCAG 2.1 AA — never regress):
    - On open: focus moves to modal container (tabIndex={-1} + .focus())
    - While open: focus trapped within modal (Tab/Shift+Tab cycle)
    - On close: focus restored to the element that triggered open
    - Escape key closes from anywhere within the modal
    - All interactive elements within modal have accessible names

### 4. Canvas render invariant — add:

  ALL CANVAS ELEMENTS IN SMRITI:
    - Must call ctx.clearRect(0, 0, W, H) at the start of every frame
    - Must call ctx.save() / ctx.restore() around transformed draws
    - Must use Canvas 2D API only (getContext("2d"))
    - WebGL, THREE.js, and WebGPU are prohibited in Smriti components
    - ResizeObserver must update canvas.width/height on container resize
    - devicePixelRatio scaling must be applied after resize

### 5. Production gate invariants — add:

  test_smriti_production.py GOVERNANCE:
    - ALL 11 tests must pass in CI before any merge to main
    - Adding new tests to this file is encouraged
    - Removing or skipping tests requires explicit sign-off
    - Test names are contractual: do not rename them
    - The telescope test is the permanent sentinel: if it ever
      fails, all engineering work stops until it passes
```

---

## COMPLETE TEST COUNT REFERENCE — ALL SPRINTS
```
Sprint | Tests | New  | Key deliverables
-------|-------|------|------------------
Base   |    84 |   —  | Original TOORI
S1     |   139 |  +55 | TPDS, SAG, CWMA, ECGD, Setu-2, JEPAWorkerPool, SmetiDB
S2+3   |   146 |   +7 | PyAV, watchdog, SAG persistence, all UI components
S4     |   162 |  +16 | Storage config, watch folders, budget, prune
S5     | ≥ 195 |  +33 | Production gate, migration, feedback, Mandala Worker,
       |       |      | Deepdive patches, co-occurrence graph, WCAG 2.1 AA
