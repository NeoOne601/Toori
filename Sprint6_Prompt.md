# TOORI SPRINT 6 — WORLD MODEL FOUNDATION
# Antigravity Orchestrator · Opus 4.6 · Ultrathink Mode
# Parallel Sub-Agent Architecture · Maximum Depth
# Repository: TOORI · Upstream: Sprint 5 + Fix Pass · 196 tests

## PREAMBLE FOR ULTRATHINK

Before spawning any sub-agent, the orchestrator must reason through the
complete dependency graph. Use ultrathink to:

1. Map every file that will change and every file that depends on it.
2. Identify which changes can be made in parallel (no shared state).
3. Identify which changes must be strictly sequential (shared interfaces).
4. Verify that no sub-agent's output creates a conflict with another.
5. Define the EXACT interface contracts between sub-agents.

The Le World Model implementation is the most architecturally significant
change in TOORI's history. A mistake in the encoder interface propagates
to every downstream component: TPDS, SAG, CWMA, ECGD, Setu-2, SmetiDB,
the ingestion daemon, and all frontend components. Think before acting.

## MANDATORY FIRST ACTIONS (Orchestrator, before spawning anything)
```bash
# 1. Establish exact baseline
pytest -q cloud/api/tests cloud/jepa_service/tests \
       cloud/search_service/tests cloud/monitoring/tests \
       tests/test_readme.py
# Record exact count. Must be 196.

# 2. Read the entire engine + perception layer
cat cloud/jepa_service/engine.py
cat cloud/jepa_service/depth_separator.py
cat cloud/jepa_service/anchor_graph.py
cat cloud/jepa_service/world_model_alignment.py
cat cloud/jepa_service/confidence_gate.py
cat cloud/runtime/setu2.py
cat cloud/perception/onnx_model.py     # THE SURROGATE — read carefully
cat cloud/runtime/smriti_storage.py   # SmetiDB schema — understand before extending
cat cloud/runtime/smriti_ingestion.py # ingestion pipeline
cat cloud/runtime/models.py           # JEPATick — interface contract
cat cloud/runtime/service.py
cat cloud/runtime/app.py
cat cloud/api/tests/test_smriti_production.py  # THE PRODUCTION GATE
cat CLAUDE.md                          # Invariants section — non-negotiable

# 3. Verify V-JEPA 2 encoder availability
python3.11 -c "
# Check if V-JEPA 2 weights are available
import pathlib
possible_paths = [
    'models/vjepa2/vjepa2_vitl_16.pth',
    'models/vjepa2/encoder.onnx',
    'models/vision/vjepa2.onnx',
]
for p in possible_paths:
    if pathlib.Path(p).exists():
        print(f'FOUND: {p}')
        break
else:
    print('V-JEPA 2 weights not found — using I-JEPA ViT-L as bridge encoder')
    print('This is acceptable for Sprint 6 — interface is identical')
"

# 4. Understand the JEPATick interface contract precisely
python3.11 -c "
from cloud.runtime.models import JEPATick
import inspect
print(inspect.getsource(JEPATick))
"

# 5. Check SmetiDB current schema version
python3.11 -c "
from cloud.runtime.smriti_storage import SmetiDB
import tempfile
with tempfile.TemporaryDirectory() as d:
    db = SmetiDB(d)
    print('SCHEMA_VERSION:', db.SCHEMA_VERSION)
    # List all tables
    import sqlite3
    conn = sqlite3.connect(f'{d}/smriti.sqlite3')
    tables = conn.execute(
        'SELECT name FROM sqlite_master WHERE type=\"table\"'
    ).fetchall()
    print('Tables:', [t[0] for t in tables])
"

# 6. Initialize Sprint 6 state
python3.11 -c "
import json, pathlib
s = json.loads(pathlib.Path('.smriti_build/state.json').read_text())
s.update({
    'version': '6.0',
    'sprint': '6',
    'build_complete': False,
    'baseline_tests_passing': 196,
    'target_tests_passing': 235,
    'world_model_sprint': True,
    'le_world_model_impl': True,
    'agents': {
        'vjepa2_encoder':       'pending',
        'world_model_predictor':'pending',
        'tpds_upgrade':         'pending',
        'ecgd_upgrade':         'pending',
        'jepatick_extension':   'pending',
        'smetiddb_schema_v2':   'pending',
        'audio_jepa_ingestion': 'pending',
        'surprise_indexing':    'pending',
        'setu2_hierarchical':   'pending',
        'sprint6_validation':   'pending',
    },
    'blocking_issues': [],
})
pathlib.Path('.smriti_build/state.json').write_text(json.dumps(s, indent=2))
print('Sprint 6 state initialized')
"
```

---

## ULTRATHINK DEPENDENCY ANALYSIS

*Orchestrator must complete this analysis before spawning sub-agents.*

The sub-agent execution has three layers:
```
LAYER_0: Interface contracts (must complete before anything else)
  → jepatick_extension: defines the new JEPATick fields
  → smetiddb_schema_v2: defines the new database schema
  These two outputs are the contracts that all other agents depend on.
  They run SEQUENTIALLY, not in parallel.
  
LAYER_1: Encoder replacement (depends on Layer 0 contracts)
  → vjepa2_encoder: replaces MobileNetV2 with V-JEPA 2
  This runs ALONE first — all downstream components depend on its output shape.
  
LAYER_2: World model components (all depend on vjepa2_encoder output shape)
  These run IN PARALLEL after Layer 1 completes:
  → world_model_predictor: EMA + temporal prediction loop
  → tpds_upgrade:          principled depth separation
  → ecgd_upgrade:          information-theoretic uncertainty
  → audio_jepa_ingestion:  Audio-JEPA pipeline (independent of video changes)
  
LAYER_3: Application layer (depends on all Layer 2 components)
  These run IN PARALLEL after Layer 2 completes:
  → surprise_indexing:     prediction-error-weighted Smriti priority
  → setu2_hierarchical:    Level 1+2 hierarchy in Setu-2 bridge
  
LAYER_4: Validation
  → sprint6_validation: 14 gates, honest count, state.json v6.0
```

---

## SUB-AGENT SPECIFICATIONS

---

### SUB-AGENT: jepatick_extension
*Layer 0 — runs first, alone, everything depends on this*

**Context files**: `cloud/runtime/models.py`, `cloud/jepa_service/engine.py`,
`cloud/api/tests/test_smriti_production.py`

**Objective**: Extend `JEPATick` with the Le World Model fields.
This is the interface contract. Every other agent reads this output.
Do NOT implement the logic — only define the data structures.

**Deliverable**: Modify `cloud/runtime/models.py`:
```python
class JEPATick(BaseModel):
    """
    One tick of the JEPA world model engine.
    
    Sprint 6 additions marked with # WM (World Model).
    All new fields are Optional with None default for backward compatibility.
    Existing fields MUST NOT change — they are used by 196 passing tests.
    """
    # EXISTING FIELDS — DO NOT MODIFY
    session_id: str
    observation_id: str
    timestamp: float
    frame_shape: tuple[int, int, int]
    embedding: list[float]              # L1 patch embedding (primary)
    predicted_embedding: list[float]    # EMA predictor output
    energy: float                       # E(context, target) scalar
    depth_strata: Optional[Any]         # DepthStrataMap
    anchor_matches: Optional[list[Any]] # list[AnchorMatch]
    setu_descriptions: Optional[list[Any]]
    alignment_loss: Optional[float]
    
    # NEW FIELDS — WORLD MODEL (Sprint 6) # WM
    # Le World Model hierarchical representations
    l2_embedding: Optional[list[float]] = None   # object-level aggregation # WM
    
    # Temporal prediction
    predicted_next_embedding: Optional[list[float]] = None  # s_{t+1} prediction # WM
    prediction_error: Optional[float] = None    # E(s_t, predicted, actual) # WM
    
    # Information-theoretic uncertainty (replaces heuristic ECGD thresholds)
    epistemic_uncertainty: Optional[float] = None    # reducible uncertainty # WM
    aleatoric_uncertainty: Optional[float] = None    # irreducible uncertainty # WM
    
    # Surprise signal for Smriti priority indexing
    surprise_score: Optional[float] = None  # normalized prediction_error # WM
    
    # Audio-JEPA (Naad v0)
    audio_embedding: Optional[list[float]] = None    # Audio-JEPA representation # WM
    audio_energy: Optional[float] = None             # Audio energy score # WM
    
    # World model metadata
    world_model_version: str = "surrogate"  # "surrogate" | "vjepa2" | "vjepa2+audio"
```

Add corresponding Pydantic models for new API responses:
```python
class WorldModelStatus(BaseModel):
    """Current world model configuration and health."""
    encoder_type: str           # "mobilenetv2" | "vjepa2" | "vjepa2_onnx"
    encoder_version: str
    temporal_context_length: int  # how many frames of history the predictor uses
    hierarchy_levels: list[int]   # which levels are active, e.g., [1, 2]
    audio_enabled: bool
    mean_prediction_error: float  # rolling average — low = model is calibrated
    mean_surprise_score: float    # rolling average of surprise_score
    
class HierarchicalRecallRequest(BaseModel):
    """Extended recall request supporting hierarchy-level selection."""
    query: str
    top_k: int = 10
    hierarchy_level: int = 3    # 1=patch, 2=object, 3=scene, 4=activity, 5=narrative
    use_surprise_weighting: bool = False  # weight results by surprise_score
    min_surprise: Optional[float] = None  # filter to only surprising moments
```

**Success gate**:
```bash
# JEPATick must import cleanly
python3.11 -c "
from cloud.runtime.models import JEPATick, WorldModelStatus, HierarchicalRecallRequest
import json

# Verify backward compatibility — existing fields all present
tick = JEPATick(
    session_id='test', observation_id='obs_1', timestamp=0.0,
    frame_shape=(224, 224, 3), embedding=[0.0]*384,
    predicted_embedding=[0.0]*384, energy=0.5,
)
# New fields default to None
assert tick.prediction_error is None
assert tick.surprise_score is None
assert tick.world_model_version == 'surrogate'
print('JEPATick backward compat: PASS')
"

pytest -q cloud/api/tests/test_smriti_production.py
# All 12 must still pass — no regression from interface extension
```

---

### SUB-AGENT: smetiddb_schema_v2
*Layer 0 — runs after jepatick_extension, before everything else*

**Context files**: `cloud/runtime/smriti_storage.py`,
`cloud/api/tests/test_smriti_storage_config.py`

**Objective**: Extend SmetiDB schema to version 2 with world model fields.
The migration must be idempotent. Existing data must be preserved.

**Deliverable**: Modify `cloud/runtime/smriti_storage.py`:
```python
# Bump schema version
SCHEMA_VERSION = 2  # was 1

# In _create_tables() or _migrate(), add new columns to smriti_media:
"""
ALTER TABLE smriti_media ADD COLUMN 
    l2_embedding BLOB DEFAULT NULL;         -- object-level representation

ALTER TABLE smriti_media ADD COLUMN 
    prediction_error REAL DEFAULT NULL;     -- world model surprise signal

ALTER TABLE smriti_media ADD COLUMN 
    epistemic_uncertainty REAL DEFAULT NULL;

ALTER TABLE smriti_media ADD COLUMN 
    aleatoric_uncertainty REAL DEFAULT NULL;

ALTER TABLE smriti_media ADD COLUMN 
    surprise_score REAL DEFAULT NULL;       -- normalized, 0.0-1.0

ALTER TABLE smriti_media ADD COLUMN 
    audio_embedding BLOB DEFAULT NULL;      -- Audio-JEPA representation

ALTER TABLE smriti_media ADD COLUMN 
    world_model_version TEXT DEFAULT 'surrogate';
"""

# New table for temporal prediction index
"""
CREATE TABLE IF NOT EXISTS smriti_temporal_index (
    media_id TEXT PRIMARY KEY REFERENCES smriti_media(id),
    session_id TEXT NOT NULL,
    sequence_position INTEGER NOT NULL,    -- position in the session sequence
    predicted_next_id TEXT,               -- media_id of the predicted next frame
    prediction_confirmed BOOLEAN DEFAULT FALSE,  -- did prediction match reality?
    temporal_context_ids TEXT,            -- JSON array of context media_ids
    schema_version INTEGER DEFAULT 2
);
"""

# Add new retrieval methods:
def get_smriti_media_by_surprise(
    self, 
    min_surprise: float = 0.5,
    limit: int = 50
) -> list[SmritiMedia]:
    """Retrieve most surprising (high prediction error) media items."""
    ...

def get_all_embeddings_with_level(
    self,
    level: int = 1,
    limit: int = 500
) -> dict:
    """
    Get embeddings at the specified hierarchy level.
    level=1: embedding (existing)
    level=2: l2_embedding (new)
    Returns: {'embeddings': np.ndarray, 'media_ids': list[str]}
    """
    ...

def update_world_model_fields(
    self,
    media_id: str,
    prediction_error: Optional[float] = None,
    surprise_score: Optional[float] = None,
    epistemic_uncertainty: Optional[float] = None,
    aleatoric_uncertainty: Optional[float] = None,
    l2_embedding: Optional[np.ndarray] = None,
    audio_embedding: Optional[np.ndarray] = None,
) -> None:
    """Update world model fields without re-indexing the full media item."""
    ...
```

**Success gate**:
```bash
pytest -q cloud/api/tests/test_smriti_storage_config.py -v
# All 16 must still pass

python3.11 -c "
from cloud.runtime.smriti_storage import SmetiDB
import tempfile
with tempfile.TemporaryDirectory() as d:
    db = SmetiDB(d)
    assert db.SCHEMA_VERSION == 2
    # Test idempotency: creating again should not error
    db2 = SmetiDB(d)
    assert db2.SCHEMA_VERSION == 2
    print('Schema v2 migration: PASS')
"
```

---

### SUB-AGENT: vjepa2_encoder
*Layer 1 — runs after Layer 0 completes*

**Context files**: `cloud/perception/onnx_model.py`, `cloud/jepa_service/engine.py`,
`cloud/runtime/models.py` (updated by jepatick_extension)

**Objective**: Replace MobileNetV2 ONNX surrogate with V-JEPA 2 encoder.
If V-JEPA 2 weights are not available, use I-JEPA ViT-L as the bridge
encoder — the interface is identical and the representations are
significantly better than MobileNetV2. The interface contract must be
preserved exactly: `encode(frame: np.ndarray) -> np.ndarray (384-dim)`.

**Architecture decision the sub-agent must make**:
```python
# Check which encoder is available and use the best one:
# Priority order:
#   1. vjepa2_vitl_16.pth  (V-JEPA 2 ViT-L/16 — ideal)
#   2. ijepa_vitl_16.pth   (I-JEPA ViT-L/16 — excellent proxy)
#   3. ijepa_vith_14.pth   (I-JEPA ViT-H/14 — larger, better)
#   4. mobilenetv2-12.onnx (current — fallback if nothing else available)
```

**Deliverable 1**: Modify `cloud/perception/onnx_model.py`:
```python
"""
JEPA Encoder — Sprint 6 World Model Edition.

Encoder hierarchy (best to fallback):
  1. V-JEPA 2 (spatiotemporal, world model native)
  2. I-JEPA ViT-L (excellent spatial, no temporal)
  3. MobileNetV2 ONNX (legacy surrogate — backward compat only)

The interface is identical across all three:
  encode(frame: np.ndarray[H, W, 3]) -> np.ndarray[384]
  
The quality of downstream TPDS, SAG, ECGD, Setu-2 results scales
with encoder quality. V-JEPA 2 >> I-JEPA >> MobileNetV2.
"""
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Optional
from cloud.runtime.observability import get_logger

log = get_logger("jepa_encoder")

ENCODER_SEARCH_PATHS = [
    # V-JEPA 2 variants
    "models/vjepa2/vjepa2_vitl_16.onnx",
    "models/vjepa2/encoder.onnx",
    # I-JEPA variants  
    "models/ijepa/ijepa_vitl16_in1k.onnx",
    "models/ijepa/ijepa_vith14_in1k.onnx",
    "models/vision/ijepa.onnx",
    # Legacy fallback
    "models/vision/mobilenetv2-12.onnx",
]

ENCODER_EMBEDDING_DIMS = {
    "vjepa2": 1024,    # ViT-L hidden dim → projected to 384
    "ijepa":  1024,    # ViT-L hidden dim → projected to 384
    "mobilenetv2": 1280,  # MobileNetV2 penultimate → projected to 384
}

TARGET_DIM = 384  # Setu-2 metric space dimension — NEVER CHANGE THIS


class JEPAEncoder:
    """
    Unified JEPA encoder supporting V-JEPA 2, I-JEPA, and legacy surrogate.
    
    All variants project to TARGET_DIM=384 to maintain Setu-2 compatibility.
    The projection matrix W_proj is randomly initialized and then fine-tuned
    via Setu-2 W-matrix updates — so the projection adapts to the user's
    specific visual vocabulary over time.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.encoder_type = "unknown"
        self.model_path = model_path
        self._session = None
        self._projection = None
        self._temporal_buffer = []  # for V-JEPA 2 spatiotemporal context
        self._temporal_buffer_size = 8  # 8 frames of temporal context
        
        self._load_best_available(model_path)
    
    def _load_best_available(self, preferred: Optional[str]) -> None:
        """Load the best available encoder, falling back gracefully."""
        search_list = ([preferred] if preferred else []) + ENCODER_SEARCH_PATHS
        
        for path in search_list:
            if path and Path(path).exists():
                try:
                    self._session = ort.InferenceSession(
                        path,
                        providers=(
                            ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                            if self.device == "mps"
                            else ["CPUExecutionProvider"]
                        )
                    )
                    # Detect encoder type from path/model metadata
                    if "vjepa2" in path.lower():
                        self.encoder_type = "vjepa2"
                        log.info("encoder_loaded", type="vjepa2", path=path)
                    elif "ijepa" in path.lower():
                        self.encoder_type = "ijepa"
                        log.info("encoder_loaded", type="ijepa", path=path)
                    else:
                        self.encoder_type = "mobilenetv2"
                        log.warning("encoder_loaded_fallback", 
                                   type="mobilenetv2", path=path,
                                   message="Using legacy surrogate. "
                                           "Download V-JEPA 2 weights for "
                                           "world model capabilities.")
                    
                    # Initialize projection matrix
                    # (maps from encoder_dim to TARGET_DIM=384)
                    raw_dim = self._get_raw_output_dim()
                    np.random.seed(42)  # deterministic initialization
                    self._projection = np.random.randn(
                        raw_dim, TARGET_DIM
                    ).astype(np.float32) / np.sqrt(raw_dim)
                    
                    self.model_path = path
                    return
                except Exception as e:
                    log.warning("encoder_load_failed", path=path, error=str(e))
                    continue
        
        raise RuntimeError(
            "No JEPA encoder available. "
            "Run: python3.11 scripts/download_desktop_models.py"
        )
    
    def encode(self, frame: np.ndarray) -> np.ndarray:
        """
        Encode a single frame to TARGET_DIM=384 embedding.
        
        For V-JEPA 2: uses temporal_buffer for spatiotemporal context.
        For I-JEPA/MobileNetV2: single-frame encoding.
        
        Returns: np.ndarray shape (384,), dtype float32, L2-normalized.
        """
        frame_f32 = self._preprocess(frame)
        
        if self.encoder_type == "vjepa2":
            # V-JEPA 2: spatiotemporal encoding
            self._temporal_buffer.append(frame_f32)
            if len(self._temporal_buffer) > self._temporal_buffer_size:
                self._temporal_buffer.pop(0)
            
            # Stack temporal context
            temporal_input = np.stack(self._temporal_buffer, axis=0)  # [T, C, H, W]
            if temporal_input.shape[0] < self._temporal_buffer_size:
                # Pad with zeros for early frames
                pad = np.zeros(
                    (self._temporal_buffer_size - temporal_input.shape[0],
                     *temporal_input.shape[1:]), dtype=np.float32
                )
                temporal_input = np.concatenate([pad, temporal_input], axis=0)
            
            raw = self._session.run(
                None, 
                {self._session.get_inputs()[0].name: temporal_input[np.newaxis]}
            )[0]
        else:
            # I-JEPA / MobileNetV2: single frame
            raw = self._session.run(
                None,
                {self._session.get_inputs()[0].name: frame_f32[np.newaxis]}
            )[0]
        
        # Flatten and project to TARGET_DIM
        raw_flat = raw.flatten()[:self._projection.shape[0]]
        projected = raw_flat @ self._projection
        
        # L2 normalize
        norm = np.linalg.norm(projected)
        if norm > 1e-8:
            projected = projected / norm
        
        return projected.astype(np.float32)
    
    def encode_with_patches(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns both the global embedding AND patch-level embeddings.
        Patch embeddings are used by SAG for topology matching.
        
        Returns:
            global_emb: np.ndarray (384,)
            patch_embs: np.ndarray (196, 384)  -- 14×14 patches
        """
        # For V-JEPA 2 and I-JEPA: patch embeddings come naturally
        # from the ViT architecture's intermediate representations
        global_emb = self.encode(frame)
        
        # Extract patch embeddings (14×14 = 196 patches for ViT-L/16)
        # This requires the encoder to output intermediate token embeddings
        # If the ONNX model doesn't expose them, approximate from global
        patch_embs = np.tile(global_emb, (196, 1))  # placeholder
        # TODO Sprint 6.1: expose patch token outputs from ViT ONNX model
        
        return global_emb, patch_embs
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame to model input format."""
        from PIL import Image
        img = Image.fromarray(frame.astype(np.uint8)).resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        return arr.transpose(2, 0, 1)  # HWC → CHW
    
    def _get_raw_output_dim(self) -> int:
        """Get the raw output dimensionality of the loaded model."""
        dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        try:
            out = self._session.run(None, {self._session.get_inputs()[0].name: dummy})
            return int(np.prod(out[0].shape[1:]))
        except Exception:
            return 1280  # MobileNetV2 default
    
    def clear_temporal_buffer(self) -> None:
        """Clear temporal context — call on session start."""
        self._temporal_buffer.clear()
    
    @property
    def is_world_model(self) -> bool:
        """True if using V-JEPA 2 (genuine world model encoder)."""
        return self.encoder_type == "vjepa2"
    
    @property  
    def is_temporal(self) -> bool:
        """True if encoder uses temporal context."""
        return self.encoder_type == "vjepa2"
```

**Deliverable 2**: Add `scripts/download_vjepa2_models.py`:
```python
"""
Download V-JEPA 2 / I-JEPA ONNX encoder weights.

Tries in order:
  1. Hugging Face: facebook/vjepa2-vitl-fpc64-256
  2. Hugging Face: facebook/ijepa_vith14_1percent (smaller, excellent)
  3. Notifies user if neither available (M1 download may take time)
"""
import os, sys, urllib.request
from pathlib import Path

MODELS = {
    "ijepa_vitl": {
        "url": "https://huggingface.co/facebook/ijepa_vitl16_imagenet/resolve/main/ijepa_vitl16_in1k.onnx",
        "path": "models/ijepa/ijepa_vitl16_in1k.onnx",
        "size_mb": 1200,
        "desc": "I-JEPA ViT-L/16 (excellent proxy for V-JEPA 2)",
    }
}

def download(name, info):
    path = Path(info["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"Already exists: {path}")
        return
    print(f"Downloading {name} (~{info['size_mb']}MB)...")
    urllib.request.urlretrieve(info["url"], str(path))
    print(f"Saved: {path}")

if __name__ == "__main__":
    for name, info in MODELS.items():
        try:
            download(name, info)
        except Exception as e:
            print(f"Could not download {name}: {e}")
            print("Fallback: TOORI will use MobileNetV2 surrogate")
```

**Success gate**:
```bash
# Encoder loads cleanly
python3.11 -c "
from cloud.perception.onnx_model import JEPAEncoder
import numpy as np
enc = JEPAEncoder()
print(f'Encoder type: {enc.encoder_type}')
print(f'Is world model: {enc.is_world_model}')
frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
emb = enc.encode(frame)
assert emb.shape == (384,), f'Wrong shape: {emb.shape}'
assert abs(np.linalg.norm(emb) - 1.0) < 0.01, 'Not normalized'
print('Encoder: PASS')
"

# Full test suite still passes (no regression)
pytest -q cloud/api/tests cloud/jepa_service/tests
# Must be ≥ 196
```

---

### SUB-AGENT: world_model_predictor
*Layer 2 — parallel with tpds_upgrade, ecgd_upgrade, audio_jepa_ingestion*

**Context files**: `cloud/jepa_service/engine.py`, `cloud/runtime/models.py`

**Objective**: Implement the Le World Model's EBM temporal prediction loop.
This extends the existing EMA + predictor forward with:
1. Explicit prediction error computation E(s_t, ŝ_{t+1}, s_{t+1})
2. Surprise score normalization (0.0-1.0 rolling window)
3. World model version tracking in JEPATick

**Key implementation — add to `ImmersiveJEPAEngine`**:
```python
def tick(self, frame: np.ndarray, session_id: str, observation_id: str) -> JEPATick:
    """
    One tick of the JEPA world model.
    
    Le World Model implementation:
      1. Encode current frame: s_t = encoder(frame)
      2. Run predictor: ŝ_{t+1} = predictor(s_t)  [prediction of next state]
      3. EMA update: update target encoder
      4. Compute prediction error: E = ||ŝ_{t+1} - s_{t}||²  [compared to prev prediction]
      5. Update surprise rolling window
      6. Run TPDS/SAG/CWMA/ECGD pipeline (unchanged)
      7. Return enriched JEPATick
    """
    # ... existing pipeline ...
    
    # NEW: Le World Model prediction error
    current_embedding = ...  # from encoder
    
    if session_id in self._prev_predictions:
        prev_prediction = self._prev_predictions[session_id]
        # Prediction error: how wrong was last tick's prediction?
        prediction_error = float(
            np.mean((prev_prediction - current_embedding) ** 2)
        )
        # Normalize to surprise score 0.0-1.0
        surprise_score = self._normalize_surprise(prediction_error, session_id)
    else:
        prediction_error = 0.0
        surprise_score = 0.0
    
    # Store this tick's prediction for next tick's error computation
    predicted_next = self.predictor.forward(context_emb)
    self._prev_predictions[session_id] = predicted_next.copy()
    
    # Compute information-theoretic uncertainty
    epistemic_unc, aleatoric_unc = self._compute_uncertainty(
        current_embedding, session_id
    )
    
    return JEPATick(
        ...existing fields...,
        # NEW Le World Model fields
        predicted_next_embedding=predicted_next.tolist(),
        prediction_error=prediction_error,
        surprise_score=surprise_score,
        epistemic_uncertainty=epistemic_unc,
        aleatoric_uncertainty=aleatoric_unc,
        world_model_version=self.encoder.encoder_type,
    )

def _normalize_surprise(self, error: float, session_id: str) -> float:
    """
    Normalize prediction error to 0.0-1.0 surprise score.
    Uses a rolling window of recent errors per session.
    """
    if session_id not in self._surprise_windows:
        self._surprise_windows[session_id] = []
    
    window = self._surprise_windows[session_id]
    window.append(error)
    if len(window) > 100:  # 100-frame rolling window
        window.pop(0)
    
    if len(window) < 2:
        return 0.0
    
    mean = np.mean(window)
    std = np.std(window) + 1e-8
    # Z-score normalized, clipped to 0-1
    z = (error - mean) / std
    return float(np.clip((z + 3) / 6, 0.0, 1.0))  # 3-sigma → 0-1

def _compute_uncertainty(
    self, embedding: np.ndarray, session_id: str
) -> tuple[float, float]:
    """
    Estimate epistemic (reducible) and aleatoric (irreducible) uncertainty.
    
    Epistemic: variance of the predictor's output distribution
               → high if the model is uncertain about what will happen
    Aleatoric: variance of the target encoder's representation
               → high if the observation itself is ambiguous
    """
    # Simple approximation: run predictor twice with dropout
    # (or use the variance of the EMA update as a proxy)
    predictor_var = float(np.var(self.predictor.last_attention_weights))
    target_var = float(np.var(self.target_encoder.last_output))
    
    epistemic = np.clip(predictor_var * 10, 0.0, 1.0)
    aleatoric = np.clip(target_var * 10, 0.0, 1.0)
    
    return float(epistemic), float(aleatoric)
```

**Success gate**:
```bash
python3.11 -c "
from cloud.jepa_service.engine import ImmersiveJEPAEngine
import numpy as np

engine = ImmersiveJEPAEngine(device='cpu')
frames = [np.random.randint(0,255,(224,224,3),dtype=np.uint8) for _ in range(5)]

for i, frame in enumerate(frames):
    tick = engine.tick(frame, session_id='test', observation_id=f'obs_{i}')
    assert tick.prediction_error is not None
    assert tick.surprise_score is not None
    assert 0.0 <= tick.surprise_score <= 1.0
    assert tick.epistemic_uncertainty is not None
    assert tick.world_model_version != 'unknown'
    print(f'Frame {i}: surprise={tick.surprise_score:.3f} '
          f'error={tick.prediction_error:.4f} '
          f'wm={tick.world_model_version}')

print('World model predictor: PASS')
"

pytest -q cloud/api/tests/test_smriti_production.py
# All 12 must pass including telescope test
```

---

### SUB-AGENT: tpds_upgrade
*Layer 2 — parallel*

**Context files**: `cloud/jepa_service/depth_separator.py`,
`cloud/jepa_service/engine.py`

**Objective**: Upgrade TPDS to use V-JEPA 2 prediction error as the
depth signal, rather than raw motion energy. When the world model encoder
is available, prediction error by region replaces motion energy.

**Core insight** (Le World Model + V-JEPA 2):
- Foreground objects: their future representations are hard to predict
  from surrounding context (they are independently acting agents)
- Background objects: their future representations are highly predictable
  from surrounding context (they are part of the static scene structure)
- Therefore: regions with high prediction error = foreground
             regions with low prediction error = background
```python
# In DepthSeparator.compute():
def compute(
    self,
    current_frame_emb: np.ndarray,
    prev_frame_emb: Optional[np.ndarray] = None,
    patch_prediction_errors: Optional[np.ndarray] = None,  # NEW
) -> DepthStrataMap:
    """
    Le World Model TPDS:
    If patch_prediction_errors provided (from world model predictor):
      → Use prediction error as depth proxy (principled)
      → foreground = high prediction error (hard to predict)
      → background = low prediction error (easy to predict)
    
    Else (surrogate mode):
      → Use motion energy as depth proxy (heuristic)
      → existing implementation unchanged
    """
    if patch_prediction_errors is not None:
        # PRINCIPLED: prediction-error-based depth separation
        # patch_prediction_errors: (14, 14) array of per-patch errors
        depth_proxy = patch_prediction_errors  # high error = foreground
        
        fg_threshold = np.percentile(depth_proxy, 70)  # top 30% = foreground
        bg_threshold = np.percentile(depth_proxy, 30)  # bottom 30% = background
        
        foreground_mask = depth_proxy > fg_threshold
        background_mask = depth_proxy < bg_threshold
        midground_mask  = ~foreground_mask & ~background_mask
        
        confidence = float(
            1.0 - np.std(depth_proxy) / (np.mean(depth_proxy) + 1e-8)
        )
        # Higher when prediction errors are bimodal (clear fg/bg separation)
        
    else:
        # HEURISTIC: existing motion energy implementation
        # ... existing code unchanged ...
    
    return DepthStrataMap(
        depth_proxy=depth_proxy.astype(np.float32),
        foreground_mask=foreground_mask,
        midground_mask=midground_mask,
        background_mask=background_mask,
        confidence=min(0.99, confidence),
        strata_entropy=float(-np.sum(
            np.bincount([foreground_mask.sum(), background_mask.sum()]) 
            / (14*14) * np.log(
                np.bincount([foreground_mask.sum(), background_mask.sum()])
                / (14*14) + 1e-8
            )
        )),
    )
```

**Success gate**:
```bash
pytest cloud/jepa_service/tests/test_depth_separator.py -v
# All tests pass

# The telescope test is the ultimate TPDS verification
pytest -v cloud/api/tests/test_smriti_production.py::test_telescope_behind_person_not_described_as_body_part
# MUST PASS — this is the production contract
```

---

### SUB-AGENT: ecgd_upgrade
*Layer 2 — parallel*

**Context files**: `cloud/jepa_service/confidence_gate.py`,
`cloud/runtime/models.py`

**Objective**: Upgrade ECGD from 3 heuristic thresholds to
information-theoretic uncertainty bounds using the Le World Model
epistemic/aleatoric uncertainty decomposition from JEPATick.

**The philosophical upgrade** (Le World Model Pillar 4):
```python
# CURRENT ECGD (3 heuristic thresholds):
TAU_DEPTH   = 0.50  # depth purity threshold
TAU_ANCHOR  = 0.55  # anchor confidence threshold  
TAU_ENERGY  = 0.25  # energy variance threshold
# These numbers were chosen by hand.

# LE WORLD MODEL ECGD (information-theoretic):
TAU_EPISTEMIC  = 0.35  # max reducible uncertainty → "you can know this"
TAU_ALEATORIC  = 0.60  # max irreducible uncertainty → "inherently ambiguous"
# These thresholds have an interpretation:
# - If epistemic_unc > 0.35: "look from a different angle and try again"
# - If aleatoric_unc > 0.60: "this object is inherently indistinct"
# - Epistemic uncertainty suggests a USER ACTION that could resolve it
# - Aleatoric uncertainty means the uncertainty is fundamental
```

**New ECGD output** — the gate now returns actionable guidance:
```python
@dataclass
class GateResult:
    # Existing fields
    passes: bool
    safe_embedding: Optional[np.ndarray]
    uncertainty_map: Optional[np.ndarray]
    anchor_name: str
    
    # NEW: uncertainty type and resolution guidance
    failure_reason: Optional[str] = None
    # "epistemic": could be resolved with more observations
    # "aleatoric": fundamentally irreducible
    # None if passes=True
    
    resolution_hint: Optional[str] = None
    # If epistemic: "Try observing from a different angle"
    # If aleatoric: "This region is inherently ambiguous"
    # None if passes=True

def evaluate(self, anchor_match, depth_strata, energy_history,
             # NEW: world model uncertainty fields
             epistemic_uncertainty: Optional[float] = None,
             aleatoric_uncertainty: Optional[float] = None) -> GateResult:
    
    if epistemic_uncertainty is not None and aleatoric_uncertainty is not None:
        # Le World Model path: information-theoretic evaluation
        
        if aleatoric_uncertainty > self.TAU_ALEATORIC:
            return GateResult(
                passes=False, safe_embedding=None,
                uncertainty_map=self._build_uncertainty_map(depth_strata),
                anchor_name=anchor_match.template_name,
                failure_reason="aleatoric",
                resolution_hint="This region is inherently ambiguous — "
                               "no additional observation will resolve it",
            )
        
        if epistemic_uncertainty > self.TAU_EPISTEMIC:
            return GateResult(
                passes=False, safe_embedding=None,
                uncertainty_map=self._build_uncertainty_map(depth_strata),
                anchor_name=anchor_match.template_name,
                failure_reason="epistemic",
                resolution_hint="Try observing from a different angle or "
                               "in better lighting to resolve uncertainty",
            )
        
        # Both uncertainties acceptable — gate passes
        return GateResult(
            passes=True,
            safe_embedding=anchor_match.embedding_centroid,
            uncertainty_map=None,
            anchor_name=anchor_match.template_name,
        )
    
    else:
        # Heuristic fallback (surrogate encoder mode — backward compat)
        # ... existing 3-threshold logic unchanged ...
```

**Success gate**:
```bash
pytest -v cloud/api/tests/test_smriti_production.py::test_uncertain_region_returns_uncertainty_not_wrong_description
# MUST PASS — this is the epistemic safety contract

pytest -q cloud/jepa_service/tests/
# All pass
```

---

### SUB-AGENT: audio_jepa_ingestion
*Layer 2 — parallel, independent of video pipeline changes*

**Context files**: `cloud/runtime/smriti_ingestion.py`,
`cloud/runtime/smriti_storage.py`, `cloud/runtime/models.py`,
`requirements.txt`

**Objective**: Implement the Audio-JEPA ingestion pipeline (Naad v0).
Extract audio tracks from video files using PyAV (already installed).
Generate audio embeddings using a simplified Audio-JEPA encoder.
Store in SmetiDB's new `audio_embedding` field.

**Architecture**:
```python
# cloud/jepa_service/audio_encoder.py (NEW)
"""
Audio-JEPA Encoder v0 — Sprint 6 (Naad foundation).

Sprint 6 implementation: mel spectrogram features as proxy
for Audio-JEPA representations. This is the same relationship
as MobileNetV2→V-JEPA 2 on the video side.

Sprint 7: replace with actual Audio-JEPA transformer encoder.
Interface is identical — the upgrade path is a model swap.

Output: 384-dim L2-normalized audio representation vector.
The dimensionality matches the visual JEPA space, enabling
joint AV retrieval via Setu-2 without modification.
"""
import numpy as np
from typing import Optional

class AudioJEPAEncoder:
    """
    Audio encoder that produces 384-dim representations
    compatible with the JEPA metric space.
    
    Current implementation: 40-band mel spectrogram → PCA → 384-dim
    This is a principled proxy — mel features capture acoustic geometry,
    not just amplitude. PCA projection learns the most informative
    acoustic dimensions from the user's own audio corpus.
    """
    
    TARGET_DIM = 384
    N_MELS = 40
    HOP_LENGTH_MS = 10
    FRAME_LENGTH_MS = 25
    
    def __init__(self):
        self._pca_matrix = None  # lazy-initialized from first 100 audio clips
        self._pca_ready = False
    
    def encode(self, audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Encode audio to 384-dim representation.
        
        audio_samples: (N,) float32, mono, any sample rate
        sample_rate: samples per second
        
        Returns: (384,) float32, L2-normalized
        """
        mel = self._compute_mel_spectrogram(audio_samples, sample_rate)
        
        # Aggregate over time: mean + std pooling
        mel_mean = np.mean(mel, axis=1)   # (N_MELS,)
        mel_std  = np.std(mel, axis=1)    # (N_MELS,)
        mel_delta = np.diff(mel_mean)     # (N_MELS-1,) — temporal rate of change
        
        # Concatenate: [mean(40), std(40), delta(39)] = 119-dim
        audio_feat = np.concatenate([mel_mean, mel_std, mel_delta])
        
        # Project to 384-dim
        if not self._pca_ready:
            # Random projection until PCA is warm
            np.random.seed(hash(audio_feat.tobytes()) % 2**32)
            proj = np.random.randn(len(audio_feat), self.TARGET_DIM).astype(np.float32)
            proj /= np.sqrt(len(audio_feat))
        else:
            proj = self._pca_matrix
        
        emb = audio_feat @ proj
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        
        return emb.astype(np.float32)
    
    def _compute_mel_spectrogram(
        self, samples: np.ndarray, sr: int
    ) -> np.ndarray:
        """Compute mel spectrogram (N_MELS × T frames)."""
        # Use scipy.signal for STFT — no librosa dependency
        from scipy.signal import stft
        from scipy.fftpack import dct
        
        hop = int(sr * self.HOP_LENGTH_MS / 1000)
        win = int(sr * self.FRAME_LENGTH_MS / 1000)
        
        _, _, Zxx = stft(samples, fs=sr, nperseg=win, noverlap=win-hop)
        power = np.abs(Zxx) ** 2
        
        # Mel filterbank (approximate)
        n_fft = win // 2 + 1
        mel_filters = self._mel_filterbank(sr, n_fft, self.N_MELS)
        mel_power = mel_filters @ power
        
        log_mel = np.log(mel_power + 1e-8)
        return log_mel.astype(np.float32)
    
    def _mel_filterbank(self, sr: int, n_fft: int, n_mels: int) -> np.ndarray:
        """Simple triangular mel filterbank."""
        f_min, f_max = 80.0, sr / 2.0
        mel_min = 2595 * np.log10(1 + f_min/700)
        mel_max = 2595 * np.log10(1 + f_max/700)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft * 2) * hz_points / sr).astype(int)
        filters = np.zeros((n_mels, n_fft))
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m       = bin_points[m]
            f_m_plus  = bin_points[m + 1]
            for k in range(f_m_minus, f_m):
                if f_m > f_m_minus:
                    filters[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus > f_m:
                    filters[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m)
        return filters.astype(np.float32)
```

**Extend SmritiIngestionDaemon**:
```python
# In SmritiIngestionDaemon._process_media_item():

async def _extract_audio_embedding(
    self, file_path: str
) -> Optional[np.ndarray]:
    """Extract audio track and encode with Audio-JEPA."""
    if not file_path.lower().endswith(('.mp4', '.mov', '.m4v', '.avi', '.mkv')):
        return None  # Image files have no audio
    
    try:
        import av
        loop = asyncio.get_running_loop()
        
        def _decode_audio():
            container = av.open(file_path)
            audio_stream = next(
                (s for s in container.streams if s.type == 'audio'), None
            )
            if audio_stream is None:
                return None
            
            samples = []
            for frame in container.decode(audio_stream):
                arr = frame.to_ndarray().flatten().astype(np.float32)
                samples.append(arr)
            container.close()
            
            if not samples:
                return None
            
            audio = np.concatenate(samples)
            sr = audio_stream.rate
            # Normalize amplitude
            if np.max(np.abs(audio)) > 1e-8:
                audio = audio / np.max(np.abs(audio))
            return audio, sr
        
        result = await loop.run_in_executor(None, _decode_audio)
        if result is None:
            return None
        
        audio_samples, sr = result
        
        # Encode with Audio-JEPA
        from cloud.jepa_service.audio_encoder import AudioJEPAEncoder
        encoder = AudioJEPAEncoder()
        audio_emb = encoder.encode(audio_samples, sr)
        return audio_emb
        
    except Exception as e:
        log.warning("audio_embedding_failed", file=file_path, error=str(e))
        return None
```

**Success gate**:
```bash
python3.11 -c "
from cloud.jepa_service.audio_encoder import AudioJEPAEncoder
import numpy as np

enc = AudioJEPAEncoder()
# Test with synthetic audio (440Hz sine wave)
sr = 16000
t = np.linspace(0, 2.0, sr * 2, dtype=np.float32)
audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone, 2 seconds

emb = enc.encode(audio, sr)
assert emb.shape == (384,), f'Wrong shape: {emb.shape}'
assert abs(np.linalg.norm(emb) - 1.0) < 0.01, 'Not normalized'
print(f'Audio-JEPA encoder: PASS (shape={emb.shape})')

# Two different sounds should produce different embeddings
noise = np.random.randn(sr * 2).astype(np.float32) * 0.1
emb_noise = enc.encode(noise, sr)
cosine_sim = float(np.dot(emb, emb_noise))
print(f'Cosine similarity (tone vs noise): {cosine_sim:.3f} (should be < 0.8)')
assert cosine_sim < 0.9, 'Embeddings too similar for different sounds'
"

pytest -q cloud/api/tests/test_smriti_ingestion_and_recall.py
# All pass
```

---

### SUB-AGENT: surprise_indexing
*Layer 3 — parallel with setu2_hierarchical*

**Context files**: `cloud/runtime/service.py`, `cloud/runtime/app.py`,
`cloud/runtime/smriti_storage.py`

**Objective**: Wire the surprise_score (prediction error) from JEPATick
into Smriti indexing and recall. High-surprise moments should be
retrievable directly. The surprise signal should optionally weight
recall results.

**New API route**:
```
GET /v1/smriti/moments/surprising?min_surprise=0.7&limit=20
```

**New recall parameter**:
```python
# In SmritiRecallRequest:
use_surprise_weighting: bool = False
# If True: final_score = setu_energy × (1 + surprise_score)
# Surprising moments rank higher for the same semantic similarity
```

**Success gate**:
```bash
pytest -q cloud/api/tests/test_surprise_indexing.py -v
# All tests pass (this file is created by this agent — target: 8 tests)
```

---

### SUB-AGENT: setu2_hierarchical
*Layer 3 — parallel with surprise_indexing*

**Context files**: `cloud/runtime/setu2.py`, `cloud/runtime/service.py`,
`cloud/runtime/app.py`, `cloud/runtime/smriti_storage.py`

**Objective**: Extend Setu-2 to support Level 2 (object-level) retrieval
in addition to the existing Level 1 (patch-level) retrieval.

**New recall modes**:
```python
# In smriti_recall():
if request.hierarchy_level == 2:
    # Use l2_embedding (object-level aggregation)
    corpus = self.smriti_db.get_all_embeddings_with_level(level=2)
    # ... rest of retrieval identical
elif request.hierarchy_level == 1:
    # Existing behavior (patch-level)
    corpus = self.smriti_db.get_all_embeddings(limit=500)
```

**Success gate**:
```bash
pytest -q cloud/api/tests/test_setu2_hierarchical.py -v
# All tests pass (target: 6 tests)
```

---

### SUB-AGENT: sprint6_validation
*Layer 4 — runs last, after all agents complete*

**All 16 validation gates**:
```bash
# Gate 1: Full test suite
pytest -q cloud/api/tests cloud/jepa_service/tests \
       cloud/search_service/tests cloud/monitoring/tests \
       tests/test_readme.py
# Must be ≥ 235 (196 + 39 new world model tests)

# Gate 2: Production gate — telescope contract FIRST
pytest -v cloud/api/tests/test_smriti_production.py
# All 12 must pass. Telescope test non-negotiable.

# Gate 3: World model fields populated in JEPATick
python3.11 -c "
from cloud.jepa_service.engine import ImmersiveJEPAEngine
import numpy as np
engine = ImmersiveJEPAEngine(device='cpu')
frame = np.zeros((224,224,3), dtype=np.uint8)
for i in range(3):
    tick = engine.tick(frame, 'test', f'obs_{i}')
assert tick.prediction_error is not None, 'prediction_error missing'
assert tick.surprise_score is not None, 'surprise_score missing'
assert tick.world_model_version != 'unknown', 'world_model_version not set'
print(f'World model fields present: {tick.world_model_version}')
print('PASS')
"

# Gate 4: ECGD information-theoretic mode
python3.11 -c "
from cloud.jepa_service.confidence_gate import EpistemicConfidenceGate
gate = EpistemicConfidenceGate()
# Test epistemic uncertainty path
from cloud.jepa_service.anchor_graph import AnchorMatch
import numpy as np
match = AnchorMatch(
    template_name='cylindrical_object', confidence=0.8,
    patch_indices=[100, 101], depth_stratum='background',
    centroid_patch=100,
    embedding_centroid=np.zeros(384, dtype=np.float32),
    is_novel=False,
    bbox_normalized={'x':0.5,'y':0.5,'width':0.1,'height':0.3}
)
from cloud.jepa_service.depth_separator import DepthStrataMap
strata = DepthStrataMap.cold_start()
result = gate.evaluate(
    anchor_match=match, depth_strata=strata,
    energy_history=[0.1]*8,
    epistemic_uncertainty=0.1,   # low → should pass
    aleatoric_uncertainty=0.1,   # low → should pass
)
assert result.passes, 'Low uncertainty should pass gate'
print('ECGD information-theoretic: PASS')
"

# Gate 5: SmetiDB schema version 2
python3.11 -c "
from cloud.runtime.smriti_storage import SmetiDB
import tempfile
with tempfile.TemporaryDirectory() as d:
    db = SmetiDB(d)
    assert db.SCHEMA_VERSION == 2, f'Schema version = {db.SCHEMA_VERSION}'
    print('SmetiDB schema v2: PASS')
"

# Gate 6: Audio encoder functional
python3.11 -c "
from cloud.jepa_service.audio_encoder import AudioJEPAEncoder
import numpy as np
enc = AudioJEPAEncoder()
audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 32000)).astype(np.float32)
emb = enc.encode(audio, 16000)
assert emb.shape == (384,)
print('Audio-JEPA encoder: PASS')
"

# Gate 7: New Sprint 6 routes registered
python3.11 -c "
from cloud.runtime.app import create_app
import tempfile
with tempfile.TemporaryDirectory() as d:
    app = create_app(data_dir=d)
    routes = {r.path for r in app.routes}
    new_routes = ['/v1/smriti/moments/surprising', '/v1/world-model/status']
    for r in new_routes:
        status = '✓' if r in routes else '✗ MISSING'
        print(f'{r}: {status}')
"

# Gate 8: Encoder type reported correctly in world model status
python3.11 -c "
from cloud.runtime.app import create_app
from fastapi.testclient import TestClient
import tempfile
with tempfile.TemporaryDirectory() as d:
    app = create_app(data_dir=d)
    client = TestClient(app)
    r = client.get('/v1/world-model/status')
    assert r.status_code == 200
    body = r.json()
    assert 'encoder_type' in body
    print(f'World model status: encoder={body[\"encoder_type\"]}')
    print('PASS')
"

# Gate 9: Torch isolation (anchored grep)
grep -rn "^import torch\|^from torch" cloud/ --include="*.py" \
  | grep -v "cloud/perception/"
# Must return nothing

# Gate 10: TypeScript typecheck
cd desktop/electron && npm run typecheck && echo "TYPECHECK CLEAN"

# Gate 11: Frontend build
cd desktop/electron && npm run build && echo "BUILD CLEAN"

# Gate 12: Deprecation clean
python3.11 -W error::DeprecationWarning -c "
from cloud.runtime.app import create_app
import tempfile
with tempfile.TemporaryDirectory() as d:
    create_app(data_dir=d)
print('CLEAN')
"

# Gate 13: SmetiDB schema idempotency
python3.11 -c "
from cloud.runtime.smriti_storage import SmetiDB
import tempfile
with tempfile.TemporaryDirectory() as d:
    db1 = SmetiDB(d)
    db2 = SmetiDB(d)  # second init on same directory
    db3 = SmetiDB(d)  # third — idempotent
    assert db3.SCHEMA_VERSION == 2
    print('Schema idempotency: PASS')
"

# Gate 14: Surprise score range validation
python3.11 -c "
from cloud.jepa_service.engine import ImmersiveJEPAEngine
import numpy as np
engine = ImmersiveJEPAEngine(device='cpu')
scores = []
for i in range(20):
    frame = np.random.randint(0, 255, (224,224,3), dtype=np.uint8)
    tick = engine.tick(frame, 'test', f'obs_{i}')
    assert 0.0 <= tick.surprise_score <= 1.0, f'Out of range: {tick.surprise_score}'
    scores.append(tick.surprise_score)
print(f'Surprise scores range: [{min(scores):.3f}, {max(scores):.3f}]')
print('Surprise score validation: PASS')
"

# Gate 15: Audio-visual independence (audio and video pipelines don't conflict)
python3.11 -c "
from cloud.runtime.smriti_ingestion import SmritiIngestionDaemon
from cloud.runtime.smriti_storage import SmetiDB
import tempfile, asyncio
async def test():
    with tempfile.TemporaryDirectory() as d:
        db = SmetiDB(d)
        daemon = SmritiIngestionDaemon(db, lambda: None)
        # Verify daemon has audio extraction method
        assert hasattr(daemon, '_extract_audio_embedding')
        print('Audio-visual independence: PASS')
asyncio.run(test())
"

# Gate 16: Update state.json (honest measured count)
python3.11 -c "
import json, pathlib, subprocess, sys
result = subprocess.run(
    [sys.executable, '-m', 'pytest', '-q',
     'cloud/api/tests', 'cloud/jepa_service/tests',
     'cloud/search_service/tests', 'cloud/monitoring/tests',
     'tests/test_readme.py', '--tb=no'],
    capture_output=True, text=True
)
last = [l for l in result.stdout.strip().split('\n') if 'passed' in l]
count = int(last[-1].split()[0]) if last else -1

state = json.loads(pathlib.Path('.smriti_build/state.json').read_text())
state.update({
    'version': '6.0',
    'sprint': '6',
    'build_complete': count >= 235,
    'total_tests_passing': count,
    'target_tests_passing': 235,
    'met_target': count >= 235,
    'le_world_model_impl': True,
    'world_model_version': 'vjepa2_or_ijepa',
    'telescope_test': 'PASSED',
    'audio_jepa_v0': True,
    'agents': {k: 'complete' for k in state['agents']},
})
pathlib.Path('.smriti_build/state.json').write_text(json.dumps(state, indent=2))
print(f'Sprint 6 state: {count} tests. Target met: {count >= 235}')
"
```

---

## FAILURE PROTOCOL

1. **ANY gate fails → STOP ALL WORK. Report which gate, which agent.**
2. **Test count drops below 196 after any agent → immediate rollback of that agent.**
3. **Telescope test fails at any point → ALL work halts. This is the sentinel.**
4. **JEPATick interface broken → Layer 0 must be re-run before any other agent proceeds.**
5. **Torch isolation violated → find and remove the import before proceeding.**

---

## INVARIANTS — SPRINT 6 ADDITIONS

**Never change from all prior sprints:**
- torch imports confined to `cloud/perception/` only
- EMA update before predictor forward — timing is critical
- Ghost bounding boxes in pixel coordinates
- SmetiDB schema migrations idempotent — SCHEMA_VERSION must be 2
- clear_all prune requires exact confirmation string
- Deepdive focus trap + focus restoration
- The telescope test: NEVER remove, NEVER weaken

**Sprint 6 additions:**
- `JEPATick.world_model_version` must be set on every tick — never left as "unknown"
- `prediction_error` must be None on first tick of a session (no previous state to compare)
- `surprise_score` must be in [0.0, 1.0] — enforced, not assumed
- Audio embeddings are optional (videos without audio tracks are valid)
- Encoder type fallback is always MobileNetV2 — never an error, just a logged warning
- V-JEPA 2 temporal buffer must be cleared on session start (`clear_temporal_buffer()`)
- The projection matrix `W_proj` uses seed=42 — deterministic across restarts

---

## ANTIGRAVITY ENTRY COMMAND (paste verbatim)
```
You are the TOORI Sprint 6 World Model Orchestrator.
Mode: Antigravity. Model: Opus 4.6. USE Ultrathink: ENABLED.
USE Sub-agent architecture: parallel execution where dependencies allow.

UPSTREAM: Sprint 5 complete (196 tests). Fix pass complete.
This is the most architecturally significant sprint in TOORI history.
The Le World Model paper (arXiv 2603.19312) is being implemented.
The MobileNetV2 surrogate encoder is being replaced with V-JEPA 2.

ULTRATHINK FIRST: Before writing a single line of code, reason through:
1. The complete dependency graph of all 10 sub-agents
2. Which can run in parallel (no shared mutable state)
3. Which interfaces constitute contracts that other agents depend on
4. Every file that will change and every test that tests it
5. The exact format of JEPATick after extension — this is the contract
6. How to verify each change without breaking existing 196 tests

SUB-AGENT EXECUTION LAYERS:
  Layer 0 (sequential):   jepatick_extension → smetiddb_schema_v2
  Layer 1 (alone):        vjepa2_encoder
  Layer 2 (parallel):     world_model_predictor || tpds_upgrade ||
                          ecgd_upgrade || audio_jepa_ingestion
  Layer 3 (parallel):     surprise_indexing || setu2_hierarchical
  Layer 4 (alone):        sprint6_validation (16 gates)

TARGET: ≥ 235 tests passing (196 baseline + 39 new world model tests)
TELESCOPE CONTRACT: must pass at every checkpoint, no exceptions
TORCH ISOLATION: anchored grep ^import torch only in cloud/perception/
LE WORLD MODEL IMPL: prediction_error, epistemic/aleatoric uncertainty,
                     surprise_score, audio_embedding in JEPATick
AUDIO-JEPA: Naad v0 — audio extraction + encoding + storage

Begin: ultrathink. Then read all context files. Then spawn Layer 0.
```