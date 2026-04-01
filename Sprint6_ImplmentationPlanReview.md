Sprint 6 — World Model Foundation
The most architecturally significant sprint in TOORI's history. Implements the Le World Model paper concepts: V-JEPA 2 encoder replacement, temporal prediction with surprise scoring, information-theoretic uncertainty decomposition, Audio-JEPA v0, and hierarchical recall.

Baseline: 196 tests passing | Target: ≥ 235 tests (196 + 39 new)

User Review Required
IMPORTANT

The existing JEPATick is a Python @dataclass (not a Pydantic BaseModel as the Sprint 6 prompt assumes). All new fields will be added as Optional dataclass fields with None defaults. The to_payload() method and JEPATickPayload Pydantic model will both be extended to serialize the new fields. This preserves backward compatibility.

IMPORTANT

No V-JEPA 2 or I-JEPA ONNX weights were found locally. The encoder will load with the existing MobileNetV2 surrogate fallback path from the DINOv2/SAM perception pipeline. The JEPAEncoder class is being added as a new file cloud/perception/onnx_model.py that will gracefully fallback — the existing PerceptionPipeline is preserved unchanged.

WARNING

The prompt specifies onnx_model.py replacement — but the current encoder is dinov2_encoder.py + sam_segmenter.py. We will add onnx_model.py as a new file alongside the existing pipeline. The ImmersiveJEPAEngine continues to use PerceptionPipeline for DINOv2+SAM perception, and additionally instantiates JEPAEncoder for world model embeddings.

Proposed Changes
Layer 0 — Interface Contracts (Sequential)
Sub-Agent: jepatick_extension
[MODIFY] 
models.py
Add 9 new Optional fields to JEPATick dataclass (all defaulting to None/"surrogate")
Extend to_payload() to serialize new fields
Extend JEPATickPayload Pydantic model with matching fields
Add new Pydantic models: WorldModelStatus, HierarchicalRecallRequest
Fields added to JEPATick:

Field	Type	Default	Purpose
l2_embedding	Optional[list[float]]	None	Object-level aggregation
predicted_next_embedding	Optional[list[float]]	None	s_{t+1} prediction
prediction_error	Optional[float]	None	E(predicted, actual)
epistemic_uncertainty	Optional[float]	None	Reducible uncertainty
aleatoric_uncertainty	Optional[float]	None	Irreducible uncertainty
surprise_score	Optional[float]	None	Normalized prediction error
audio_embedding	Optional[list[float]]	None	Audio-JEPA representation
audio_energy	Optional[float]	None	Audio energy score
world_model_version	str	"surrogate"	Encoder version label
Sub-Agent: smetiddb_schema_v2
[MODIFY] 
smriti_storage.py
Bump SCHEMA_VERSION from 1 to 2
Add migration (2, ...) to MIGRATIONS list with idempotent ALTER TABLE ADD COLUMN + new smriti_temporal_index table
New columns: l2_embedding, prediction_error, epistemic_uncertainty, aleatoric_uncertainty, surprise_score, audio_embedding, world_model_version
Add 3 new methods: get_smriti_media_by_surprise(), get_all_embeddings_with_level(), update_world_model_fields()
Layer 1 — Encoder Replacement
Sub-Agent: vjepa2_encoder
[NEW] 
onnx_model.py
New JEPAEncoder class supporting V-JEPA 2, I-JEPA, and MobileNetV2 fallback
encode(frame) → np.ndarray(384,) — L2-normalized, deterministic projection
encode_with_patches(frame) → (global_emb, patch_embs)
Graceful fallback chain: vjepa2 → ijepa → mobilenetv2 → numpy-only surrogate
Uses onnxruntime when models are available, falls back to numpy random projection
[NEW] 
download_vjepa2_models.py
Download script for V-JEPA 2 / I-JEPA weights (optional)
Layer 2 — World Model Components (Parallel)
Sub-Agent: world_model_predictor
[MODIFY] 
engine.py
Add _prev_predictions, _surprise_windows dicts to ImmersiveJEPAEngine.__init__
Enrich tick() to compute: prediction_error, surprise_score, epistemic_uncertainty, aleatoric_uncertainty, predicted_next_embedding, world_model_version
Add _normalize_surprise() — rolling window z-score normalization
Add _compute_uncertainty() — predictor/target variance proxy
Set all new JEPATick fields at construction
Sub-Agent: tpds_upgrade
[MODIFY] 
depth_separator.py
Add patch_prediction_errors: Optional[np.ndarray] = None to update() signature
When patch_prediction_errors is provided: use prediction-error-based depth proxy (principled)
When None: existing motion energy heuristic (unchanged)
Sub-Agent: ecgd_upgrade
[MODIFY] 
confidence_gate.py
Add epistemic_uncertainty and aleatoric_uncertainty optional params to evaluate()
Add TAU_EPISTEMIC = 0.35 and TAU_ALEATORIC = 0.60 thresholds
Add failure_reason and resolution_hint fields to GateResult
When world model uncertainties provided: use information-theoretic path
Else: existing 3-threshold heuristic (unchanged)
Sub-Agent: audio_jepa_ingestion
[NEW] 
audio_encoder.py
AudioJEPAEncoder — mel spectrogram → 384-dim L2-normalized embedding
Uses scipy.signal.stft — no librosa dependency
Deterministic random projection until PCA matrix is warm
[MODIFY] 
smriti_ingestion.py
Add _extract_audio_embedding() method to SmritiIngestionDaemon
Extract audio tracks from video files using PyAV
Encode with AudioJEPAEncoder
Layer 3 — Application Layer (Parallel)
Sub-Agent: surprise_indexing
[MODIFY] 
app.py
Add GET /v1/smriti/moments/surprising route
Add GET /v1/world-model/status route
Import new models (WorldModelStatus)
[MODIFY] 
service.py
Add get_world_model_status() method
Add get_surprising_moments() method
Wire surprise_score into recall weighting when use_surprise_weighting=True
[MODIFY] 
models.py
Add use_surprise_weighting and min_surprise to SmritiRecallRequest
[NEW] 
test_surprise_indexing.py
8 tests: surprise route, min_surprise filter, weighting, empty results, etc.
Sub-Agent: setu2_hierarchical
[MODIFY] 
setu2.py
No structural changes needed — Setu-2 bridge processes embeddings agnostically
[MODIFY] 
service.py
Support hierarchy_level parameter in recall, routing to get_all_embeddings_with_level()
[NEW] 
test_setu2_hierarchical.py
6 tests: level 1 recall, level 2 recall, fallback when no l2 embeddings, etc.
Layer 4 — Validation
[NEW] 
test_world_model_foundation.py
~25 tests covering: JEPATick extension backward compat, world model fields populated, encoder load, audio encoder, surprise scoring, ECGD info-theoretic path, TPDS prediction-error path, schema v2
Open Questions
IMPORTANT

Encoder strategy: The Sprint 6 prompt references ONNX-based V-JEPA 2 weights which aren't available. The JEPAEncoder will use a numpy-only surrogate (random projection) as the encoder, matching the existing system's MobileNetV2 surrogate approach. The interface is identical — when real weights become available, it's a model file swap. Is this acceptable?

IMPORTANT

Test count: The prompt targets 235 tests (39 new). Given the 196 baseline and the scope of new test files, I'll aim for ~40 new tests to meet the target. The readme test failure (197th test) is pre-existing and unrelated — should I fix it?

Verification Plan
Automated Tests
pytest -q cloud/api/tests cloud/jepa_service/tests cloud/search_service/tests cloud/monitoring/tests tests/test_readme.py — must be ≥ 235
Telescope test: pytest -v cloud/api/tests/test_smriti_production.py::test_telescope_behind_person_not_described_as_body_part
Torch isolation: grep -rn "^import torch\|^from torch" cloud/ --include="*.py" | grep -v "cloud/perception/"
TypeScript: cd desktop/electron && npm run typecheck
Schema idempotency: Create SmetiDB 3 times on same directory
Manual Verification
Sprint 6 validation gates 1-16 from the prompt