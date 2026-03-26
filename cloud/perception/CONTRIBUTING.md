# Contributing To `cloud/perception`

This package is intentionally lazy-loaded and must stay import-safe when `torch`
or model weights are missing.

Rules for changes:

- Keep torch-specific imports inside `cloud/perception/`.
- Preserve the numpy fallback path and keep it deterministic.
- Maintain the public output contract:
  - `DinoV2Embedding.pooled_embedding` is 128-d.
  - `DinoV2Embedding.patch_tokens` is `(196, 384)`.
  - `DinoV2Embedding.patch_mask` is `(14, 14)`.
  - `SamSegmentation.masks` is shape-correct and stable across repeated calls.
- Prefer additive changes over replacing the fallback logic.
- Add tests for any new output shape or timeout behavior.

The package should remain usable in offline CI without downloading weights.
