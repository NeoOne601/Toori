# Perception Backends

## PercEncoder Contract

New backbones plug into `cloud/perception/` through the same minimal contract:

- `encode(frame) -> patch_tokens`
- Output shape must be `(N, d)` with the current desktop/runtime expectation of `N=196`
- Returned arrays must be `float32`
- Retrieval compatibility must still produce a pooled 128-d embedding for observation storage and search

## Registration

Register new backbones in `PerceptionPipeline` inside [cloud/perception/__init__.py](/Users/macuser/toori/cloud/perception/__init__.py).

- Keep all heavy framework imports confined to `cloud/perception/`
- Preserve deterministic fallback behavior when weights are unavailable
- Do not import `torch` into `cloud/jepa_service/engine.py`

## Benchmark Requirements

Every new backbone proposal should include:

- output shape verification
- latency numbers on the target hardware path
- fallback behavior when weights are missing
- notes on mask compatibility if the backbone changes patch layout

## Current Default

- Desktop/runtime backbone: DINOv2-small
- Segmentation companion: MobileSAM with timeout fallback to random patch masking
