# Contributing to Toori

## Contribution Surfaces

### 1. Perception Backbone (PRIMARY)

`cloud/perception/` is the main contribution surface.

- Add new encoders implementing the perception contract.
- Current desktop/runtime backbone: DINOv2-small.
- Community targets: SigLIP, V-JEPA 2.1 weights, and locale-specific language-grounded backbones.
- See [cloud/perception/CONTRIBUTING.md](/Users/macuser/toori/cloud/perception/CONTRIBUTING.md).

### 2. Predictor Architecture

`cloud/jepa_service/engine.py` owns `ImmersiveJEPAEngine` and the `f_theta` predictor path.

- Current predictor: 4-layer MLP.
- Community targets: transformer predictors, recurrent predictors, diffusion-style predictors.
- Stable interface: predictor changes must preserve `tick(frame) -> JEPATick`.

### 3. Consumer Mode Translations

`desktop/electron/src/components/ConsumerMode.tsx` is the consumer-facing translation surface.

- Add language files under `desktop/electron/src/i18n/` when the app formalizes localized bundles.
- Priority languages: Bengali, Hindi, Tamil, Swahili, Arabic.
- See [docs/contributing/ui-translations.md](/Users/macuser/toori/docs/contributing/ui-translations.md).

### 4. SDK Extensions

`sdk/` contains Python, TypeScript, Swift, and Kotlin clients.

- `sdk/INTERFACE.md` is expected to stay stable once added.
- Internal client implementation can evolve as long as public method signatures remain compatible.

## What Must Never Change

These are stable interfaces unless the change is explicitly versioned and documented:

- `/v1/events` websocket event schema is additive-only.
- `/v1/living-lens/tick` request/response schema remains stable.
- `JEPATick` public field names in [cloud/runtime/models.py](/Users/macuser/toori/cloud/runtime/models.py) remain stable.
- SDK public method signatures remain stable.

## License

- Engine and SDK code: Apache 2.0
- UI proof surface and translation-facing presentation assets: CC-BY-SA 4.0 where noted
