# UI Translations

## Consumer Mode Translation Surface

Consumer Mode is the plain-language entry point for the proof surface.

Add translation work around stable keys for:

- room stability copy
- unexpected change copy
- occlusion start/end copy
- entity appearance/disappearance copy
- proof export labels

## Template

Each locale should preserve placeholders exactly:

- `consumer.scene_stable`
- `consumer.shift_nearby`
- `consumer.unexpected_change`
- `consumer.occlusion_start`
- `consumer.occlusion_end`
- `consumer.entity_appeared`
- `consumer.entity_disappeared`
- `consumer.proof_export`

## Review Process

- confirm the scientific meaning is preserved
- keep product terms like `Living Lens` and `Consumer Mode` consistent
- do not translate endpoint paths, model ids, or metric names unless the UI already presents a localized alias
- confirm layout still works in the Electron surface before merging
