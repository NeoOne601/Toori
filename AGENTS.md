# AGENTS.md

This file provides guidance to Codex when working in this repository.

## Overview

Toori is now an executable cross-platform project built around a loopback-first Python runtime and three client surfaces:

- Electron desktop client in `desktop/electron`
- SwiftUI iOS client sources in `mobile/ios`
- Jetpack Compose Android client sources in `mobile/android`

The runtime stores real observations from images or camera frames, computes local descriptors or embeddings, supports optional reasoning backends, and exposes a stable API for first-party clients and plugin consumers.

## Development Commands

### Runtime

- Start the runtime:
  `TOORI_DATA_DIR=.toori bash scripts/run_runtime.sh`
- Run the verified backend test suite:
  `pytest -q cloud/api/tests cloud/jepa_service/tests cloud/search_service/tests cloud/monitoring/tests tests/test_readme.py`

### Desktop

- Install dependencies:
  `cd desktop/electron && npm install`
- Type-check:
  `cd desktop/electron && npm run typecheck`
- Build:
  `cd desktop/electron && npm run build`
- Launch:
  `cd desktop/electron && npm start`

### iOS

- Open the Xcode project:
  `mobile/ios/TooriLens.xcodeproj`
- CLI simulator build:
  `xcodebuild -project mobile/ios/TooriLens.xcodeproj -scheme TooriLens -configuration Debug -sdk iphonesimulator -derivedDataPath .xcode-derived CODE_SIGNING_ALLOWED=NO build`

### Android

- Open the project root in Android Studio:
  `mobile/android`
- The repo includes Gradle build files, but wrapper/bootstrap verification still depends on local Android Studio or a Gradle installation.

## Architecture & Structure

### Major Components

- `cloud/runtime`
  Settings, schemas, provider registry, observation storage, search, event streaming, and runtime orchestration.
- `cloud/api/main.py`
  Main FastAPI entrypoint for the runtime on `127.0.0.1:7777`.
- `cloud/jepa_service/app.py`
  Compatibility perception service over the shared runtime.
- `cloud/search_service/main.py`
  Compatibility search service over the shared runtime.
- `desktop/electron`
  Electron shell plus React/Vite operator UI.
- `mobile/ios/TooriApp`
  SwiftUI client, camera capture, runtime client, and settings flow.
- `mobile/android/app/src/main/java/com/toori/app`
  Jetpack Compose client, runtime client, and settings flow.
- `sdk`
  Python, TypeScript, Swift, and Kotlin client SDKs for plugin use.
- `docs`
  System design, user manual, plugin guide, and architecture material.

### Primary Data Flow

1. A client captures a real image or frame.
2. The runtime analyzes it with a primary local perception provider and stores an `Observation`.
3. Local search ranks prior observations from real stored history.
4. Optional reasoning backends add richer language output.
5. Results are returned through HTTP and streamed over WebSocket events.

### Key Files

- `cloud/runtime/models.py`
  Canonical runtime contracts.
- `cloud/runtime/providers.py`
  Perception and reasoning provider selection.
- `cloud/runtime/service.py`
  Main analyze, query, settings, and provider health behavior.
- `desktop/electron/src/App.tsx`
  Desktop product UI.
- `README.md`
  Quickstart and operator-facing overview.
- `CLAUDE.md`
  Detailed implementation guidance for future agents.

## Additional Guidance

- Use `apply_patch` for file edits.
- Prefer `rg` and `rg --files` for search.
- Do not reintroduce placeholder or mock user-facing results. Search results must map to real stored observations.
- Keep ONNX, CoreML, and TFLite-compatible perception as the primary local path. `ollama` and MLX remain optional, desktop-only, and health-checked.
- Launch the runtime only with Python 3.11. The JEPA stack depends on native extensions that are validated on 3.11 and can crash on newer interpreters.
- Keep generated artifacts out of source control. Runtime state belongs in `.toori/`; build output belongs in platform-specific output directories.
- Update `README.md`, `CLAUDE.md`, and the docs in `docs/` when the runtime contract or user workflow changes.
