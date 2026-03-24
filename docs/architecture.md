# System Architecture

Toori is organized around a loopback-first runtime and three client surfaces.

## Components

- **Runtime API**: FastAPI app on `127.0.0.1:7777` handling `analyze`, `query`, `settings`, `provider health`, `observations`, and event streaming.
- **Observation Store**: SQLite plus file-backed thumbnails/images in `.toori/`.
- **Provider Registry**: selects local perception and optional reasoning providers with circuit-breaker fallback.
- **Desktop Client**: Electron + React operator shell with live lens, replay, search, integrations, and settings.
- **iOS Client**: SwiftUI source tree using AVFoundation and the shared runtime contract.
- **Android Client**: Jetpack Compose source tree using CameraX and the shared runtime contract.
- **SDK Layer**: lightweight Python, TypeScript, Swift, and Kotlin clients for plugin usage.

## Runtime Flow

1. Capture a real frame.
2. Compute a local embedding through the best available perception provider.
3. Persist the observation, thumbnail, metadata, and embedding.
4. Search prior observations locally.
5. Optionally invoke `ollama`, MLX, or cloud reasoning to produce a natural-language answer.
6. Stream the result to the active client and any plugin subscribers.
