# Toori User Manual

## 1. What You Are Running

Toori has two practical modes:

- **Browser-first proof mode**
  - recommended for development and daily testing
  - camera permissions behave like a normal web app
  - best current path for the JEPA proof surface
- **Electron packaging mode**
  - intended for a real signed macOS bundle
  - required if you want a proper app identity under Camera privacy

The product surfaces are:

- **Live Lens** for manual capture and debugging
- **Living Lens** for continuous JEPA proof behavior and world-model evidence

## 2. Start The Runtime

```bash
cd /Users/macuser/toori
python3.11 scripts/setup_backend.py
```

This starts the loopback runtime and prepares local observation storage.

Verify it:

```bash
curl http://127.0.0.1:7777/healthz
curl http://127.0.0.1:7777/v1/providers/health
```

## 3. Launch The Proof Surface

Browser mode:

```bash
cd /Users/macuser/toori/desktop/electron
npm install
npm run web
```

Open:

- [http://127.0.0.1:4173](http://127.0.0.1:4173)

Electron shell:

```bash
cd /Users/macuser/toori/desktop/electron
npm start
```

If you need a packaged-app proof path, the Electron app must be built as a real signed macOS bundle. The stock CLI launcher is not enough for stable Camera privacy behavior on macOS.

## 4. Live Lens

Use this when you want to manually capture a frame and inspect the result.

Steps:

1. Open **Live Lens**.
2. Allow camera access in the browser, or approve the packaged app in macOS Camera privacy if you are testing Electron.
3. Pick a camera device.
4. Observe the preview state, resolution, track state, and last frame timestamp.
5. Click **Capture Frame** when you want to create a stored observation.

What the result means:

- the latest observation is the last stored frame
- nearest memory shows the most similar past observations
- provider badges show which backends were used

## 5. Living Lens

This is the proof surface.

It should be used to evaluate:

- prediction consistency
- temporal continuity
- surprise
- persistence through occlusion

What to look for:

- the system should keep a stable entity thread when the object disappears briefly and returns
- surprise should rise when something truly changes
- the world-model view should show predicted state versus observed state
- the persistence graph should show visible, occluded, and re-identified phases
- `Passive mode` means continuous monitoring. The camera stays on, the scene model keeps updating, and you do not need to press capture for every tick.
- If the labels look vague, focus on the predicted/observed panels and the persistence graph rather than the one-line caption.

The browser UI is the easiest place to validate this today. If the app is not packaged as a signed macOS bundle, browser mode is the correct proof path.

## 6. Run A Live Challenge

Use the live camera and follow the guided sequence:

1. Show a distinct object.
2. Partially occlude it.
3. Fully occlude it.
4. Reveal it again.
5. Move the camera away.
6. Return to the same scene.
7. Introduce a different object or rearrange the scene.

This is the clearest way to demonstrate what Toori is claiming about scene continuity and persistence.
The challenge view should show:

- the step you are on
- the current predicted state
- the observed state
- the continuity and surprise signals
- a score comparison against frame captioning and generic retrieval

## 7. Compare Against Baselines

Toori should be evaluated against:

- **Frame captioning**
  - one frame in, one caption out
  - no temporal memory
- **Generic embedding retrieval**
  - similar scene lookup
  - no prediction

The JEPA proof surface is stronger when it can show:

- better continuity across occlusion
- better persistence after reappearance
- clearer surprise on true change

## 8. Configure Settings

Open **Settings** and configure:

- `providers.onnx.model_path`
- `providers.ollama.base_url`
- `providers.ollama.model`
- `providers.mlx.model_path`
- `providers.mlx.metadata.command`
- `providers.cloud.base_url`
- `providers.cloud.model`
- `providers.cloud.api_key`
- theme mode: `dark`, `light`, or `system`
- camera device preference
- decode mode and thresholds

Suggested defaults:

- browser mode for proof work
- `auto` reasoning backend for general use
- `onnx` as the primary desktop perception path
- `ollama` before MLX before cloud for desktop reasoning

## 9. Camera Troubleshooting

If the preview is blank:

- use browser mode first
- confirm the browser permission prompt was granted
- check that the camera device selector is not stuck on a denied Electron identity
- use the runtime health endpoint to verify the backend is up

If Electron does not appear under macOS Camera privacy:

- that means the stock CLI launch still does not have a stable bundle identity
- use browser mode until the app is packaged as a real signed bundle

## 10. Search, Replay, And Integrations

### Memory Search

- search over stored observations
- use it to compare current live scenes against prior scenes

### Session Replay

- inspect the recorded timeline
- review what the model thought, what it stored, and which provider answered

### Integrations

- the runtime exposes a loopback API and SDKs
- other apps should integrate through the runtime, not by scraping the UI

The world-model proof endpoints are:

- `POST /v1/living-lens/tick`
- `GET /v1/world-state`
- `POST /v1/challenges/evaluate`

These are the routes that matter for the JEPA proof surface. `Living Lens` uses them to show prediction consistency, temporal continuity, surprise, and persistence over time.

## 11. How To Read The Proof Signals

The proof surface uses these ideas:

- **Prediction consistency**: how well the next frame matches the latent expectation
- **Temporal continuity**: whether the scene stays coherent over time
- **Surprise**: how strongly the next frame diverges from the prediction
- **Persistence**: whether the same entity or scene thread survives interruptions
- **Stable elements**: scene parts that the model expected and still saw
- **Changed elements**: scene parts that appeared, disappeared, or shifted enough to matter
- **World-model readout**: the short operator-friendly explanation of what the system expects, what it saw, and what stayed the same

If the model says an object is still there after occlusion, and the entity track reconnects after the object returns, that is evidence for persistence.

## 12. What To Expect Today

Toori is already useful as a live observation memory system, but the JEPA proof is the main narrative now.

You should expect:

- real camera capture
- local observation storage
- live memory search
- proof signals in Living Lens
- browser-first reliability
- a readable comparison between JEPA-style continuity and simpler baselines

You should not expect:

- a claim that this is a final research benchmark
- cloud dependence for the proof
- Electron CLI launches to behave like a fully packaged macOS app
