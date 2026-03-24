# Hybrid Edge‑Cloud JEPA Design

## Architecture Overview
```
+-------------------+      +--------------------+      +-------------------+
|   iMac M1 Client  | ---> | FastAPI API GW     | ---> |  JEPA Service     |
| (Electron UI)    |      | (auth, /embed,    |      | (TorchServe stub)|
|                  |      |  /search)         |      +-------------------+
+-------------------+      +--------------------+               |
                                 |                         |
                                 v                         v
                         +-------------------+      +-------------------+
                         | Search Service    |      |  Prometheus      |
                         | (FAISS index)    |      |  (metrics)       |
                         +-------------------+      +-------------------+
```

* **Client** – Electron app captures an image, posts to `/embed`, receives a 128‑dim zero‑vector placeholder, then posts to `/search` and displays results.
* **API Gateway** – FastAPI service authenticates via JWT (Auth0 JWKS), forwards embed requests to JEPA and search requests to the FAISS service, and aggregates responses.
* **JEPA Service** – Docker‑containerized FastAPI that loads a placeholder model (`_load_model`) returning a deterministic 128‑dim zero vector. In production it will be replaced with a TorchServe‑hosted JEPA model.
* **Search Service** – FastAPI wrapper around a FAISS‑IVF/PQ index (currently `None`). Future implementation will load a persisted index and return top‑k IDs.
* **Observability** – Each service runs its own `CollectorRegistry` to expose `process_cpu_seconds_total` and can be extended with latency histograms. Prometheus scrapes both services; Grafana visualises metrics.

## Component Breakdown
| Component | Language / Runtime | Responsibilities |
|-----------|-------------------|-------------------|
| **Electron UI** | JavaScript (Node/Electron) | Camera capture, calls `/embed` → `/search`, displays IDs |
| **API Gateway** | Python FastAPI | JWT validation, request orchestration, error handling |
| **JEPA Service** | Python FastAPI (Docker) | Loads JEPA model, provides `/embed` endpoint |
| **Search Service** | Python FastAPI (Docker) | Loads FAISS index, provides `/search` endpoint |
| **Prometheus Exporter** | `prometheus_client` lib (Python) | Exposes per‑service gauges, histogram placeholders |
| **CI/CD Pipeline** | GitHub Actions + Fastlane (mobile) | Build Docker images, run tests, push to registry |
| **Docker Runtime** | Docker Engine | Isolates services; each runs on a cloud VM (Ubuntu) |
| **Auth0** | Managed SaaS | Issues JWTs, JWKS endpoint for verification |

## Data Flow (Step‑by‑Step)
1. **User presses “Capture**” → Electron obtains image (base64).
2. **POST `/embed`** → API GW extracts JWT, forwards payload to JEPA Service.
3. **JEPA Service** → Runs model (currently returns `[0.0]*128`) and returns embedding.
4. **API GW** → Receives embedding, immediately POSTs to `/search` with same JWT.
5. **Search Service** → Queries FAISS index (placeholder returns `[]`) and replies with list of IDs.
6. **API GW** → Merges embedding & search results into JSON response.
7. **Electron UI** → Displays IDs (or “no results”).

*End‑to‑end latency target:* ≤ 1 second (measured from step 1 to step 7).

## Security Design
* **Authentication** – All client requests must include a bearer JWT. The API GW validates signatures against the Auth0 JWKS endpoint.
* **Authorization** – Scope‑based checks can be added later; current design only requires a valid token.
* **Transport** – Services are intended to be exposed behind an HTTPS‑terminating reverse proxy (e.g., Caddy or Nginx).
* **Secret Management** – `.env.sample` ships with placeholders (`AUTH0_DOMAIN`, `API_SECRET`). Real secrets are injected via the CI pipeline (GitHub Actions secrets).
* **Least Privilege** – Each Docker container runs as a non‑root user; only the API GW has network access to the other services.

## Performance & Scalability
* **Target latency**: ≤ 1 s (already met by stubs).
* **Horizontal scaling** – Services are stateless; additional replicas can be launched behind a load balancer if load increases.
* **Resource sizing** – JEPA Service will require a GPU‑enabled VM when the real model is used; Search Service runs on CPU.
* **Caching** – Future task includes an on‑device FAISS‑lite cache to reduce round‑trips for recent queries.

## Deployment & Operations
* **Docker images** – Built via GitHub Actions, pushed to a private registry.
* **Infrastructure** – Single cloud VM (Ubuntu) runs Docker Compose with three services (api‑gw, jepa, search).
* **Prometheus** – Scrapes `/metrics` on each service (unique `CollectorRegistry`).
* **Grafana** – Dashboard (`grafana_dashboard.json`) displays CPU seconds and can be extended with latency histograms.
* **CI/CD** – On each push, actions build images, run unit/integration tests, and push to the registry. A manual approval step is required for production deployments.

## Observability & Monitoring
* **Metrics per service** – `process_cpu_seconds_total` (existing). Planned extensions: request latency histogram, error counters.
* **Health checks** – `/healthz` endpoints can be added later for Kubernetes readiness/liveness (not needed for a single‑VM deployment now).
* **Logging** – Stdout captured by Docker; can be routed to a centralized log service later.

## Testing Strategy
* **Unit tests** – Pure functions, metric initialization.
* **Integration tests** – FastAPI test client with dependency overrides for auth and model stubs.
* **Contract tests** – Validate JSON schema of `/embed` and `/search` responses.
* **Load tests** – Planned using Locust/K6 to verify the 1‑second SLA under concurrent traffic.
* **Security tests** – Verify 401 on missing/invalid JWT, token expiration handling.

---

*Design approved by the user.*

**Date:** 2026‑03‑24

---

## 1. High‑Level Architecture

```
+-------------------+          HTTPS API          +---------------------------+
|   Mobile App      | <-------------------------> |   Cloud Service (GPU)    |
|   (iOS/Android)   |   Embedding + Requests     |  - JEPA‑based encoder    |
|                     |                              |  - FAISS‑IVF/PQ index   |
|  • UI layer       |                              |  - Prediction endpoint   |
|  • Light encoder   |                              |  - Search endpoint       |
|    (MobileViT‑XS) |                              |  - Auth & rate‑limit    |
+-------------------+                              +---------------------------+
        ^   |
        |   | Local inference (on‑device)
        |   v
+-------------------+
|   On‑Device Cache |
|   (FAISS‑lite)    |
|   • Store recent |
|     embeddings    |
+-------------------+
```

**Key Points**
1. **On‑Device Encoder** – MobileViT‑XS runs via Core ML (iOS) and NNAPI (Android) to produce a 128‑dim embedding.
2. **Edge‑Cloud Transfer** – The embedding (plus minimal metadata) is sent over TLS 1.3 to the cloud API.
3. **Cloud JEPA Model** – A larger Llama‑3.2‑JEPA model refines the embedding, performs prediction, and queries a FAISS‑IVF/PQ index containing millions of reference vectors.
4. **Result Return** – Top‑k nearest items plus confidence scores are returned and displayed.
5. **Offline Fallback** – If the network is unavailable, the app queries the on‑device FAISS‑lite cache (limited to recent items).

---

## 2. Component Breakdown

| Component | Responsibility | Technology / Library | Notes |
|-----------|----------------|-----------------------|-------|
| **Mobile App (iOS & Android)** | UI, capture media, invoke encoder, manage cache, display results; supports i18n, accessibility (VoiceOver, TalkBack); minimum iOS 14 / Android 8 (API 26) with NEON support for NNAPI acceleration | SwiftUI + Combine (iOS) / Jetpack Compose + Kotlin Coroutines (Android) – potentially shared via Kotlin Multiplatform (KMM) | Reuses business logic across platforms and meets platform accessibility guidelines |
| **On‑Device Encoder** | Convert image/video → 128‑dim embedding | MobileViT‑XS → Core ML model (iOS) & TensorFlow Lite / NNAPI model (Android) | Model ≤ 10 MB, inference < 30 ms on modern phones |
| **Local FAISS‑lite Cache** | Store recent embeddings for offline similarity search (≈10 k vectors) | FAISS‑lite compiled for Swift/Java via native bridge (arm64 & x86_64 binaries packaged via SwiftPM/CocoaPods) | LRU eviction (max 10 k entries, TTL 48 h), persisted encrypted in app sandbox |
| **HTTPS API Gateway** | Auth, rate‑limit, request routing, TLS termination; deployed in multiple regions with health‑check load‑balancing for high availability | FastAPI + uvicorn, optionally behind AWS API Gateway or GCP Cloud Run with regional failover | JWT‑based auth (Bearer token) |
| **Auth Service** | Issue short‑lived JWTs, validate client IDs | OAuth 2.0 (client‑credentials) via Auth0 or Firebase Auth | Tokens signed RS256, 5 min expiry |
| **Cloud JEPA Encoder** | Refine on‑device embedding, run predictive tasks; versioned with semantic tags (e.g., v1.2.0) and backward‑compatible API | Llama‑3.2‑JEPA (PyTorch) on NVIDIA A100 GPUs | Loaded via TorchServe for low‑latency inference, supports rolling upgrades via canary deployment |
| **FAISS‑IVF/PQ Index** | Large‑scale nearest‑neighbor search (10 M+ vectors) | FAISS (IVF‑Flat + Product Quantization) on GPU | Rebuilt nightly from source database |
| **Search / Prediction Service** | Receive embedding, query index, return top‑k results, optional classification | FastAPI endpoint `POST /search` | JSON response with results and metadata |
| **Monitoring & Logging** | Observe latency, error rates, health, usage metrics (requests per device, query distribution) | Prometheus + Grafana, ELK stack, OpenTelemetry | Alerts on > 200 ms latency, > 1 % error, anomalous usage spikes |
| **CI/CD Pipelines** | Build, test, and deploy mobile and cloud artifacts | GitHub Actions → Fastlane (mobile) & Docker + Argo CD (cloud) | Automated version bump, changelog generation |

---

## 3. Data Flow (Request / Response Sequence)
1. **Capture** – User selects or records an image/video (≤ 5 s).
2. **Pre‑process** – Resize to 224 × 224, normalize pixels.
3. **Local Encoding** – MobileViT‑XS produces embedding `e_local` (128‑dim).
4. **Cache Insert** – `e_local` stored in on‑device FAISS‑lite (key = image hash).
5. **Network Check** – If online, the app strips any PII or location metadata from the payload and sends JSON `{embedding: base64(e_local), metadata}` over HTTPS to `POST /search`.
6. **Auth** – Request includes `Authorization: Bearer <jwt>`; if missing/expired the app fetches a fresh token from the Auth Service.
7. **Cloud JEPA Refinement** – Server passes `e_local` through JEPA, yielding `e_refined`.
8. **Similarity Search** – `e_refined` queries the FAISS‑IVF/PQ index → top‑k nearest IDs.
9. **Result Assembly** – For each ID, fetch title, thumbnail URL, confidence score.
10. **Response** – JSON `{results:[{id, score, title, thumb_url}], timestamp}` returned.
11. **Display** – UI shows scrollable list; tapping opens detail view.
12. **Offline Fallback** – If step 5 fails, app queries the local FAISS‑lite cache and returns those results with an “offline” badge.

**Latency Targets**
- On‑device encode: ≤ 30 ms
- Network RTT (4G/5G): 50‑150 ms
- Cloud JEPA + FAISS query: ≤ 100 ms (GPU)
- End‑to‑end (online): **≈ 200‑300 ms**

---

## 4. Security & Privacy
| Aspect | Measure |
|--------|---------|
| **Transport** | TLS 1.3 with strong cipher suites; certificates via Let’s Encrypt or cloud provider |
| **Authentication** | JWT access tokens (5 min expiry) issued via OAuth 2.0 client‑credentials; signed RS256; public key via JWKS |
| **Data Minimisation** | Only the 128‑dim embedding (≈ 500 bytes) is transmitted; raw image never leaves the device |
| **At‑Rest Encryption** | Cloud storage (S3/GCS) encrypted SSE‑AES256; mobile caches encrypted with iOS Keychain / Android Keystore |
| **GDPR / CCPA** | `DELETE /user/{id}/embeddings` endpoint allows users to erase their data; pseudonymous identifiers only |
| **Rate Limiting** | API gateway enforces adaptive limits (default 200 requests/min per device, burst up to 500) with token‑bucket algorithm |
| **Logging** | No raw embeddings logged; logs contain request IDs and timestamps only |
| **Audit** | Regular security scans (OWASP ZAP) and dependency checks (Dependabot) |

---

## 5. Testing Strategy
| Level | Goal | Tools |
|-------|------|-------|
| **Unit** | Validate encoder shape, cache ops, token handling | XCTest (iOS), JUnit (Android), pytest (Python) |
| **Integration** | End‑to‑end request/response, token refresh, offline fallback | Postman/Newman, pytest‑asyncio for API, Espresso / XCUITest |
| **Performance** | Measure latency, GPU utilization, memory footprint | locust (load testing), Firebase Performance Monitoring, Instruments (iOS) |
| **Security** | Verify TLS, JWT validation, rate‑limit enforcement | OWASP ZAP, jwt‑cli |
| **Regression** | Snapshot UI tests, full suite on every PR | GitHub Actions, Detox for cross‑platform UI |
| **Chaos** | Simulate network loss, token expiry, GPU overload | `tc` traffic control, chaos‑monkey scripts |

**Pass Criteria** – 95th‑percentile latency ≤ 200 ms, error rate < 1 %, unit test coverage ≥ 100 % for core modules.

---

## 6. Deployment & Operations
| Stage | Artifacts | Process |
|-------|-----------|----------|
| **Mobile Build** | `.ipa` & `.aab` bundles | Fastlane lane `beta` → TestFlight / Google Play Internal Track; version bump via semantic‑release |
| **Container Image** | `registry.example.com/jepa-search:<sha>` | Dockerfile built, scanned by Trivy, pushed to Artifact Registry |
| **Kubernetes** | Deployment `jepa-search`, Service (LoadBalancer) | Helm chart with autoscaling (CPU target 70 %), pod‑disruption‑budget |
| **CI/CD** | GitHub Actions workflow | `lint → unit → integration → containerize → helm upgrade` |
| **Monitoring** | Prometheus alerts, Grafana dashboards, Loki logs | Alert on latency > 250 ms, error_rate > 2 %, CPU > 80 % for > 5 min |
| **Rollback** | `helm rollback` to previous release, mobile OTA fallback version | Automated rollback if health checks fail post‑deployment |
| **Documentation** | OpenAPI spec, SDK clients (Kotlin & Swift) | Generated via swagger‑codegen, version‑controlled under `docs/` |

---

**Next Steps**
1. Commit this spec to `docs/superpowers/specs/2026-03-24-hybrid-edge-cloud-design.md`.
2. Run the `spec-document-reviewer` sub‑agent to validate completeness and consistency.
3. After reviewer approval, I will ask you to review the committed file before moving to the implementation‑plan phase.

*Please confirm that the spec looks correct or let me know any modifications you’d like before I commit it.*