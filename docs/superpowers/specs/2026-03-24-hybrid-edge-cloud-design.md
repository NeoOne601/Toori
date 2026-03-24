# Hybrid Edge‑Cloud JEPA Design

**Date:** 2026‑03‑24

---

## 1. High‑Level Architecture

```
+-------------------+          HTTPS API          +---------------------------+
|   Mobile App      | <-------------------------> |   Cloud Service (GPU)    |
| (iOS / Android)  |   Embedding + Requests      |  - JEPA‑based encoder    |
|                  |                              |  - FAISS‑IVF/PQ index   |
| • UI layer       |                              |  - Prediction endpoint   |
| • Light encoder  |                              |  - Search endpoint       |
|   (MobileViT‑XS) |                              |  - Auth & rate‑limit    |
+-------------------+                              +---------------------------+
        ^   |
        |   | Local inference (on‑device)
        |   v
+-------------------+
| On‑Device Cache   |
| (FAISS‑lite)      |
| • Store recent    |
|   embeddings      |
+-------------------+
```

*The system consists of a mobile client with an on‑device encoder and optional cache, which communicates via a TLS‑secured API to a cloud service hosting the JEPA model and a large FAISS index.*

---

## 2. Component Breakdown

| Component | Responsibility | Technology / Library | Notes |
|-----------|----------------|----------------------|-------|
| **Mobile App (iOS & Android)** | UI, media capture, invoke encoder, manage cache, display results; supports i18n & accessibility | SwiftUI + Combine (iOS) / Jetpack Compose + Kotlin Coroutines (Android) – can be shared via Kotlin Multiplatform (KMM) | Minimum iOS 14 / Android 8 (API 26) with NEON support for NNAPI acceleration |
| **On‑Device Encoder** | Convert image/video → 128‑dim embedding | MobileViT‑XS model → Core ML (iOS) & TensorFlow Lite / NNAPI (Android) | ≤ 10 MB, inference < 30 ms on modern phones |
| **On‑Device FAISS‑lite Cache** | Store recent embeddings for offline similarity search (≈10 k vectors) | FAISS‑lite compiled for Swift/Java via native bridge (arm64 & x86_64) | LRU eviction, TTL 48 h, encrypted in app sandbox |
| **HTTPS API Gateway** | Authenticate requests, rate‑limit, route to JEPA & search services, TLS termination | FastAPI + uvicorn, optional deployment behind AWS API Gateway or GCP Cloud Run | JWT‑based auth (Bearer token) |
| **Auth Service** | Issue short‑lived JWTs, validate client IDs | OAuth 2.0 client‑credentials via Auth0 or Firebase Auth | Tokens signed RS256, 5 min expiry |
| **Cloud JEPA Service** | Refine on‑device embedding, run predictive tasks | Llama‑3.2‑JEPA (PyTorch) on NVIDIA A100 GPUs, served via TorchServe | Versioned with semantic tags, supports rolling upgrades |
| **FAISS‑IVF/PQ Index** | Large‑scale nearest‑neighbor search (10 M+ vectors) | FAISS (IVF‑Flat + Product Quantization) on GPU | Rebuilt nightly from source database |
| **Search / Prediction Service** | Receive refined embedding, query index, return top‑k results | FastAPI endpoint `POST /search` | JSON response includes IDs, scores, titles, thumbnails |
| **Monitoring & Logging** | Export metrics, collect logs, alert on anomalies | Prometheus + Grafana, Loki + OpenTelemetry | Alerts on latency > 250 ms, error > 2 % |
| **CI/CD Pipelines** | Build, test, and deploy mobile and cloud artifacts | GitHub Actions → Fastlane (mobile) & Docker (cloud) | Automated version bump, changelog generation |

---

## 3. Data Flow (Request / Response Sequence)
1. **Capture** – User records or selects an image/video (≤ 5 s).
2. **Pre‑process** – Image resized to 224 × 224, pixel‑wise normalization.
3. **Local Encoding** – MobileViT‑XS produces embedding `e_local` (128‑dim).
4. **Cache Insert** – `e_local` stored in the on‑device FAISS‑lite cache (key = image hash).
5. **Network Check** – If online, the app strips any PII and sends JSON `{embedding: base64(e_local), metadata}` over HTTPS to `POST /search`.
6. **Authentication** – Request includes `Authorization: Bearer <jwt>`; missing or expired tokens are refreshed via the Auth Service.
7. **JEPA Refinement (cloud)** – The service runs `e_local` through the JEPA model, producing `e_refined`.
8. **Similarity Search** – `e_refined` queries the FAISS‑IVF/PQ index, returning the top‑k nearest IDs.
9. **Result Assembly** – For each ID the service fetches title, thumbnail URL, and confidence score.
10. **Response** – JSON `{results:[{id, score, title, thumb_url}], timestamp}` is returned to the client.
11. **Display** – The mobile UI shows a scrollable list; tapping opens a detail view.
12. **Offline Fallback** – If step 5 fails, the client queries the local FAISS‑lite cache and returns those results with an “offline” badge.

**Latency Targets**
- On‑device encode: ≤ 30 ms
- Network RTT (4G/5G): 50‑150 ms
- Cloud JEPA + FAISS query: ≤ 100 ms (GPU)
- End‑to‑end (online): **≈ 200‑300 ms**

---

## 4. Security & Privacy
| Aspect | Measure |
|--------|----------|
| **Transport** | TLS 1.3 with strong cipher suites; certificates via Let’s Encrypt or cloud provider |
| **Authentication** | JWT access tokens (5 min expiry) issued via OAuth 2.0 client‑credentials; signed RS256; JWKS for verification |
| **Data Minimisation** | Only the 128‑dim embedding (≈ 500 bytes) is transmitted; raw media never leaves the device |
| **At‑Rest Encryption** | Cloud Object Storage (S3/GCS) encrypted SSE‑AES256; mobile cache encrypted with iOS Keychain / Android Keystore |
| **Privacy Rights** | `DELETE /user/{id}/embeddings` endpoint allows users to erase their data; identifiers are pseudonymous |
| **Rate Limiting** | Token‑bucket algorithm: default 200 req/min per device, burst up to 500 req/min |
| **Logging** | No raw embeddings are logged; logs contain request IDs, timestamps, and status codes only |
| **Auditing** | Weekly OWASP ZAP scans, Dependabot dependency alerts, and quarterly third‑party security review |

---

## 5. Performance & Scalability
- **Target latency**: ≤ 300 ms end‑to‑end (online). SLO: 95th‑percentile ≤ 250 ms, error‑rate ≤ 1 %.
- **Horizontal scaling** – Stateless services; additional replicas can be added behind a load balancer.
- **Resource sizing** – JEPA Service runs on GPU‑enabled VMs (e.g., NVIDIA A100) in production; Search Service runs on CPU‑optimized instances.
- **Caching** – On‑device FAISS‑lite cache reduces round‑trips for recent queries.
- **Rate‑limit enforcement** – API gateway enforces adaptive limits to protect backend resources.

---

## 6. Deployment & Operations
- **Container images** – Built via GitHub Actions, scanned with Trivy, and pushed to a private container registry.
- **Runtime** – Currently deployed on a single Ubuntu VM using Docker Compose (`api-gateway`, `jepa-service`, `search-service`). A Helm chart is maintained for optional Kubernetes deployment.
- **CI/CD** – GitHub Actions workflow: `lint ➜ unit ➜ integration ➜ docker build ➜ helm upgrade` (or Docker Compose restart). Manual approval required for production rollout.
- **Observability** – Prometheus scrapes `/metrics` from each service (each uses its own `CollectorRegistry`). Grafana dashboards visualize latency, request rates, and error percentages.
- **Alerting** – PagerDuty integration triggers on latency > 250 ms, error rate > 2 %, or CPU utilization > 80 % for 5 min.
- **Rollback** – `docker compose restart` for quick rollback on the VM; Helm `rollback` for Kubernetes.
- **Documentation** – OpenAPI spec generated from FastAPI, SDKs for Kotlin & Swift generated via Swagger Codegen and version‑controlled under `docs/`.

---

## 7. Observability & Monitoring
- **Metrics** – `process_cpu_seconds_total` (per service), request latency histograms, error counters.
- **Health checks** – `/healthz` endpoint for each service (returns 200 if ready). Used by the load balancer and Kubernetes probes.
- **Logging** – Structured JSON logs collected by Loki; correlated with trace IDs via OpenTelemetry.
- **Dashboards** – Grafana dashboard (`grafana_dashboard.json`) shows per‑service CPU, request latency, error rates, and cache hit‑ratio.

---

## 8. Testing Strategy
| Level | Goal | Tools |
|-------|------|-------|
| **Unit** | Validate encoder shape, cache ops, token handling | XCTest (iOS), JUnit (Android), pytest (Python) |
| **Integration** | End‑to‑end request/response, token refresh, offline fallback | Postman/Newman, pytest‑asyncio, Espresso / XCUITest |
| **Performance** | Measure latency, GPU utilisation, memory footprint | Locust, Firebase Performance Monitoring, Instruments |
| **Security** | Verify TLS, JWT validation, rate‑limit enforcement | OWASP ZAP, jwt‑cli |
| **Regression** | Snapshot UI tests, full suite on each PR | GitHub Actions, Detox (cross‑platform) |
| **Chaos** | Simulate network loss, token expiry, GPU overload | `tc` traffic control, chaos‑monkey scripts |

**Pass Criteria** – 95th‑percentile latency ≤ 250 ms, error rate < 1 %, unit test coverage ≥ 100 % for core modules.

---

## 9. Next Steps
1. Commit this refined spec to `docs/superpowers/specs/2026-03-24-hybrid-edge-cloud-design.md`.
2. Run the `spec-document-reviewer` sub‑agent to validate completeness and consistency.
3. After reviewer approval, I will ask you to review the committed file before proceeding to the implementation‑plan phase.

*Please confirm the spec is satisfactory or let me know any changes you’d like before I commit it.*