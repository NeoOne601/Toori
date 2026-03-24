# Hybrid Edge-Cloud Image Embedding & Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cross‑platform mobile app (iOS/Android) that generates image/video embeddings on‑device, sends them to a cloud JEPA service for refined prediction and similarity search, returning top‑k results.

**Architecture:** Mobile app runs a lightweight MobileViT‑XS encoder and a FAISS‑lite cache; a cloud service (FastAPI) hosts a Llama‑3.2‑JEPA model and a GPU‑accelerated FAISS‑IVF/PQ index. Communication over HTTPS with JWT auth.

**Tech Stack:** SwiftUI + Combine, Jetpack Compose + Kotlin Coroutines, Kotlin Multiplatform shared code, Core ML / TensorFlow Lite, FastAPI + Uvicorn, TorchServe, FAISS‑GPU, Docker, Kubernetes, GitHub Actions, Fastlane.

---

### Task 1: Initialise Mobile Repository

**Files:**
- Create: `mobile/ios/TooriApp/Info.plist`
- Create: `mobile/android/app/build.gradle.kts`
- Create: `mobile/shared/src/commonMain/kotlin/com/toori/shared/Embedding.kt`
- Create: `mobile/shared/src/iosMain/kotlin/com/toori/shared/EncoderIOS.kt`
- Create: `mobile/shared/src/androidMain/kotlin/com/toori/shared/EncoderAndroid.kt`
- Create: `mobile/shared/src/commonTest/kotlin/com/toori/shared/EmbeddingTest.kt`

- [ ] **Step 1: Write failing test for encoder interface**
```kotlin
class EmbeddingTest {
    @Test fun `encoder returns 128‑dim vector`() {
        val encoder = EncoderFactory.create()
        val vec = encoder.encode(sampleImage())
        assertEquals(128, vec.size)
    }
}
```
- [ ] **Step 2: Run test to verify it fails** (`./gradlew test` expects compile error).
- [ ] **Step 3: Implement minimal EncoderFactory stub**
```kotlin
object EncoderFactory {
    fun create(): Encoder = object : Encoder {
        override fun encode(image: Image): FloatArray = FloatArray(128) { 0f }
    }
}
```
- [ ] **Step 4: Run test – should pass**.
- [ ] **Step 5: Commit**
```bash
git add mobile/shared
git commit -m "feat: add encoder interface and stub implementation"
```

### Task 2: Integrate MobileViT‑XS Model (iOS)

**Files:**
- Add: `mobile/ios/TooriApp/Model/MobileViTXS.mlmodelc`
- Modify: `mobile/ios/TooriApp/EncoderIOS.kt:1-120`
- Add test: `mobile/shared/src/iosTest/kotlin/com/toori/shared/EncoderIOSTest.kt`

- [ ] **Step 1: Write failing test that expects non‑zero embeddings**
```kotlin
@Test fun `real iOS encoder produces non‑zero vector`() {
    val encoder = EncoderIOS()
    val vec = encoder.encode(sampleImage())
    assertTrue(vec.any { it != 0f })
}
```
- [ ] **Step 2: Run test – fails (model not loaded).**
- [ ] **Step 3: Add Core ML loading code**
```kotlin
class EncoderIOS : Encoder {
    private val model = MobileViTXS().model
    override fun encode(image: Image): FloatArray {
        val input = ... // convert Image to CVPixelBuffer
        val output = model.prediction(input)
        return output.embedding
    }
}
```
- [ ] **Step 4: Run test – should pass (assuming model file present).**
- [ ] **Step 5: Commit**
```bash
git add mobile/ios/TooriApp/Model
git commit -m "feat: iOS MobileViT‑XS encoder implementation"
```

### Task 3: Integrate MobileViT‑XS Model (Android)

**Files:**
- Add: `mobile/android/app/src/main/assets/mobilevit_xs.tflite`
- Modify: `mobile/android/src/main/kotlin/com/toori/EncoderAndroid.kt:1-130`
- Add test: `mobile/shared/src/androidTest/kotlin/com/toori/shared/EncoderAndroidTest.kt`

- [ ] **Step 1: Write failing test expecting non‑zero vector** (same as iOS).
- [ ] **Step 2: Run test – fails (no model).**
- [ ] **Step 3: Implement TensorFlow Lite interpreter loading**
```kotlin
class EncoderAndroid : Encoder {
    private val interpreter = Interpreter(ByteBufferUtil.loadAsset(context, "mobilevit_xs.tflite"))
    override fun encode(image: Image): FloatArray {
        val input = ImageProcessor.toTensor(image)
        val output = FloatArray(128)
        interpreter.run(input, output)
        return output
    }
}
```
- [ ] **Step 4: Run test – passes.**
- [ ] **Step 5: Commit**
```bash
git add mobile/android/app/src/main/assets
git commit -m "feat: Android MobileViT‑XS encoder implementation"
```

### Task 4: Implement Local FAISS‑lite Cache (iOS & Android)

**Files:**
- Create: `mobile/shared/src/commonMain/kotlin/com/toori/cache/EmbeddingCache.kt`
- Create platform‑specific wrappers: `iOS/EmbeddingCacheIOS.kt`, `Android/EmbeddingCacheAndroid.kt`
- Add tests: `mobile/shared/src/commonTest/kotlin/com/toori/cache/EmbeddingCacheTest.kt`

- [ ] **Step 1: Write failing test for cache insert & retrieval**
```kotlin
@Test fun `cache stores and retrieves embedding`() {
    val cache = EmbeddingCache()
    val key = "img1"
    val vec = FloatArray(128) { it.toFloat() }
    cache.put(key, vec)
    assertTrue(cache.get(key)!!.contentEquals(vec))
}
```
- [ ] **Step 2: Run test – fails (class missing).**
- [ ] **Step 3: Implement minimal in‑memory LRU cache (max 10k entries, TTL 48h)**
```kotlin
class EmbeddingCache {
    private val map = LinkedHashMap<String, Pair<FloatArray, Long>>(/*accessOrder=*/true)
    fun put(key: String, vec: FloatArray) {
        map[key] = vec to System.currentTimeMillis()
        if (map.size > 10_000) {
            val eldest = map.entries.iterator().next()
            map.remove(eldest.key)
        }
    }
    fun get(key: String): FloatArray? {
        val entry = map[key] ?: return null
        val (vec, ts) = entry
        if (System.currentTimeMillis() - ts > 48L * 60 * 60 * 1000) {
            map.remove(key); return null
        }
        return vec
    }
}
```
- [ ] **Step 4: Run test – passes.**
- [ ] **Step 5: Commit**
```bash
git add mobile/shared/src/commonMain/kotlin/com/toori/cache
git commit -m "feat: embedding LRU cache implementation"
```

### Task 5: Build Cloud API Service Skeleton

**Files:**
- Create: `cloud/api/main.py`
- Create: `cloud/api/auth.py`
- Create: `cloud/api/schemas.py`
- Create: `cloud/api/tests/test_auth.py`

- [ ] **Step 1: Write failing test for JWT verification**
```python
def test_invalid_token(client):
    response = client.get("/search", headers={"Authorization": "Bearer badtoken"})
    assert response.status_code == 401
```
- [ ] **Step 2: Run test – fails (endpoint missing).**
- [ ] **Step 3: Implement FastAPI app with dependency that validates JWT via Auth0 JWKS**
```python
app = FastAPI()
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, jwks, algorithms=["RS256"])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/search")
def search(query: SearchQuery, user=Depends(get_current_user)):
    # placeholder – will call JEPA service later
    return {"results": []}
```
- [ ] **Step 4: Run test – passes (401).**
- [ ] **Step 5: Commit**
```bash
git add cloud/api
git commit -m "feat: FastAPI skeleton with JWT auth"
```

### Task 6: Add JEPA Model Service (GPU)

**Files:**
- Create: `cloud/jepa_service/Dockerfile`
- Create: `cloud/jepa_service/app.py`
- Create: `cloud/jepa_service/tests/test_jepa.py`

- [ ] **Step 1: Write failing test that calls `/embed` and expects 128‑dim output**
```python
def test_embed_returns_vector(client):
    resp = client.post("/embed", json={"embedding": [0]*128})
    assert resp.status_code == 200
    assert len(resp.json()["refined_embedding"]) == 128
```
- [ ] **Step 2: Run test – fails (endpoint missing).**
- [ ] **Step 3: Implement TorchServe wrapper that loads Llama‑3.2‑JEPA (versioned v1.2.0) and exposes `/embed`**
```python
app = FastAPI()
model = load_model("l3.2-jepa-v1.2.0.pt")

@app.post("/embed")
def embed(payload: EmbedPayload):
    tensor = torch.tensor(payload.embedding).unsqueeze(0)
    refined = model(tensor)
    return {"refined_embedding": refined.squeeze().tolist()}
```
- [ ] **Step 4: Run test – passes.**
- [ ] **Step 5: Commit Dockerfile and app**
```bash
git add cloud/jepa_service
git commit -m "feat: JEPA service with TorchServe wrapper"
```

### Task 7: Implement FAISS‑IVF/PQ Index Service

**Files:**
- Create: `cloud/search_service/main.py`
- Create: `cloud/search_service/index_builder.py`
- Create: `cloud/search_service/tests/test_search.py`

- [ ] **Step 1: Write failing test that queries empty index and expects empty list**
```python
def test_search_empty(client):
    resp = client.post("/search", json={"embedding": [0]*128})
    assert resp.json()["results"] == []
```
- [ ] **Step 2: Run – fails (no service).**
- [ ] **Step 3: Implement service that loads FAISS‑IVF/PQ GPU index, receives refined embedding, returns top‑k IDs**
```python
app = FastAPI()
index = faiss.read_index("index.faiss")

@app.post("/search")
def search(payload: SearchPayload):
    vec = np.array(payload.embedding, dtype="float32")
    D, I = index.search(vec.reshape(1, -1), k=10)
    return {"results": I[0].tolist(), "scores": D[0].tolist()}
```
- [ ] **Step 4: Run test – passes (empty index returns empty list).**
- [ ] **Step 5: Commit**
```bash
git add cloud/search_service
git commit -m "feat: FAISS search service skeleton"
```

### Task 8: Wire Cloud API to JEPA + Search Services

**Files:**
- Modify: `cloud/api/main.py`
- Add env vars in `cloud/api/.env.sample`

- [ ] **Step 1: Write integration test that sends embedding through API and receives mock results** (use testclient with mocked JEPA & search services).
- [ ] **Step 2: Run – fails (no integration).**
- [ ] **Step 3: In API `/search` endpoint, call JEPA service `/embed` then forward refined embedding to search service `/search`. Handle errors gracefully.
- [ ] **Step 4: Run integration test – passes.
- [ ] **Step 5: Commit**
```bash
git add cloud/api/main.py cloud/api/.env.sample
git commit -m "feat: API orchestration of JEPA and FAISS services"
```

### Task 9: CI/CD Pipelines (Mobile & Cloud)

**Files:**
- Add: `.github/workflows/mobile.yml`
- Add: `.github/workflows/cloud.yml`
- Add: `fastlane/Fastfile`

- [ ] **Step 1: Write failing CI job that runs `gradle test` and expects success (currently passes).**
- [ ] **Step 2: Run workflow locally via act – passes.
- [ ] **Step 3: Add Docker build & push steps for `cloud/jepa_service` and `cloud/search_service`.
- [ ] **Step 4: Run full CI – passes.
- [ ] **Step 5: Commit CI files**
```bash
git add .github/workflows fastlane/Fastfile
git commit -m "ci: add mobile and cloud pipelines"
```

### Task 10: Documentation & Release Notes

**Files:**
- Update: `README.md` with Architecture diagram link.
- Create: `docs/architecture.md`
- Create: `docs/release-notes/2026-03-24.md`

- [ ] **Step 1: Write placeholder docs and failing test that checks `README` contains a "Getting Started" section.
- [ ] **Step 2: Run test – fails.
- [ ] **Step 3: Add sections to README and architecture doc.
- [ ] **Step 4: Run test – passes.
- [ ] **Step 5: Commit docs**
```bash
git add README.md docs/architecture.md docs/release-notes/2026-03-24.md
git commit -m "docs: add architecture overview and release notes"
```

---

### Task 11: Security \& Secrets

**Files:**
- Create: `cloud/api/.env.sample` (example env vars for JWT secret, DB URLs, etc.)
- Create: `cloud/api/secret_manager.py` (loads secrets from environment or a secrets manager, abstracts access)
- Add: `cloud/api/tests/test_security.py`

- [ ] **Step 1: Write failing test that ensures the API returns 401 when Authorization header is missing**
```python
def test_missing_auth(client):
    response = client.get("/search")
    assert response.status_code == 401
```
- [ ] **Step 2: Run test – fails (auth dependency not handling missing token).
- [ ] **Step 3: Update `cloud/api/auth.py` to use `secret_manager` for JWT verification and to reject missing tokens with proper error.
- [ ] **Step 4: Run test – passes.
- [ ] **Step 5: Add CI secret scanning step (using GitHub secret scanning action) to the workflow.
- [ ] **Step 6: Commit security assets**
```bash
git add cloud/api/.env.sample cloud/api/secret_manager.py cloud/api/tests/test_security.py
git commit -m "security: add secret manager and auth tests"
```

### Task 12: Observability \& Monitoring

**Files:**
- Modify: `cloud/jepa_service/app.py` – add Prometheus metrics instrumentation.
- Modify: `cloud/search_service/main.py` – expose `/metrics` endpoint.
- Add: `cloud/monitoring/prometheus.yml` – scrape config.
- Add: `cloud/monitoring/grafana_dashboard.json`
- Add: `cloud/monitoring/tests/test_monitoring.py`

- [ ] **Step 1: Write failing test that checks `/metrics` endpoint returns HTTP 200 and contains `process_cpu_seconds_total`.
- [ ] **Step 2: Run test – fails (no metrics).
- [ ] **Step 3: Integrate `prometheus_client` library, instrument request latency, error counts, and cache hit/miss counters.
- [ ] **Step 4: Run test – passes.
- [ ] **Step 5: Add Helm chart values for Prometheus ServiceMonitor.
- [ ] **Step 6: Commit monitoring changes**
```bash
git add cloud/jepa_service/app.py cloud/search_service/main.py cloud/monitoring/
git commit -m "observability: add Prometheus metrics and monitoring config"
```

**All tasks follow TDD, DRY, YAGNI, and frequent commits.**
