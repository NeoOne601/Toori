from __future__ import annotations

import asyncio
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, make_asgi_app

from .error_types import (
    SmritiError,
    SmritiIngestionError,
    SmritiPipelineError,
    SmritiRateLimitError,
    SmritiRecallError,
    SmritiSchemaError,
    SmritiUncertaintyError,
)
from .observability import CorrelationContext, TokenBucketRateLimiter, get_logger
from .service import RuntimeContainer
from .models import (
    AnalyzeRequest,
    AudioQueryRequest,
    AudioQueryResponse,
    ChallengeEvaluateRequest,
    JEPAForecastRequest,
    LivingLensTickRequest,
    PlanningRolloutRequest,
    PlanningRolloutResponse,
    ProviderHealthResponse,
    ProofReportGenerateRequest,
    QueryRequest,
    RecoveryBenchmarkRun,
    RecoveryBenchmarkRunRequest,
    RuntimeSettings,
    RuntimeSnapshotResponse,
    ToolStateObserveRequest,
    ToolStateObserveResponse,
    WorldModelConfig,
    WorldModelConfigUpdate,
    WorldModelStatus,
    ShareObservationEventRequest,
    ShareObservationRequest,
    ShareObservationResponse,
    SmritiIngestRequest,
    SmritiIngestResponse,
    SmritiMigrationRequest,
    SmritiMigrationResult,
    SmritiPruneRequest,
    SmritiPruneResult,
    SmritiRecallFeedback,
    SmritiRecallFeedbackResult,
    SmritiRecallRequest,
    SmritiRecallResponse,
    SmritiStorageConfig,
    SmritiTagPersonRequest,
    SmritiTagPersonResponse,
    StorageUsageReport,
    WatchFolderStatus,
)


def _assert_supported_python() -> None:
    if sys.version_info[:2] == (3, 11):
        return
    raise RuntimeError(
        "Toori runtime requires Python 3.11. "
        f"Detected {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}. "
        "Launch the backend with python3.11 to avoid unsupported native-extension crashes."
    )


@asynccontextmanager
async def _runtime_lifespan(app: FastAPI):
    runtime = getattr(app.state, "runtime", None)
    smriti_daemon = getattr(app.state, "smriti_daemon", None)
    try:
        if runtime is not None and hasattr(runtime, "_load_sag_templates"):
            runtime._load_sag_templates()
        if smriti_daemon is not None:
            await smriti_daemon.start()
        if runtime is not None and hasattr(runtime, "restore_smriti_watch_folders"):
            await runtime.restore_smriti_watch_folders()
        yield
    finally:
        if smriti_daemon is not None:
            await smriti_daemon.stop()
        if runtime is not None:
            if hasattr(runtime, "_save_sag_templates"):
                runtime._save_sag_templates()
            if hasattr(runtime, "shutdown"):
                await runtime.shutdown()


def create_app(data_dir: str | None = None) -> FastAPI:
    from .smriti_ingestion import SmritiIngestionDaemon

    _assert_supported_python()
    registry = CollectorRegistry()
    analyze_counter = Counter(
        "toori_analyze_requests_total",
        "Analyze requests",
        registry=registry,
    )
    analyze_latency = Histogram(
        "toori_analyze_latency_seconds",
        "Analyze latency",
        registry=registry,
    )
    share_events = Counter(
        "toori_share_events_total",
        "Observation share events",
        ["event_type"],
        registry=registry,
    )
    app = FastAPI(title="Toori Runtime", version="1.0.0", lifespan=_runtime_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/metrics", make_asgi_app(registry=registry))
    app.state.runtime = RuntimeContainer(data_dir=data_dir)
    app.state.rate_limiter = TokenBucketRateLimiter(rate_per_second=20.0, burst=60)
    app.state.smriti_daemon = SmritiIngestionDaemon(
        app.state.runtime.smriti_db,
        app.state.runtime._ensure_jepa_pool,
        runtime_container=app.state.runtime,
    )
    app.state.runtime.smriti_daemon = app.state.smriti_daemon

    def rate_limit_dependency(endpoint: str):
        def dependency(request: Request) -> None:
            client_host = request.client.host if request.client and request.client.host else "unknown"
            key = f"{endpoint}:{client_host}"
            if not app.state.rate_limiter.allow(key):
                retry_after_s = 1.0 / app.state.rate_limiter.rate_per_second if app.state.rate_limiter.rate_per_second else 1.0
                raise SmritiRateLimitError(endpoint=endpoint, retry_after_s=retry_after_s)

        return dependency

    @app.middleware("http")
    async def correlation_middleware(request: Request, call_next):
        previous_correlation_id = CorrelationContext.get()
        correlation_id = request.headers.get("X-Correlation-ID") or CorrelationContext.new()
        CorrelationContext.set(correlation_id)
        logger = get_logger("runtime")
        started = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0
            status_code = getattr(response, "status_code", 500)
            logger.info(
                "http_request_completed",
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=round(duration_ms, 3),
            )
            CorrelationContext.set(previous_correlation_id)
            if response is not None:
                response.headers["X-Correlation-ID"] = correlation_id

    def _smriti_error_payload(error: str, exc: SmritiError, **extra: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "error": error,
            "message": str(exc),
            "correlation_id": exc.correlation_id,
        }
        payload.update(extra)
        return payload

    @app.exception_handler(SmritiRateLimitError)
    async def rate_limit_handler(request: Request, exc: SmritiRateLimitError):
        get_logger("runtime").warning(
            "rate_limited",
            endpoint=exc.endpoint,
            retry_after_s=exc.retry_after_s,
        )
        return JSONResponse(
            status_code=429,
            content=_smriti_error_payload(
                "rate_limited",
                exc,
                retry_after_s=exc.retry_after_s,
            ),
            headers={"Retry-After": str(int(exc.retry_after_s))},
        )

    @app.exception_handler(SmritiUncertaintyError)
    async def uncertainty_handler(request: Request, exc: SmritiUncertaintyError):
        get_logger("runtime").info(
            "uncertainty_gate_triggered",
            region_id=exc.region_id,
            consistency_score=exc.consistency_score,
        )
        return JSONResponse(
            status_code=200,
            content={
                "status": "uncertain",
                "region_id": exc.region_id,
                "consistency_score": exc.consistency_score,
                "uncertainty_map": exc.uncertainty_map.tolist(),
                "message": "Smriti is not confident enough to describe this region.",
                "correlation_id": exc.correlation_id,
            },
        )

    @app.exception_handler(SmritiError)
    async def smriti_error_handler(request: Request, exc: SmritiError):
        if isinstance(exc, SmritiSchemaError):
            status_code = 409
            error = "schema_error"
        elif isinstance(exc, (SmritiPipelineError, SmritiIngestionError, SmritiRecallError)):
            status_code = 400 if not isinstance(exc, SmritiPipelineError) else 500
            error = "ingestion_error" if isinstance(exc, SmritiIngestionError) else "recall_error"
            if isinstance(exc, SmritiPipelineError):
                error = "pipeline_error"
        else:
            status_code = 500
            error = "smriti_error"
        get_logger("runtime").error(
            "smriti_error",
            error=error,
            status_code=status_code,
            detail=str(exc),
        )
        return JSONResponse(status_code=status_code, content=_smriti_error_payload(error, exc))

    def require_auth(x_api_key: str | None = Header(default=None)) -> None:
        settings = app.state.runtime.get_settings()
        if settings.auth_mode == "disabled":
            return
        if settings.auth_mode == "loopback":
            return
        expected = settings.providers.get("cloud")
        configured_key = expected.api_key if expected else None
        if not configured_key or x_api_key != configured_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/v1/settings",
        response_model=RuntimeSettings,
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("settings.get"))],
    )
    def get_settings() -> RuntimeSettings:
        return app.state.runtime.get_settings()

    @app.put(
        "/v1/settings",
        response_model=RuntimeSettings,
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("settings"))],
    )
    def put_settings(settings: RuntimeSettings) -> RuntimeSettings:
        return app.state.runtime.update_settings(settings)

    @app.get("/v1/world-model/status", response_model=WorldModelStatus)
    def world_model_status() -> WorldModelStatus:
        return app.state.runtime.get_world_model_status()

    @app.get(
        "/v1/world-model/config",
        response_model=WorldModelConfig,
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("world_model.config.get"))],
    )
    def get_world_model_config() -> WorldModelConfig:
        return app.state.runtime.get_vjepa2_settings()

    @app.put(
        "/v1/world-model/config",
        response_model=WorldModelConfig,
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("world_model.config"))],
    )
    def put_world_model_config(payload: WorldModelConfigUpdate) -> WorldModelConfig:
        return app.state.runtime.update_vjepa2_settings(payload.model_path, payload.cache_dir, payload.n_frames)

    @app.post(
        "/v1/world-model/retry-native",
        response_model=WorldModelStatus,
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("world_model.retry_native"))],
    )
    async def post_world_model_retry_native() -> WorldModelStatus:
        return await app.state.runtime.retry_native_jepa()

    @app.get(
        "/v1/providers/health",
        response_model=ProviderHealthResponse,
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("providers.health"))],
    )
    def provider_health():
        return app.state.runtime.provider_health()

    @app.get(
        "/v1/observations",
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("observations"))],
    )
    def observations(
        session_id: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
        summary_only: bool = Query(default=False),
    ):
        return app.state.runtime.list_observations(
            session_id=session_id,
            limit=limit,
            summary_only=summary_only,
        )

    @app.get(
        "/v1/file",
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("file"))],
    )
    def runtime_file(path: str = Query(..., min_length=1)):
        target = Path(path).expanduser().resolve()
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        data_root = Path(app.state.runtime.data_dir).resolve()
        try:
            target.relative_to(data_root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="File path is outside runtime data directory") from exc
        import mimetypes
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        return Response(
            content=target.read_bytes(),
            media_type=content_type,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "public, max-age=3600",
            },
        )

    @app.post(
        "/v1/analyze",
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("analyze"))],
    )
    def analyze(payload: AnalyzeRequest):
        analyze_counter.inc()
        with analyze_latency.time():
            return app.state.runtime.analyze(payload)

    @app.post(
        "/v1/living-lens/tick",
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("living_lens.tick"))],
    )
    async def living_lens_tick(payload: LivingLensTickRequest):
        analyze_counter.inc()
        with analyze_latency.time():
            return await app.state.runtime.living_lens_tick(payload)

    @app.post(
        "/v1/jepa/forecast",
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("jepa.forecast"))],
    )
    def jepa_forecast(
        payload: JEPAForecastRequest,
        k: int | None = Query(default=None, ge=1, le=32),
    ):
        horizon = k if k is not None else payload.k
        return app.state.runtime.forecast_jepa(payload.session_id, horizon)

    @app.post(
        "/v1/query",
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("query"))],
    )
    def query(payload: QueryRequest):
        return app.state.runtime.query(payload)

    @app.get(
        "/v1/world-state",
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("world_state"))],
    )
    def world_state(session_id: str = Query(..., min_length=1)):
        return app.state.runtime.get_world_state(session_id)

    @app.get(
        "/v1/runtime/snapshot",
        response_model=RuntimeSnapshotResponse,
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("runtime.snapshot"))],
    )
    def runtime_snapshot(
        session_id: str = Query(..., min_length=1),
        observation_limit: int = Query(default=12, ge=1, le=48),
    ):
        return app.state.runtime.get_runtime_snapshot(session_id, observation_limit=observation_limit)

    @app.post(
        "/v1/tool-state/observe",
        response_model=ToolStateObserveResponse,
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("tool_state.observe"))],
    )
    def tool_state_observe(payload: ToolStateObserveRequest):
        return app.state.runtime.observe_tool_state(payload)

    @app.post(
        "/v1/planning/rollout",
        response_model=PlanningRolloutResponse,
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("planning.rollout"))],
    )
    def planning_rollout(payload: PlanningRolloutRequest):
        return app.state.runtime.plan_rollout(payload)

    @app.post(
        "/v1/planning/rollout/narrate",
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("planning.rollout.narrate"))],
    )
    async def narrate_rollout(payload: dict):
        try:
            from .gemma4_bridge import Gemma4Bridge, deterministic_rollout_summary
            mlx_p = app.state.runtime.providers.get("mlx")
            if not mlx_p:
                return {"summary": deterministic_rollout_summary(payload)}
            bridge_call = app.state.runtime._mlx_bridge_call()
            if bridge_call is None:
                return {"summary": deterministic_rollout_summary(payload)}
            bridge = Gemma4Bridge(bridge_call)
            text = await bridge.narrate_rollout(payload)
            return {"summary": text}
        except Exception as e:
            import logging
            from .gemma4_bridge import deterministic_rollout_summary
            logging.getLogger(__name__).error("Narrate rollout failed: %s", e)
            return {"summary": deterministic_rollout_summary(payload)}

    @app.post(
        "/v1/benchmarks/recovery/run",
        response_model=RecoveryBenchmarkRun,
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("benchmarks.recovery.run"))],
    )
    def run_recovery_benchmark(payload: RecoveryBenchmarkRunRequest):
        return app.state.runtime.run_recovery_benchmark(payload)

    @app.get(
        "/v1/benchmarks/recovery/{benchmark_id}",
        response_model=RecoveryBenchmarkRun,
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("benchmarks.recovery.get"))],
    )
    def get_recovery_benchmark(benchmark_id: str):
        benchmark = app.state.runtime.get_recovery_benchmark(benchmark_id)
        if benchmark is None:
            raise HTTPException(status_code=404, detail="Recovery benchmark not found")
        return benchmark

    @app.post(
        "/v1/challenges/evaluate",
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("challenges.evaluate"))],
    )
    def evaluate_challenge(payload: ChallengeEvaluateRequest):
        return app.state.runtime.evaluate_challenge(payload)

    @app.post(
        "/v1/proof-report/generate",
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("proof_report.generate"))],
    )
    async def generate_proof_report(payload: ProofReportGenerateRequest):
        return await app.state.runtime.generate_proof_report(payload.session_id, payload.chart_b64)

    @app.post(
        "/v1/share/observation",
        response_model=ShareObservationResponse,
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("share.observation"))],
    )
    def share_observation(payload: ShareObservationRequest):
        try:
            response = app.state.runtime.build_observation_share(payload.session_id, payload.observation_id)
            app.state.runtime.record_observation_share_event(
                response.session_id,
                response.observation_id,
                "share_clicked",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Observation not found") from exc
        share_events.labels(event_type="share_clicked").inc()
        return response

    @app.post(
        "/v1/share/observation/event",
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("share.observation.event"))],
    )
    def share_observation_event(payload: ShareObservationEventRequest):
        try:
            app.state.runtime.record_observation_share_event(
                payload.session_id,
                payload.observation_id,
                payload.event_type,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Observation not found") from exc
        share_events.labels(event_type=payload.event_type).inc()
        return {"recorded": True}

    @app.get(
        "/v1/proof-report/latest",
        dependencies=[Depends(require_auth), Depends(rate_limit_dependency("proof_report.latest"))],
    )
    def latest_proof_report():
        latest = app.state.runtime.latest_proof_report()
        if latest is None or not latest.exists():
            raise HTTPException(status_code=404, detail="Proof report not found")
        content = latest.read_bytes()
        if not content.startswith(b"%PDF"):
            raise HTTPException(status_code=500, detail="Proof report is not a valid PDF")
        return Response(
            content=content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{latest.name}"',
                "Cache-Control": "no-cache",
            },
        )

    @app.websocket("/v1/events")
    async def events_socket(websocket: WebSocket):
        await websocket.accept()
        queue = app.state.runtime.events.subscribe()
        try:
            while True:
                message = await queue.get()
                await websocket.send_json(message.model_dump(mode="json"))
        except WebSocketDisconnect:
            app.state.runtime.events.unsubscribe(queue)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            app.state.runtime.events.unsubscribe(queue)
            raise

    recall_limiter = TokenBucketRateLimiter(rate_per_second=5.0, burst=10)

    @app.post("/v1/smriti/ingest", dependencies=[Depends(require_auth)])
    async def smriti_ingest(payload: SmritiIngestRequest) -> SmritiIngestResponse:
        daemon = getattr(app.state, "smriti_daemon", None)
        if daemon is None:
            raise HTTPException(status_code=503, detail="Ingestion daemon not initialized")
        if payload.folder_path:
            queued = await daemon.watch_folder(payload.folder_path)
            return SmritiIngestResponse(queued=queued, status="watching")
        if payload.file_path:
            accepted = await daemon.ingest_file(payload.file_path)
            return SmritiIngestResponse(
                queued=1 if accepted else 0,
                status="queued" if accepted else "duplicate",
            )
        raise HTTPException(status_code=422, detail="folder_path or file_path required")

    @app.get("/v1/smriti/status", dependencies=[Depends(require_auth)])
    async def smriti_status() -> dict:
        daemon = getattr(app.state, "smriti_daemon", None)
        stats = daemon.get_stats() if daemon else {}
        return {"ingestion": stats, "status": "running" if daemon else "not_initialized"}

    @app.post("/v1/smriti/recall", dependencies=[Depends(require_auth)])
    async def smriti_recall(payload: SmritiRecallRequest) -> SmritiRecallResponse:
        session_key = getattr(payload, "session_id", "default")
        if not recall_limiter.allow(session_key):
            raise SmritiRateLimitError(endpoint="/v1/smriti/recall", retry_after_s=1.0)
        return await app.state.runtime.smriti_recall(payload)

    @app.post("/v1/audio/query", dependencies=[Depends(require_auth)])
    async def audio_query(payload: AudioQueryRequest) -> AudioQueryResponse:
        """Endpoint for Audio-JEPA Phase 1 hum/mic retrieval."""
        session_key = getattr(payload, "session_id", "default")
        if not recall_limiter.allow(session_key):
            raise SmritiRateLimitError(endpoint="/v1/audio/query", retry_after_s=1.0)
        return await app.state.runtime.audio_query(payload)

    @app.post("/v1/smriti/recall/feedback", dependencies=[Depends(require_auth)])
    async def recall_feedback(payload: SmritiRecallFeedback) -> SmritiRecallFeedbackResult:
        return app.state.runtime.smriti_recall_feedback(payload)

    @app.get("/v1/smriti/media/{media_id}/neighbors", dependencies=[Depends(require_auth)])
    async def smriti_neighbors(
        media_id: str,
        top_k: int = Query(default=6, ge=1, le=20),
    ) -> dict:
        return app.state.runtime.smriti_media_neighbors(media_id, top_k=top_k)

    @app.get("/v1/smriti/media/{media_id}", dependencies=[Depends(require_auth)])
    async def smriti_media_detail(media_id: str) -> dict:
        detail = app.state.runtime.smriti_media_detail(media_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Media not found")
        return detail

    @app.post("/v1/smriti/tag/person", dependencies=[Depends(require_auth)])
    async def smriti_tag_person(payload: SmritiTagPersonRequest) -> SmritiTagPersonResponse:
        return await app.state.runtime.smriti_tag_person(payload)

    @app.get("/v1/smriti/person/{person_name}/journal", dependencies=[Depends(require_auth)])
    async def smriti_person_journal(person_name: str) -> dict:
        return app.state.runtime.smriti_person_journal(person_name)

    @app.get("/v1/smriti/clusters", dependencies=[Depends(require_auth)])
    async def smriti_clusters() -> dict:
        return app.state.runtime.smriti_clusters()

    @app.get("/v1/smriti/metrics", dependencies=[Depends(require_auth)])
    async def smriti_metrics() -> dict:
        return app.state.runtime.smriti_metrics()

    @app.get("/v1/smriti/storage", dependencies=[Depends(require_auth)])
    async def get_smriti_storage() -> SmritiStorageConfig:
        return app.state.runtime.get_smriti_storage_config()

    @app.put("/v1/smriti/storage", dependencies=[Depends(require_auth)])
    async def update_smriti_storage(config: SmritiStorageConfig) -> SmritiStorageConfig:
        return app.state.runtime.update_smriti_storage_config(config)

    @app.get("/v1/smriti/storage/usage", dependencies=[Depends(require_auth)])
    async def get_storage_usage() -> StorageUsageReport:
        return app.state.runtime.get_storage_usage()

    @app.get("/v1/smriti/watch-folders", dependencies=[Depends(require_auth)])
    async def list_watch_folders() -> list[WatchFolderStatus]:
        resolved = app.state.runtime.get_smriti_storage_config()
        return [
            app.state.runtime.get_watch_folder_status(folder_path)
            for folder_path in resolved.watch_folders
        ]

    @app.post("/v1/smriti/watch-folders", dependencies=[Depends(require_auth)])
    async def add_watch_folder(payload: dict) -> WatchFolderStatus:
        folder_path = payload.get("path", "")
        if not folder_path:
            raise HTTPException(status_code=422, detail="path is required")
        return app.state.runtime.add_watch_folder(folder_path)

    @app.delete("/v1/smriti/watch-folders", dependencies=[Depends(require_auth)])
    async def remove_watch_folder(path: str = Query(...)) -> dict[str, str]:
        app.state.runtime.remove_watch_folder(path)
        return {"removed": path}

    @app.post("/v1/smriti/storage/prune", dependencies=[Depends(require_auth)])
    async def prune_storage(payload: SmritiPruneRequest) -> SmritiPruneResult:
        return app.state.runtime.prune_smriti_storage(payload)

    @app.post("/v1/smriti/storage/migrate", dependencies=[Depends(require_auth)])
    async def migrate_smriti_storage(payload: SmritiMigrationRequest) -> SmritiMigrationResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, app.state.runtime.migrate_smriti_data, payload)

    return app
