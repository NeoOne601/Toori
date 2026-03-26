from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, make_asgi_app

from .service import RuntimeContainer
from .models import (
    AnalyzeRequest,
    ChallengeEvaluateRequest,
    JEPAForecastRequest,
    LivingLensTickRequest,
    ProofReportGenerateRequest,
    QueryRequest,
    RuntimeSettings,
)


def create_app(data_dir: str | None = None) -> FastAPI:
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
    app = FastAPI(title="Toori Runtime", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/metrics", make_asgi_app(registry=registry))
    app.state.runtime = RuntimeContainer(data_dir=data_dir)

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

    @app.get("/v1/settings", response_model=RuntimeSettings, dependencies=[Depends(require_auth)])
    def get_settings() -> RuntimeSettings:
        return app.state.runtime.get_settings()

    @app.put("/v1/settings", response_model=RuntimeSettings, dependencies=[Depends(require_auth)])
    def put_settings(settings: RuntimeSettings) -> RuntimeSettings:
        return app.state.runtime.update_settings(settings)

    @app.get("/v1/providers/health", dependencies=[Depends(require_auth)])
    def provider_health():
        return app.state.runtime.provider_health()

    @app.get("/v1/observations", dependencies=[Depends(require_auth)])
    def observations(
        session_id: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
    ):
        return app.state.runtime.list_observations(session_id=session_id, limit=limit)

    @app.get("/v1/file", dependencies=[Depends(require_auth)])
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

    @app.post("/v1/analyze", dependencies=[Depends(require_auth)])
    def analyze(payload: AnalyzeRequest):
        analyze_counter.inc()
        with analyze_latency.time():
            return app.state.runtime.analyze(payload)

    @app.post("/v1/living-lens/tick", dependencies=[Depends(require_auth)])
    def living_lens_tick(payload: LivingLensTickRequest):
        analyze_counter.inc()
        with analyze_latency.time():
            return app.state.runtime.living_lens_tick(payload)

    @app.post("/v1/jepa/forecast", dependencies=[Depends(require_auth)])
    def jepa_forecast(
        payload: JEPAForecastRequest,
        k: int | None = Query(default=None, ge=1, le=32),
    ):
        horizon = k if k is not None else payload.k
        return app.state.runtime.forecast_jepa(payload.session_id, horizon)

    @app.post("/v1/query", dependencies=[Depends(require_auth)])
    def query(payload: QueryRequest):
        return app.state.runtime.query(payload)

    @app.get("/v1/world-state", dependencies=[Depends(require_auth)])
    def world_state(session_id: str = Query(..., min_length=1)):
        return app.state.runtime.get_world_state(session_id)

    @app.post("/v1/challenges/evaluate", dependencies=[Depends(require_auth)])
    def evaluate_challenge(payload: ChallengeEvaluateRequest):
        return app.state.runtime.evaluate_challenge(payload)

    @app.post("/v1/proof-report/generate", dependencies=[Depends(require_auth)])
    def generate_proof_report(payload: ProofReportGenerateRequest):
        return app.state.runtime.generate_proof_report(payload.session_id, payload.chart_b64)

    @app.get("/v1/proof-report/latest", dependencies=[Depends(require_auth)])
    def latest_proof_report():
        latest = app.state.runtime.latest_proof_report()
        if latest is None or not latest.exists():
            raise HTTPException(status_code=404, detail="Proof report not found")
        return StreamingResponse(
            iter([latest.read_bytes()]),
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{latest.name}"'},
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

    return app
