"""Compatibility service exposing the perception provider as /embed."""

from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import CollectorRegistry, Gauge, make_asgi_app

from cloud.runtime.service import RuntimeContainer


class EmbedRequest(BaseModel):
    image_base64: str | None = None
    file_path: str | None = None


class EmbedResponse(BaseModel):
    embedding: list[float]
    provider: str


def create_perception_app(data_dir: str | None = None) -> FastAPI:
    registry = CollectorRegistry()
    process_cpu_seconds_total = Gauge(
        "process_cpu_seconds_total",
        "CPU seconds total",
        registry=registry,
    )
    process_cpu_seconds_total.set(0.0)

    app = FastAPI(title="Toori Perception Service")
    app.mount("/metrics", make_asgi_app(registry=registry))
    app.state.runtime = RuntimeContainer(data_dir=data_dir)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/embed", response_model=EmbedResponse)
    def embed_endpoint(req: EmbedRequest) -> EmbedResponse:
        _, image = app.state.runtime._load_image(req.image_base64, req.file_path)
        settings = app.state.runtime.get_settings()
        embedding, provider, _, _ = app.state.runtime.providers.perceive(settings, image)
        return EmbedResponse(embedding=embedding, provider=provider)

    return app


app = create_perception_app()
