from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import CollectorRegistry, Gauge, make_asgi_app

from cloud.runtime.service import RuntimeContainer

from .index_builder import load_index


class SearchRequest(BaseModel):
    observation_id: str | None = None
    query: str | None = None
    k: int = 10


def create_search_app(data_dir: str | None = None) -> FastAPI:
    registry = CollectorRegistry()
    process_cpu_seconds_total = Gauge(
        "process_cpu_seconds_total",
        "CPU seconds total",
        registry=registry,
    )
    process_cpu_seconds_total.set(0.0)

    app = FastAPI(title="Toori Search Service")
    app.mount("/metrics", make_asgi_app(registry=registry))
    app.state.runtime = RuntimeContainer(data_dir=data_dir)
    app.state.index = load_index()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "index": app.state.index.name}

    @app.post("/search")
    async def search(req: SearchRequest):
        runtime = app.state.runtime
        if req.observation_id:
            observation = runtime.store.get_observation(req.observation_id)
            if observation is None:
                return {"hits": []}
            hits = runtime.store.search_by_vector(
                observation.embedding,
                top_k=req.k,
                session_id=observation.session_id,
                exclude_id=observation.id,
            )
            return {"hits": [hit.model_dump(mode="json") for hit in hits]}
        if req.query:
            hits = runtime.store.search_by_text(req.query, top_k=req.k)
            return {"hits": [hit.model_dump(mode="json") for hit in hits]}
        return {"hits": []}

    return app


app = create_search_app()
