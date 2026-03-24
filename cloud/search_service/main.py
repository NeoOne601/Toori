"""FastAPI application exposing a FAISS search service.

The service loads an index using :func:`load_index` from
``index_builder``. For this skeleton the index is ``None``, meaning there
are no vectors to search against. The ``/search`` endpoint therefore
returns an empty list for any query.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from .index_builder import load_index

from prometheus_client import make_asgi_app

from prometheus_client import make_asgi_app, Gauge, CollectorRegistry

# Create a dedicated registry for this service to avoid metric name collisions
_search_registry = CollectorRegistry()
process_cpu_seconds_total = Gauge('process_cpu_seconds_total', 'CPU seconds total', registry=_search_registry)
process_cpu_seconds_total.set(0.0)

app = FastAPI()

# Mount Prometheus metrics endpoint using the dedicated registry
app.mount("/metrics", make_asgi_app(registry=_search_registry))


# Load the index at startup (placeholder)
index = load_index()

class SearchRequest(BaseModel):
    query: str
    k: int = 10

@app.post("/search", response_model=List[int])
async def search(req: SearchRequest):
    """Return top‑k IDs matching *query*.

    Since the placeholder index is empty, we always return an empty list.
    """
    if index is None:
        return []
    # In a real implementation we would query the FAISS index here.
    raise HTTPException(status_code=501, detail="Search not implemented for non‑empty index")
