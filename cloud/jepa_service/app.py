"""FastAPI service wrapper for the Llama‑3.2‑JEPA model.

This implementation uses a placeholder model that returns a deterministic
128‑dimensional embedding (all zeros). In a real deployment the model would be
loaded via TorchServe or a native PyTorch pipeline.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from prometheus_client import make_asgi_app

from prometheus_client import make_asgi_app, Gauge, CollectorRegistry

# Create a dedicated registry for this service to avoid metric name collisions
_jepa_registry = CollectorRegistry()
process_cpu_seconds_total = Gauge('process_cpu_seconds_total', 'CPU seconds total', registry=_jepa_registry)
process_cpu_seconds_total.set(0.0)

app = FastAPI()

# Mount Prometheus metrics endpoint using the dedicated registry
app.mount("/metrics", make_asgi_app(registry=_jepa_registry))



class EmbedRequest(BaseModel):
    """Request body for the /embed endpoint."""
    text: str


class EmbedResponse(BaseModel):
    """Response containing the 128‑dimensional embedding."""
    embedding: List[float]


# Placeholder model – in production replace with actual model loading
def _load_model():
    # In a real implementation you would load the TorchServe model here.
    # For now we return a callable that produces a 128‑dim zero vector.
    def embed(_: str) -> List[float]:
        return [0.0] * 128
    return embed


# Load the model once at startup
_model = _load_model()


@app.post("/embed", response_model=EmbedResponse)
def embed_endpoint(req: EmbedRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Text field is required")
    embedding = _model(req.text)
    if len(embedding) != 128:
        raise HTTPException(status_code=500, detail="Model returned embedding of incorrect size")
    return EmbedResponse(embedding=embedding)
