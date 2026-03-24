"""Search helpers for the local runtime.

The v1 runtime uses SQLite metadata with brute-force cosine similarity over
stored embeddings. This module exists to keep the service boundary stable so a
future HNSW/Qdrant adapter can replace the implementation without changing the
FastAPI surface.
"""

from dataclasses import dataclass


@dataclass
class LocalSearchIndex:
    name: str = "sqlite-cosine"


def load_index() -> LocalSearchIndex:
    return LocalSearchIndex()
