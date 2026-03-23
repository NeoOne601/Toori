from fastapi import FastAPI, Depends, HTTPException, status
from .auth import get_current_user

app = FastAPI()

# Helper functions that will be monkey‑patched in tests

def call_jepa(embedding: list[float]):
    """Call the external JEPA service – placeholder implementation.
    In production this would HTTP‑POST to the JEPA service and return the refined embedding.
    """
    raise NotImplementedError("JEPA service call not implemented")

def call_search(refined_embedding: list[float]):
    """Call the external FAISS search service – placeholder implementation.
    In production this would HTTP‑POST to the search service and return search results.
    """
    raise NotImplementedError("Search service call not implemented")

@app.post("/search")
def search_endpoint(embedding: list[float], user: dict = Depends(get_current_user)):
    """Receive a raw embedding, refine it via JEPA, then search.
    Returns combined result from JEPA and search services.
    """
    refined = call_jepa(embedding)
    results = call_search(refined)
    return {"refined": refined, "results": results}


@app.get("/protected")
def protected_route(user: dict = Depends(get_current_user)):
    return {"message": f"Hello, {user.get('sub', 'anonymous')}!"}
