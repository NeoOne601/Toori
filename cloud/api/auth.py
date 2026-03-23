import os
from fastapi import Header, HTTPException, status
from jose import jwt, JWTError
import httpx
from typing import Optional

# Placeholder JWKS URL; in real scenario use Auth0 JWKS endpoint
JWKS_URL = os.getenv("JWKS_URL", "https://example.com/.well-known/jwks.json")

# Cache JWKS
_jwks_cache: Optional[dict] = None

async def get_jwks():
    global _jwks_cache
    if _jwks_cache is None:
        async with httpx.AsyncClient() as client:
            resp = await client.get(JWKS_URL)
            resp.raise_for_status()
            _jwks_cache = resp.json()
    return _jwks_cache

async def get_public_key(token: str):
    # Simplified: get kid from token header and find matching key
    try:
        unverified_header = jwt.get_unverified_header(token)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token header")
    kid = unverified_header.get("kid")
    jwks = await get_jwks()
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return jwt.algorithms.RSAAlgorithm.from_jwk(key)
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Public key not found")

async def verify_jwt(token: str):
    public_key = await get_public_key(token)
    try:
        payload = jwt.decode(token, public_key, algorithms=["RS256"], audience=os.getenv("API_AUDIENCE"))
        return payload
    except JWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

async def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth scheme")
    payload = await verify_jwt(token)
    return payload
