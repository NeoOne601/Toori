from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx


class TooriClient:
    def __init__(self, base_url: str = "http://127.0.0.1:7777", api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, headers=self._headers(api_key))

    def _headers(self, api_key: str | None) -> dict[str, str]:
        return {"X-API-Key": api_key} if api_key else {}

    def analyze(
        self,
        *,
        file_path: str | None = None,
        image_bytes: bytes | None = None,
        session_id: str = "default",
        query: str | None = None,
        decode_mode: str = "auto",
        top_k: int = 6,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "query": query,
            "decode_mode": decode_mode,
            "top_k": top_k,
        }
        if file_path:
            payload["file_path"] = str(Path(file_path).expanduser())
        elif image_bytes:
            payload["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")
        else:
            raise ValueError("file_path or image_bytes is required")
        return self.client.post("/v1/analyze", json=payload).json()

    def query(self, *, query: str, session_id: str = "default", top_k: int = 6) -> dict[str, Any]:
        return self.client.post(
            "/v1/query",
            json={"query": query, "session_id": session_id, "top_k": top_k},
        ).json()

    def settings(self) -> dict[str, Any]:
        return self.client.get("/v1/settings").json()
