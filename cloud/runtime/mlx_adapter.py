from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from .models import ProviderConfig


def make_mlx_bridge_call(
    provider: Any | None,
    config: ProviderConfig | None,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    async def _call(
        *,
        prompt: str,
        image_base64: str | None = None,
        system: str | None = None,
        max_tokens: int = 256,
    ) -> dict[str, Any]:
        if provider is None or config is None or not bool(getattr(config, "enabled", False)):
            raise RuntimeError("MLX reasoning provider unavailable")

        if hasattr(provider, "aquery"):
            return await provider.aquery(
                prompt=prompt,
                image_base64=image_base64,
                system=system,
                max_tokens=max_tokens,
            )

        if hasattr(provider, "_send_receive"):
            payload = {
                "prompt": prompt,
                "image_base64": image_base64,
                "system": system,
                "max_tokens": int(max_tokens),
            }
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: provider._send_receive(
                    config,
                    payload,
                    timeout_s=float(getattr(config, "timeout_s", 150.0)),
                ),
            )

        if hasattr(provider, "query"):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: provider.query(
                    prompt=prompt,
                    image_base64=image_base64,
                    system=system,
                    max_tokens=max_tokens,
                ),
            )

        raise RuntimeError("MLX provider does not expose a supported bridge interface")

    return _call
