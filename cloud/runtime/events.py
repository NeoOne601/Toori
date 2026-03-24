from __future__ import annotations

import asyncio
from typing import List

from .models import EventMessage


class EventBus:
    def __init__(self) -> None:
        self._subscribers: List[asyncio.Queue[EventMessage]] = []

    def subscribe(self) -> asyncio.Queue[EventMessage]:
        queue: asyncio.Queue[EventMessage] = asyncio.Queue(maxsize=100)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[EventMessage]) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def publish(self, event_type: str, payload: dict) -> EventMessage:
        message = EventMessage(type=event_type, payload=payload)
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                _ = queue.get_nowait()
                queue.put_nowait(message)
        return message
