"""In-process pub/sub used to feed the SSE stream and the alerts feed."""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from tradebot.core.clock import now_utc


@dataclass(slots=True)
class Event:
    topic: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=now_utc)

    def as_dict(self) -> dict[str, Any]:
        return {"topic": self.topic, "ts": self.ts.isoformat(), **self.payload}


class EventBus:
    """Fan-out to async subscribers with bounded queues (slow clients are dropped)."""

    def __init__(self, history: int = 300, queue_size: int = 200) -> None:
        self._subs: set[asyncio.Queue[Event]] = set()
        self._history: deque[Event] = deque(maxlen=history)
        self._queue_size = queue_size
        self._lock = asyncio.Lock()

    async def publish(self, topic: str, **payload: Any) -> None:
        ev = Event(topic=topic, payload=payload)
        self._history.append(ev)
        for q in list(self._subs):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                with contextlib.suppress(Exception):
                    q.get_nowait()
                with contextlib.suppress(Exception):
                    q.put_nowait(ev)

    def publish_nowait(self, topic: str, **payload: Any) -> None:
        ev = Event(topic=topic, payload=payload)
        self._history.append(ev)
        for q in list(self._subs):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                pass

    async def subscribe(self) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._queue_size)
        async with self._lock:
            self._subs.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[Event]) -> None:
        async with self._lock:
            self._subs.discard(q)

    def recent(self, limit: int = 50) -> list[Event]:
        return list(self._history)[-limit:]

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)


bus = EventBus()
