"""News connector framework: rate limits, retries, circuit breakers, health."""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

import httpx

from tradebot.core.clock import now_utc, to_utc
from tradebot.core.logging import get_logger
from tradebot.core.types import RawNewsItem, UniverseRow

log = get_logger("news")


@dataclass
class ConnectorHealthState:
    name: str
    last_success: datetime | None = None
    last_error: str | None = None
    last_error_at: datetime | None = None
    latency_ms: int = 0
    items_last_run: int = 0
    consecutive_failures: int = 0
    breaker_open_until: datetime | None = None
    total_runs: int = 0
    total_items: int = 0
    enabled: bool = True
    skipped: bool = False

    def breaker_open(self) -> bool:
        return self.breaker_open_until is not None and now_utc() < self.breaker_open_until

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_error": self.last_error,
            "latency_ms": self.latency_ms,
            "items_last_run": self.items_last_run,
            "consecutive_failures": self.consecutive_failures,
            "breaker_open": self.breaker_open(),
            "total_runs": self.total_runs,
            "total_items": self.total_items,
            "enabled": self.enabled,
            "skipped": self.skipped,
        }


class NewsConnector(Protocol):
    name: str

    def enabled(self) -> bool: ...

    async def fetch(self, universe: list[UniverseRow], since: datetime) -> list[RawNewsItem]: ...


def item_hash(ticker: str | None, title: str, url: str) -> str:
    key = f"{(ticker or '')}|{(title or '').strip().lower()}|{(url or '').strip()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def normalize_title(title: str) -> str:
    t = (title or "").lower()
    t = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in t)
    return " ".join(t.split())


@dataclass
class BaseConnector:
    name: str = "base"
    weight: float = 0.8
    rpm: int = 30
    max_tickers: int | None = None
    timeout: float = 12.0
    max_retries: int = 2
    enabled_flag: bool = True

    health: ConnectorHealthState = field(init=False)
    _limiter: Any = field(init=False, default=None)
    _client: httpx.AsyncClient | None = field(init=False, default=None)
    _sem: asyncio.Semaphore | None = field(init=False, default=None)
    _etags: dict[str, str] = field(init=False, default_factory=dict)
    _cached: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.health = ConnectorHealthState(name=self.name, enabled=self.enabled_flag)

    def enabled(self) -> bool:
        return self.enabled_flag and not self.health.breaker_open()

    # ------------------------------------------------------------------ http
    async def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                  "(KHTML, like Gecko) Chrome/125.0 Safari/537.36",
                    "Accept": "application/rss+xml, application/xml, application/json, text/xml;q=0.9, */*;q=0.8",
                },
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def limiter(self):
        if self._limiter is None:
            from aiolimiter import AsyncLimiter

            self._limiter = AsyncLimiter(max(self.rpm, 1), 60)
        return self._limiter

    async def semaphore(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(8)
        return self._sem

    async def get_text(self, url: str, etag_key: str | None = None) -> str | None:
        lim = await self.limiter()
        sem = await self.semaphore()
        async with sem:
            for attempt in range(self.max_retries + 1):
                try:
                    async with lim:
                        client = await self.client()
                        headers = {}
                        if etag_key and etag_key in self._etags:
                            headers["If-None-Match"] = self._etags[etag_key]
                        resp = await client.get(url, headers=headers)
                        if resp.status_code == 304 and etag_key:
                            return self._cached.get(etag_key)
                        if resp.status_code == 429:
                            await asyncio.sleep(2 + attempt * 3)
                            continue
                        resp.raise_for_status()
                        text = resp.text
                        if etag_key:
                            et = resp.headers.get("etag")
                            if et:
                                self._etags[etag_key] = et
                                self._cached[etag_key] = text
                        return text
                except Exception as exc:  # noqa: BLE001
                    if attempt >= self.max_retries:
                        raise
                    await asyncio.sleep(1.5 * (attempt + 1))
                    log.debug("retrying", connector=self.name, url=url[:80], error=str(exc)[:100])
        return None

    async def get_json(self, url: str, params: dict | None = None) -> Any:
        lim = await self.limiter()
        sem = await self.semaphore()
        async with sem:
            for attempt in range(self.max_retries + 1):
                try:
                    async with lim:
                        client = await self.client()
                        resp = await client.get(url, params=params)
                        if resp.status_code == 429:
                            await asyncio.sleep(3 + attempt * 5)
                            continue
                        resp.raise_for_status()
                        return resp.json()
                except Exception:
                    if attempt >= self.max_retries:
                        raise
                    await asyncio.sleep(1.5 * (attempt + 1))
        return None

    # -------------------------------------------------------------- lifecycle
    def mark_success(self, items: int, latency_ms: int) -> None:
        self.health.last_success = now_utc()
        self.health.items_last_run = items
        self.health.latency_ms = latency_ms
        self.health.consecutive_failures = 0
        self.health.breaker_open_until = None
        self.health.total_runs += 1
        self.health.total_items += items
        self.health.skipped = False

    def mark_failure(self, error: str) -> None:
        self.health.last_error = error[:400]
        self.health.last_error_at = now_utc()
        self.health.consecutive_failures += 1
        self.health.total_runs += 1
        if self.health.consecutive_failures >= 5:
            self.health.breaker_open_until = now_utc() + timedelta(minutes=10)
            log.warning("connector breaker opened", connector=self.name,
                        failures=self.health.consecutive_failures)

    def tickers(self, universe: list[UniverseRow]) -> list[UniverseRow]:
        if self.max_tickers:
            return universe[: self.max_tickers]
        return universe


def parse_dt(value: Any, fallback: datetime | None = None) -> datetime:
    """Robust RFC-822 / ISO timestamp parsing -> UTC."""
    fallback = fallback or now_utc()
    if value is None:
        return fallback
    if isinstance(value, datetime):
        return to_utc(value)
    if isinstance(value, (int, float)):
        try:
            return to_utc(datetime.fromtimestamp(float(value), tz=None).astimezone())
        except Exception:  # noqa: BLE001
            return fallback
    text = str(value).strip()
    if not text:
        return fallback
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(text)
        if dt is not None:
            return to_utc(dt)
    except Exception:  # noqa: BLE001
        pass
    try:
        import pandas as pd

        ts = pd.Timestamp(text)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return to_utc(ts.to_pydatetime())
    except Exception:  # noqa: BLE001
        return fallback


def clean_text(text: str, limit: int = 500) -> str:
    """Strip HTML/control chars and cap length."""
    import re

    if not text:
        return ""
    t = re.sub(r"<[^>]+>", " ", str(text))
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:limit]
