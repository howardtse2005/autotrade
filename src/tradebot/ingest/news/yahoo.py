"""Yahoo Finance news connector (via yfinance, threaded)."""

from __future__ import annotations

import asyncio
from datetime import datetime

from tradebot.core.clock import now_utc
from tradebot.core.logging import get_logger
from tradebot.core.types import RawNewsItem, UniverseRow
from tradebot.ingest.news.base import BaseConnector, clean_text, item_hash, parse_dt

log = get_logger("news.yahoo")


def _extract(entry: dict) -> dict | None:
    """yfinance has shipped two news schemas; support both."""
    content = entry.get("content") if isinstance(entry, dict) else None
    if isinstance(content, dict):
        provider = content.get("provider") or {}
        url = ""
        canon = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        if isinstance(canon, dict):
            url = canon.get("url", "")
        return {
            "title": content.get("title", ""),
            "summary": content.get("summary") or content.get("description") or "",
            "published": content.get("pubDate") or content.get("displayTime"),
            "source": provider.get("displayName", "Yahoo Finance"),
            "url": url,
        }
    if isinstance(entry, dict) and entry.get("title"):
        return {
            "title": entry.get("title", ""),
            "summary": entry.get("summary", ""),
            "published": entry.get("providerPublishTime"),
            "source": entry.get("publisher", "Yahoo Finance"),
            "url": entry.get("link", ""),
        }
    return None


class YahooNews(BaseConnector):
    name = "yahoo_finance"

    def __init__(self, weight: float = 0.8, rpm: int = 60, max_tickers: int | None = 20) -> None:
        super().__init__(name="yahoo_finance", weight=weight, rpm=rpm, max_tickers=max_tickers)

    def _sync(self, tickers: list[str]) -> dict[str, list[dict]]:
        import yfinance as yf

        out: dict[str, list[dict]] = {}
        for t in tickers:
            try:
                raw = yf.Ticker(t.replace(".", "-")).news or []
            except Exception:  # noqa: BLE001
                continue
            parsed = [p for p in (_extract(e) for e in raw) if p]
            if parsed:
                out[t] = parsed
        return out

    async def fetch(self, universe: list[UniverseRow], since: datetime) -> list[RawNewsItem]:
        rows = self.tickers(universe)
        first_seen = now_utc()
        tickers = [r.ticker for r in rows]
        data = await asyncio.to_thread(self._sync, tickers)
        results: list[RawNewsItem] = []
        for ticker, entries in data.items():
            for e in entries[:15]:
                title = clean_text(e["title"], 300)
                if not title:
                    continue
                ts = parse_dt(e["published"], first_seen)
                if ts < since:
                    continue
                results.append(
                    RawNewsItem(
                        ticker=ticker,
                        title=title,
                        snippet=clean_text(e["summary"], 400),
                        url=e["url"],
                        source=e["source"] or "Yahoo Finance",
                        source_type=self.name,
                        published_at=ts,
                        first_seen_at=first_seen,
                        relevance=0.8,
                        item_id=item_hash(ticker, title, e["url"]),
                    )
                )
        return results
