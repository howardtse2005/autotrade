"""Marketaux connector (bulk, keyed) — one request per symbol chunk."""

from __future__ import annotations

import asyncio
from datetime import datetime

from tradebot.core.clock import now_utc
from tradebot.core.types import RawNewsItem, UniverseRow
from tradebot.ingest.news.base import BaseConnector, clean_text, item_hash, parse_dt

API = "https://api.marketaux.com/v1/news/all"


class MarketauxNews(BaseConnector):
    name = "marketaux"

    def __init__(self, api_key: str, weight: float = 0.8, rpm: int = 5) -> None:
        super().__init__(name="marketaux", weight=weight, rpm=rpm, max_tickers=None)
        self.api_key = api_key
        self.enabled_flag = bool(api_key)

    async def fetch(self, universe: list[UniverseRow], since: datetime) -> list[RawNewsItem]:
        if not self.api_key:
            return []
        symbols = [r.ticker.replace(".", "-") for r in universe]
        first_seen = now_utc()
        results: list[RawNewsItem] = []
        seen: set[str] = set()

        async def one(chunk: list[str]) -> list[RawNewsItem]:
            params = {
                "symbols": ",".join(chunk),
                "filter_entities": "true",
                "language": "en",
                "limit": 50,
                "api_token": self.api_key,
            }
            try:
                data = await self.get_json(API, params)
            except Exception:  # noqa: BLE001
                return []
            if not isinstance(data, dict):
                return []
            out = []
            for it in data.get("data", []):
                title = clean_text(it.get("title", ""), 300)
                if not title:
                    continue
                ts = parse_dt(it.get("published_at"), first_seen)
                if ts < since:
                    continue
                ents = it.get("entities") or []
                tickers = [e.get("symbol") for e in ents if e.get("symbol")]
                ticker = tickers[0] if tickers else None
                if ticker and ticker not in {r.ticker for r in universe}:
                    ticker = ticker.replace("-", ".")
                h = item_hash(ticker, title, it.get("url", ""))
                if h in seen:
                    continue
                seen.add(h)
                out.append(
                    RawNewsItem(
                        ticker=ticker,
                        title=title,
                        snippet=clean_text(it.get("description") or it.get("snippet") or "", 400),
                        url=it.get("url", ""),
                        source=(it.get("source") or "Marketaux"),
                        source_type=self.name,
                        published_at=ts,
                        first_seen_at=first_seen,
                        relevance=0.75,
                        item_id=h,
                        raw={"entities": [e.get("symbol") for e in ents]},
                    )
                )
            return out

        chunks = [symbols[i : i + 20] for i in range(0, len(symbols), 20)]
        for chunk in chunks:
            results.extend(await one(chunk))
        return results
