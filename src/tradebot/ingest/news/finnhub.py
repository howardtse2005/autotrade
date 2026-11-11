"""Finnhub company-news connector (keyed)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from tradebot.core.clock import now_utc
from tradebot.core.types import RawNewsItem, UniverseRow
from tradebot.ingest.news.base import BaseConnector, clean_text, item_hash, parse_dt

API = "https://finnhub.io/api/v1"


class FinnhubNews(BaseConnector):
    name = "finnhub"

    def __init__(self, api_key: str, weight: float = 1.0, rpm: int = 55,
                 max_tickers: int | None = 30) -> None:
        super().__init__(name="finnhub", weight=weight, rpm=rpm, max_tickers=max_tickers)
        self.api_key = api_key
        self.enabled_flag = bool(api_key)

    async def fetch(self, universe: list[UniverseRow], since: datetime) -> list[RawNewsItem]:
        if not self.api_key:
            return []
        rows = self.tickers(universe)
        first_seen = now_utc()
        date_from = (since - timedelta(days=1)).strftime("%Y-%m-%d")
        date_to = now_utc().strftime("%Y-%m-%d")
        results: list[RawNewsItem] = []
        seen: set[str] = set()

        async def one(row: UniverseRow) -> list[RawNewsItem]:
            params = {
                "symbol": row.ticker.replace(".", "-"),
                "from": date_from,
                "to": date_to,
                "token": self.api_key,
            }
            try:
                data = await self.get_json(f"{API}/company-news", params)
            except Exception:  # noqa: BLE001
                return []
            if not isinstance(data, list):
                return []
            out = []
            for it in data[:25]:
                title = clean_text(it.get("headline", ""), 300)
                if not title:
                    continue
                ts = parse_dt(it.get("datetime"), first_seen)
                if ts < since:
                    continue
                h = item_hash(row.ticker, title, it.get("url", ""))
                if h in seen:
                    continue
                seen.add(h)
                out.append(
                    RawNewsItem(
                        ticker=row.ticker,
                        title=title,
                        snippet=clean_text(it.get("summary", ""), 400),
                        url=it.get("url", ""),
                        source=it.get("source", "Finnhub"),
                        source_type=self.name,
                        published_at=ts,
                        first_seen_at=first_seen,
                        relevance=0.9,
                        item_id=h,
                        raw={"category": it.get("category", "")},
                    )
                )
            return out

        chunks = [rows[i : i + 10] for i in range(0, len(rows), 10)]
        for chunk in chunks:
            batch = await asyncio.gather(*(one(r) for r in chunk))
            for b in batch:
                results.extend(b)
        return results
