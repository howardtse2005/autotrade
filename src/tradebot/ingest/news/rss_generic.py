"""Generic market RSS feeds, matched to tickers locally."""

from __future__ import annotations

import asyncio
from datetime import datetime

from tradebot.core.clock import now_utc
from tradebot.core.types import RawNewsItem, UniverseRow
from tradebot.ingest.news.base import BaseConnector, clean_text, item_hash, parse_dt
from tradebot.ingest.news.relevance import match_item


def _parse(xml: str) -> list[dict]:
    import feedparser

    feed = feedparser.parse(xml)
    out = []
    for e in feed.entries[:40]:
        out.append(
            {
                "title": e.get("title", ""),
                "link": e.get("link", ""),
                "published": e.get("published") or e.get("updated"),
                "summary": e.get("summary", "") or e.get("description", ""),
            }
        )
    return out


class GenericRSS(BaseConnector):
    name = "generic_rss"

    def __init__(self, feeds: dict[str, str], weight: float = 0.7, rpm: int = 20) -> None:
        super().__init__(name="generic_rss", weight=weight, rpm=rpm, max_tickers=None)
        self.feeds = dict(feeds or {})

    async def fetch(
        self, universe: list[UniverseRow], since: datetime, matchers=None
    ) -> list[RawNewsItem]:
        if matchers is None:
            from tradebot.ingest.news.relevance import build_matchers

            matchers = build_matchers(universe)
        first_seen = now_utc()
        results: list[RawNewsItem] = []

        async def one(name: str, url: str) -> list[RawNewsItem]:
            try:
                xml = await self.get_text(url, etag_key=url)
            except Exception:  # noqa: BLE001
                return []
            if not xml:
                return []
            entries = await asyncio.to_thread(_parse, xml)
            out = []
            for e in entries:
                title = clean_text(e["title"], 300)
                if not title:
                    continue
                ts = parse_dt(e["published"], first_seen)
                if ts < since:
                    continue
                snippet = clean_text(e["summary"], 400)
                hits = match_item(title, snippet, matchers, bulk=True)
                for ticker, rel in hits:
                    out.append(
                        RawNewsItem(
                            ticker=ticker,
                            title=title,
                            snippet=snippet,
                            url=e["link"],
                            source=name,
                            source_type=self.name,
                            published_at=ts,
                            first_seen_at=first_seen,
                            relevance=rel * self.weight,
                            item_id=item_hash(ticker, title, e["link"]),
                        )
                    )
            return out

        batch = await asyncio.gather(*(one(n, u) for n, u in self.feeds.items()))
        for b in batch:
            results.extend(b)
        return results
