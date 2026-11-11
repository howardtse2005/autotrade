"""Google News RSS connector (per-ticker search)."""

from __future__ import annotations

import asyncio
from datetime import datetime
from urllib.parse import quote_plus

from tradebot.core.clock import now_utc
from tradebot.core.types import RawNewsItem, UniverseRow
from tradebot.ingest.news.base import BaseConnector, clean_text, item_hash, parse_dt

URL = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def _parse(xml: str) -> list[dict]:
    import feedparser

    feed = feedparser.parse(xml)
    out = []
    for e in feed.entries[:20]:
        out.append(
            {
                "title": e.get("title", ""),
                "link": e.get("link", ""),
                "published": e.get("published") or e.get("updated"),
                "summary": e.get("summary", ""),
                "source": (e.get("source") or {}).get("title", ""),
            }
        )
    return out


class GoogleNewsRSS(BaseConnector):
    name = "google_news_rss"

    def __init__(self, weight: float = 0.8, rpm: int = 120, max_tickers: int | None = None) -> None:
        super().__init__(name="google_news_rss", weight=weight, rpm=rpm, max_tickers=max_tickers)

    async def fetch(self, universe: list[UniverseRow], since: datetime) -> list[RawNewsItem]:
        rows = self.tickers(universe)
        results: list[RawNewsItem] = []
        seen: set[str] = set()
        first_seen = now_utc()

        async def one(row: UniverseRow) -> list[RawNewsItem]:
            name = (row.description or row.name or row.ticker).split(" - ")[0].split("(")[0].strip()
            q = f'"{name}" OR {row.ticker} stock when:2d'
            url = URL.format(q=quote_plus(q))
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
                source = e.get("source") or "Google News"
                # Google titles are "Headline - Publisher"; drop the suffix
                if source and title.endswith(f" - {source}"):
                    title = title[: -(len(source) + 3)].strip()
                h = item_hash(row.ticker, title, e["link"])
                if h in seen:
                    continue
                seen.add(h)
                out.append(
                    RawNewsItem(
                        ticker=row.ticker,
                        title=title,
                        snippet=clean_text(e["summary"], 400),
                        url=e["link"],
                        source=source,
                        source_type=self.name,
                        published_at=parse_dt(e["published"], first_seen),
                        first_seen_at=first_seen,
                        relevance=0.8,
                        item_id=h,
                        raw={"q": q},
                    )
                )
            return out

        chunks = [rows[i : i + 12] for i in range(0, len(rows), 12)]
        for chunk in chunks:
            batch = await asyncio.gather(*(one(r) for r in chunk))
            for b in batch:
                results.extend(b)
        return results
