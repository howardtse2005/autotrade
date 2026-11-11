"""News orchestration: parallel connectors, dedupe, relevance, persistence."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import select

from tradebot.core.clock import now_utc
from tradebot.core.logging import get_logger
from tradebot.core.types import RawNewsItem, UniverseRow
from tradebot.db.models import NewsItem
from tradebot.db.session import session_scope
from tradebot.ingest.news.base import BaseConnector
from tradebot.ingest.news.dedupe import Deduplicator
from tradebot.ingest.news.finnhub import FinnhubNews
from tradebot.ingest.news.google_rss import GoogleNewsRSS
from tradebot.ingest.news.marketaux import MarketauxNews
from tradebot.ingest.news.relevance import build_matchers, match_item
from tradebot.ingest.news.rss_generic import GenericRSS
from tradebot.ingest.news.yahoo import YahooNews

log = get_logger("news.service")


@dataclass
class NewsBatch:
    items: list[RawNewsItem] = field(default_factory=list)
    by_ticker: dict[str, list[RawNewsItem]] = field(default_factory=dict)
    coverage: dict[str, int] = field(default_factory=dict)
    health: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    total_fetched: int = 0
    duplicates: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def coverage_pct(self) -> float:
        if not self.coverage:
            return 0.0
        return 100.0 * sum(1 for v in self.coverage.values() if v > 0) / len(self.coverage)


class NewsService:
    def __init__(self, cfg, env) -> None:
        self.cfg = cfg
        self.env = env
        self.connectors: list[BaseConnector] = self._build()
        self.dedup = Deduplicator(threshold=cfg.news.dedupe_similarity)

    def _cfg_for(self, name: str) -> dict:
        raw = (self.cfg.news.connectors or {}).get(name) or {}
        return raw if isinstance(raw, dict) else {}

    def _enabled(self, name: str, has_key: bool = True) -> bool:
        val = self._cfg_for(name).get("enabled", True)
        if val == "auto":
            return has_key
        return bool(val)

    def _build(self) -> list[BaseConnector]:
        out: list[BaseConnector] = []
        c = self.cfg.news.connectors
        if self._enabled("google_news_rss"):
            out.append(
                GoogleNewsRSS(
                    weight=float(self._cfg_for("google_news_rss").get("weight", 0.8)),
                    rpm=int(self._cfg_for("google_news_rss").get("rpm", 120)),
                )
            )
        if self._enabled("finnhub", bool(self.env.FINNHUB_API_KEY)):
            out.append(
                FinnhubNews(
                    self.env.FINNHUB_API_KEY,
                    weight=float(self._cfg_for("finnhub").get("weight", 1.0)),
                    rpm=int(self._cfg_for("finnhub").get("rpm", 55)),
                )
            )
        if self._enabled("marketaux", bool(self.env.MARKETAUX_API_KEY)):
            out.append(
                MarketauxNews(
                    self.env.MARKETAUX_API_KEY,
                    weight=float(self._cfg_for("marketaux").get("weight", 0.8)),
                    rpm=int(self._cfg_for("marketaux").get("rpm", 5)),
                )
            )
        if self._enabled("yahoo_finance"):
            out.append(
                YahooNews(weight=float(self._cfg_for("yahoo_finance").get("weight", 0.8)))
            )
        if self._enabled("generic_rss") and self.cfg.news.generic_feeds:
            out.append(
                GenericRSS(
                    self.cfg.news.generic_feeds,
                    weight=float(self._cfg_for("generic_rss").get("weight", 0.7)),
                    rpm=int(self._cfg_for("generic_rss").get("rpm", 20)),
                )
            )
        del c
        return out

    async def close(self) -> None:
        await asyncio.gather(*(c.close() for c in self.connectors), return_exceptions=True)

    async def collect(self, universe: list[UniverseRow], run_id: int | None = None) -> NewsBatch:
        since = now_utc() - timedelta(hours=self.cfg.news.lookback_hours)
        batch = NewsBatch(coverage={r.ticker: 0 for r in universe})
        t0 = time.time()
        matchers = build_matchers(universe)

        async def run(conn: BaseConnector):
            start = time.monotonic()
            try:
                if not conn.enabled():
                    conn.health.skipped = True
                    return []
                if isinstance(conn, GenericRSS):
                    items = await conn.fetch(universe, since, matchers=matchers)
                else:
                    items = await conn.fetch(universe, since)
                conn.mark_success(len(items), int((time.monotonic() - start) * 1000))
                return items
            except Exception as exc:  # noqa: BLE001
                conn.mark_failure(f"{type(exc).__name__}: {exc}")
                batch.errors.append(f"{conn.name}: {str(exc)[:120]}")
                log.warning("connector failed", connector=conn.name, error=str(exc)[:200])
                return []

        results = await asyncio.gather(*(run(c) for c in self.connectors), return_exceptions=True)
        merged: list[RawNewsItem] = []
        for res in results:
            if isinstance(res, list):
                merged.extend(res)
        batch.total_fetched = len(merged)

        # relevance for items whose connector did not pre-assign a ticker
        scored: list[RawNewsItem] = []
        for item in merged:
            if item.ticker is None:
                hits = match_item(item.title, item.snippet, matchers, bulk=False)
                for ticker, rel in hits[:3]:
                    clone = RawNewsItem(
                        ticker=ticker, title=item.title, snippet=item.snippet, url=item.url,
                        source=item.source, source_type=item.source_type,
                        published_at=item.published_at, first_seen_at=item.first_seen_at,
                        relevance=rel, raw=item.raw, item_id=None,
                    )
                    scored.append(clone)
            else:
                if item.item_id is None:
                    from tradebot.ingest.news.base import item_hash

                    item.item_id = item_hash(item.ticker, item.title, item.url)
                scored.append(item)

        # drop blocked domains and empty titles
        blocked = set(self.cfg.news.blocked_domains or [])
        if blocked:
            scored = [i for i in scored if not any(b in i.url for b in blocked)]

        unique = self.dedup.assign(scored)
        batch.duplicates = max(0, len(scored) - len(unique))

        by_ticker: dict[str, list[RawNewsItem]] = {}
        for item in unique:
            if not item.ticker:
                continue
            by_ticker.setdefault(item.ticker, []).append(item)
        cap = self.cfg.news.max_items_per_ticker
        for t, items in by_ticker.items():
            items.sort(key=lambda i: i.published_at, reverse=True)
            by_ticker[t] = items[:cap]
        batch.by_ticker = by_ticker
        batch.items = [i for items in by_ticker.values() for i in items]
        for t in batch.coverage:
            batch.coverage[t] = len(by_ticker.get(t, []))
        batch.health = [c.health.as_dict() for c in self.connectors]
        batch.elapsed_s = round(time.time() - t0, 2)

        if run_id is not None:
            await self.persist(batch, run_id)
        log.info("news collected", items=len(batch.items), coverage=round(batch.coverage_pct, 1),
                 seconds=batch.elapsed_s)
        return batch

    async def persist(self, batch: NewsBatch, run_id: int) -> None:
        if not batch.items:
            return
        async with session_scope() as s:
            hashes = [i.item_id for i in batch.items if i.item_id]
            existing: set[str] = set()
            if hashes:
                for chunk in [hashes[i : i + 400] for i in range(0, len(hashes), 400)]:
                    rows = await s.execute(select(NewsItem.item_hash).where(NewsItem.item_hash.in_(chunk)))
                    existing.update(r[0] for r in rows.all())
            for item in batch.items:
                if not item.item_id or item.item_id in existing:
                    continue
                s.add(
                    NewsItem(
                        item_hash=item.item_id,
                        run_id=run_id,
                        ticker=item.ticker,
                        title=item.title,
                        snippet=item.snippet,
                        url=item.url,
                        source=item.source,
                        source_type=item.source_type,
                        published_at=item.published_at,
                        first_seen_at=item.first_seen_at,
                        relevance=item.relevance,
                        cluster_id=item.cluster_id or "",
                        cluster_size=item.cluster_size,
                        raw=item.raw or None,
                    )
                )
                existing.add(item.item_id)

    async def recent(self, tickers: list[str] | None = None, limit: int = 200) -> list[NewsItem]:
        async with session_scope() as s:
            q = select(NewsItem).order_by(NewsItem.published_at.desc()).limit(limit)
            if tickers:
                q = q.where(NewsItem.ticker.in_(tickers))
            res = await s.execute(q)
            return list(res.scalars().all())

    def health_snapshot(self) -> list[dict]:
        return [c.health.as_dict() for c in self.connectors]
