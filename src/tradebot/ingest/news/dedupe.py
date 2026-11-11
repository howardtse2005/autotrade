"""Near-duplicate news clustering (title normalization + rapidfuzz)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from rapidfuzz import fuzz

from tradebot.core.types import RawNewsItem
from tradebot.ingest.news.base import normalize_title


@dataclass
class Cluster:
    cluster_id: str
    norm_title: str
    first_seen: datetime
    size: int = 1
    tickers: set[str] = field(default_factory=set)


class Deduplicator:
    def __init__(self, threshold: int = 90, window_hours: int = 24) -> None:
        self.threshold = threshold
        self.window = timedelta(hours=window_hours)
        self.clusters: list[Cluster] = []

    def assign(self, items: list[RawNewsItem]) -> list[RawNewsItem]:
        """Mutates items with cluster_id / cluster_size; returns unique representatives."""
        unique: list[RawNewsItem] = []
        for item in items:
            nt = normalize_title(item.title)
            if not nt:
                continue
            match = self._find(nt, item.published_at)
            if match is not None:
                match.size += 1
                if item.ticker:
                    match.tickers.add(item.ticker)
                item.cluster_id = match.cluster_id
                item.cluster_size = match.size
                continue
            cid = hashlib.sha1(f"{nt}|{item.published_at.date()}".encode()).hexdigest()[:16]
            cluster = Cluster(cluster_id=cid, norm_title=nt, first_seen=item.published_at)
            if item.ticker:
                cluster.tickers.add(item.ticker)
            self.clusters.append(cluster)
            item.cluster_id = cid
            item.cluster_size = 1
            unique.append(item)
        self._prune(items)
        # propagate final cluster sizes to representatives
        size_by_id = {c.cluster_id: c.size for c in self.clusters}
        for item in unique:
            item.cluster_size = size_by_id.get(item.cluster_id or "", 1)
        return unique

    def _find(self, norm_title: str, ts: datetime) -> Cluster | None:
        cutoff = ts - self.window
        for c in reversed(self.clusters):
            if c.first_seen < cutoff:
                break
            if fuzz.token_set_ratio(norm_title, c.norm_title) >= self.threshold:
                return c
        return None

    def _prune(self, items: list[RawNewsItem]) -> None:
        if not self.clusters:
            return
        newest = max((i.published_at for i in items), default=None)
        if newest is None:
            return
        cutoff = newest - self.window * 2
        self.clusters = [c for c in self.clusters if c.first_seen >= cutoff][-4000:]
