"""Ticker relevance scoring for headlines and snippets."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tradebot.core.symbols import AMBIGUOUS, company_aliases
from tradebot.core.types import UniverseRow

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9&.\-]{0,30}")


@dataclass
class TickerMatcher:
    ticker: str
    aliases: list[str] = field(default_factory=list)
    ambiguous: bool = False

    def _alias_re(self, alias: str) -> re.Pattern:
        return re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.I)

    def score(self, title: str, snippet: str) -> tuple[float, str]:
        """Return (relevance 0..1, where) for a single item."""
        t = title or ""
        s = snippet or ""
        cash = re.search(rf"\${re.escape(self.ticker)}\b", t, re.I)
        if cash:
            return 1.0, "cashtag"
        # whole-word ticker (uppercase only, to avoid matching prose)
        if not self.ambiguous:
            if re.search(rf"\b{re.escape(self.ticker)}\b", t):
                return 1.0, "ticker_title"
        for alias in self.aliases:
            if self._alias_re(alias).search(t):
                return 1.0, "company_title"
        if not self.ambiguous:
            if re.search(rf"\b{re.escape(self.ticker)}\b", s):
                return 0.6, "ticker_snippet"
        for alias in self.aliases:
            if self._alias_re(alias).search(s):
                return 0.6, "company_snippet"
        return 0.0, ""


def build_matchers(universe: list[UniverseRow]) -> dict[str, TickerMatcher]:
    out: dict[str, TickerMatcher] = {}
    for row in universe:
        aliases = company_aliases(row.description or row.name)
        out[row.ticker] = TickerMatcher(
            ticker=row.ticker,
            aliases=aliases,
            ambiguous=row.ticker in AMBIGUOUS or len(row.ticker) <= 2,
        )
    return out


def match_item(
    title: str, snippet: str, matchers: dict[str, TickerMatcher], bulk: bool = False
) -> list[tuple[str, float]]:
    """All tickers mentioned, with relevance; multi-ticker headlines share weight."""
    hits: list[tuple[str, float, str]] = []
    for ticker, m in matchers.items():
        sc, where = m.score(title, snippet)
        if sc > 0:
            hits.append((ticker, sc, where))
    if not hits:
        return []
    title_hits = [h for h in hits if h[2].endswith("title") or h[2] == "cashtag"]
    pool = title_hits or hits
    share = 1.0 / max(1, len(pool))
    out = []
    for ticker, sc, _where in pool:
        relevance = sc * (share if len(pool) > 1 else 1.0)
        if bulk:
            relevance *= 0.7
        out.append((ticker, round(max(0.05, min(1.0, relevance)), 3)))
    return out
