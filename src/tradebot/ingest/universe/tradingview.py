"""TradingView screener universe with cache, stale fallback and static watchlist."""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timedelta

import pandas as pd

from tradebot.core.clock import now_utc
from tradebot.core.logging import get_logger
from tradebot.core.symbols import canonical
from tradebot.core.types import UniverseRow
from tradebot.ingest.universe import scoring

log = get_logger("universe")

def _session_progress() -> float:
    """Fraction of the regular session elapsed (0..1); 1.0 when closed."""
    try:
        from tradebot.core.clock import market_clock

        c = market_clock()
        if not c.is_open:
            return 1.0
        total = max(c.minutes_since_open + c.minutes_to_close, 1.0)
        return float(min(max(c.minutes_since_open / total, 1.0 / 26.0), 1.0))
    except Exception:  # noqa: BLE001
        return 1.0


_SELECT = [
    "name", "description", "close", "volume", "change", "relative_volume_10d_calc",
    "market_cap_basic", "average_volume_10d_calc", "sector", "industry", "ATR", "RSI",
    "exchange", "type",
]

# Names that are not plain common stock.
_EXCLUDE_DESC = re.compile(
    r"(?:depositary|preferred|warrant|\bunit(?:s)?\b|\bnote(?:s)?\b|\bdebenture(?:s)?\b|rights|"
    r"exchange traded fund|\betf\b|closed[- ]end|trust preferred|subordinated)",
    re.I,
)
_EXCLUDE_SYMBOL = re.compile(r"^[A-Z]{1,5}(\.?[A-Z]{1,2})?[-.](P|W|U|R|A|B|C)$")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ticker"] = out["ticker"].map(canonical)
    desc = out.get("description", pd.Series("", index=out.index)).fillna("")
    name = out.get("name", pd.Series("", index=out.index)).fillna("")
    bad = desc.str.contains(_EXCLUDE_DESC) | name.str.contains(_EXCLUDE_DESC)
    out = out[~bad]
    # 5-letter symbols ending in a class letter are usually preferred/warrant lines
    out = out[~out["ticker"].str.match(r"^[A-Z]{4,5}(P|W|U|R)$")]
    out = out[~out["ticker"].str.contains(r"\s")]
    out = out.drop_duplicates(subset=["ticker"], keep="first")
    return out


class UniverseService:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._cache: list[UniverseRow] = []
        self._cache_at: datetime | None = None
        self.last_source = "none"
        self.last_error = ""
        self.stale = False
        self._lock = asyncio.Lock()

    async def get(
        self, held: list[str] | None = None, force: bool = False,
        limit: int | None = None,
    ) -> list[UniverseRow]:
        """Return the screened universe, best-ranked first.

        The full pool (up to ``universe.max_candidates``) is cached; ``limit``
        only controls how many names this caller wants back, which is what lets
        the pipeline pull further tranches while it still has empty slots.
        """
        async with self._lock:
            held = [h for h in (held or []) if h]
            n = int(limit if limit is not None else self.cfg.universe.size)
            fresh_enough = (
                self._cache_at is not None
                and now_utc() - self._cache_at < timedelta(minutes=max(1, self.cfg.pipeline.interval_minutes))
            )
            if self._cache and fresh_enough and not force:
                return self._with_held(self._cache[:n], held)

            rows: list[UniverseRow] = []
            try:
                rows = await asyncio.to_thread(self._fetch_tv)
                if rows:
                    self.last_source = "tradingview"
                    self.stale = False
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"[:200]
                log.warning("tradingview failed", error=self.last_error)

            if not rows:
                if self._cache:
                    rows = list(self._cache)
                    self.stale = True
                    self.last_source = "cache(stale)"
                else:
                    rows = self._static_rows()
                    self.stale = True
                    self.last_source = "static"

            if rows:
                self._cache = rows
                self._cache_at = now_utc()
            return self._with_held(rows[:n], held)

    def _with_held(self, rows: list[UniverseRow], held: list[str]) -> list[UniverseRow]:
        if not held or not self.cfg.universe.always_include_positions:
            return rows
        present = {r.ticker for r in rows}
        extra = []
        for h in held:
            if h not in present:
                extra.append(
                    UniverseRow(ticker=h, sector="Unknown", screen_score=-9.0, is_held=True,
                                description="")
                )
        merged = list(rows)
        for r in merged:
            r.is_held = r.ticker in held
        merged.extend(extra)
        return merged

    # ------------------------------------------------------------------ TV
    def _fetch_tv(self) -> list[UniverseRow]:
        from tradingview_screener import Query, col

        u = self.cfg.universe
        q = (
            Query()
            .select(*_SELECT)
            .where(
                col("exchange").isin(u.exchanges),
                col("type") == "stock",
                col("close").between(u.min_price, u.max_price),
                col("market_cap_basic") > u.min_market_cap,
            )
            # NOTE: no hard filter on relative_volume_10d_calc here. It compares
            # *day-cumulative* volume with a 10-day daily average, so during the
            # first hours of the session it is structurally < 0.3 for every name
            # and a "> 1.0" screen returns an empty universe right after the open.
            # Liquidity is screened below in pandas; activity is session-adjusted
            # and used as a score instead of a hard cut.
            .order_by("average_volume_10d_calc", ascending=False)
            .limit(u.candidate_pool)
        )
        t0 = time.time()
        _n, df = q.get_scanner_data()
        if df is None or df.empty:
            raise RuntimeError("empty screener response")
        df = _clean(df)
        df = df[df["close"] >= u.min_price]
        avg_vol = pd.to_numeric(df.get("average_volume_10d_calc", 0), errors="coerce").fillna(0)
        dv = pd.to_numeric(df["close"], errors="coerce") * avg_vol
        df = df[dv >= u.min_avg_dollar_volume]
        if df.empty:
            raise RuntimeError("no rows after filters")

        # Put relative volume on a comparable footing across the session: scale it
        # by how much of the session has elapsed, so "unusually active" means the
        # same thing at 10:30 and at 15:30.
        raw = pd.to_numeric(df["relative_volume_10d_calc"], errors="coerce")
        frac = _session_progress()
        df["relative_volume_raw"] = raw
        df["relative_volume_10d_calc"] = (raw / max(frac, 1.0 / 26.0)).clip(0, 20)
        if u.min_relative_volume and u.min_relative_volume > 0:
            keep = df["relative_volume_10d_calc"] >= u.min_relative_volume
            if keep.sum() >= 10:  # never let the activity cut empty the universe
                df = df[keep]
        if df.empty:
            raise RuntimeError("no rows after filters")
        df = scoring.score_frame(df, u.weights)
        df = df.sort_values("screen_score", ascending=False)
        rows = scoring.to_rows(df)
        picked = scoring.apply_sector_cap(rows, u.max_candidates)
        log.info("universe fetched", rows=len(df), picked=len(picked),
                 seconds=round(time.time() - t0, 2))
        return picked

    # --------------------------------------------------------------- static
    def _static_rows(self) -> list[UniverseRow]:
        tickers = list(self.cfg.universe.static_watchlist) or [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "AVGO", "NFLX",
            "CRM", "ORCL", "ADBE", "QCOM", "INTC", "MU", "PLTR", "COIN", "SHOP", "UBER",
            "ABNB", "SNOW", "PANW", "NOW", "INTU", "AMAT", "LRCX", "KLAC", "ASML", "MRVL",
            "PYPL", "SQ", "ROKU", "SPOT", "DASH", "NET", "DDOG", "ZS", "CRWD", "MDB",
            "TTD", "TEAM", "WDAY", "VEEV", "HUBS", "OKTA", "TWLO", "SNAP", "PINS", "LYFT",
        ]
        rows = [
            UniverseRow(ticker=t, sector="Unknown", screen_score=-float(i))
            for i, t in enumerate(tickers[: self.cfg.universe.size])
        ]
        for i, r in enumerate(rows, start=1):
            r.rank = i
        return rows
