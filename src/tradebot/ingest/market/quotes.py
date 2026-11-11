"""Top-of-book quotes with IBKR-delayed-first, yfinance fallback, and FX rates."""

from __future__ import annotations

import asyncio
import math
import time
from datetime import timedelta

import pandas as pd

from tradebot.core.clock import now_utc, to_utc
from tradebot.core.logging import get_logger
from tradebot.core.types import Quote
from tradebot.ingest.market import store

log = get_logger("market.quotes")


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def _pick(*vals):
    for v in vals:
        if _finite(v):
            return float(v)
    return None


class QuoteService:
    def __init__(self, cfg, ib_client=None, bars=None) -> None:
        self.cfg = cfg
        self.ib_client = ib_client
        self.bars = bars
        self.last_source: str = "none"
        self.last_error: str = ""
        self._fx_cache: dict[str, tuple[float, float]] = {}
        self._quote_cache: dict[str, Quote] = {}

    # ------------------------------------------------------------------ IB
    async def _from_ib(self, tickers: list[str]) -> dict[str, Quote]:
        if self.ib_client is None or not self.ib_client.is_connected:
            return {}
        if not self.ib_client.market_data_ok:
            return {}
        # Skip the whole request when the feed is delayed: the tick would be
        # 15 minutes old, and asking for it spams the log and wastes ~20 s per
        # call. yfinance (next in the order) gives a live price instead.
        if int(getattr(self.ib_client, "market_data_type", 0) or 0) != 1:
            self.last_error = "delayed ib feed - using a live source"
            return {}
        out: dict[str, Quote] = {}
        contracts = []
        for t in tickers:
            c = await self.ib_client.qualify(t)
            if c is not None:
                contracts.append(c)
        if not contracts:
            return {}
        try:
            snaps = await asyncio.wait_for(
                self.ib_client.ib.reqTickersAsync(*contracts), timeout=20
            )
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"[:180]
            log.warning("ib quotes failed", error=self.last_error)
            return {}
        ts = now_utc()
        dropped = 0
        for tk in snaps:
            sym = tk.contract.symbol
            mdt = int(getattr(tk, "marketDataType", 0) or 0)
            # A delayed (15-minute) quote is worse than useless for pricing an
            # order: it looks fresh because we stamp it with "now". Drop it and
            # let the next provider (yfinance) supply a live price.
            if mdt != 1:
                dropped += 1
                continue
            bid = _pick(getattr(tk, "bid", None), getattr(tk, "delayedBid", None))
            ask = _pick(getattr(tk, "ask", None), getattr(tk, "delayedAsk", None))
            last = _pick(getattr(tk, "last", None), getattr(tk, "delayedLast", None),
                         getattr(tk, "close", None), getattr(tk, "delayedClose", None))
            bsz = _pick(getattr(tk, "bidSize", None), getattr(tk, "delayedBidSize", None))
            asz = _pick(getattr(tk, "askSize", None), getattr(tk, "delayedAskSize", None))
            if last is None and bid is None and ask is None:
                continue
            out[sym] = Quote(
                ticker=sym, ts=ts, last=last, bid=bid, ask=ask, bid_size=bsz, ask_size=asz,
                source="ibkr", market_data_type=mdt,
            )
        if dropped:
            log.info("ignored delayed ib quotes", count=dropped)
        if out:
            self.last_source = "ibkr"
        return out

    # ------------------------------------------------------------ yfinance
    async def _from_yf(self, tickers: list[str]) -> dict[str, Quote]:
        if not tickers:
            return {}
        return await asyncio.to_thread(self._yf_sync, tickers)

    def _yf_sync(self, tickers: list[str]) -> dict[str, Quote]:
        import yfinance as yf

        out: dict[str, Quote] = {}
        ts = now_utc()
        for chunk in [tickers[i : i + 30] for i in range(0, len(tickers), 30)]:
            try:
                raw = yf.download(
                    chunk, interval="1m", period="1d", group_by="ticker",
                    auto_adjust=False, progress=False, threads=True, prepost=False,
                )
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"yfinance: {exc}"[:180]
                continue
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                for t in chunk:
                    if t not in raw.columns.get_level_values(0):
                        continue
                    sub = raw[t].dropna(subset=["Close"])
                    if sub.empty:
                        continue
                    px = float(sub["Close"].iloc[-1])
                    vol = float(sub["Volume"].iloc[-1]) if "Volume" in sub else 0.0
                    out[t] = Quote(
                        ticker=t, ts=ts, last=px, bid=None, ask=None,
                        bid_size=None, ask_size=None, source="yfinance", market_data_type=3,
                    )
                    out[t].bid_size = vol  # informational only
            elif len(chunk) == 1:
                sub = raw.dropna(subset=["Close"])
                if not sub.empty:
                    out[chunk[0]] = Quote(
                        ticker=chunk[0], ts=ts, last=float(sub["Close"].iloc[-1]),
                        source="yfinance", market_data_type=3,
                    )
        if out:
            self.last_source = "yfinance"
        return out

    # -------------------------------------------------------------- fallback
    def _from_bars(self, tickers: list[str]) -> dict[str, Quote]:
        out: dict[str, Quote] = {}
        for t in tickers:
            df = self.bars.cached_bars(t) if self.bars else pd.DataFrame()
            if df.empty:
                continue
            last_ts = to_utc(df.index[-1].to_pydatetime())
            out[t] = Quote(
                ticker=t, ts=last_ts, last=float(df["close"].iloc[-1]), source="bar_close",
                market_data_type=3,
            )
        if out:
            self.last_source = "bar_close"
        return out

    async def get_quotes(self, tickers: list[str], allow_stale: bool = False) -> dict[str, Quote]:
        tickers = [t for t in dict.fromkeys(tickers) if t]
        quotes: dict[str, Quote] = {}
        for name in self.cfg.data.quote_provider_order or ["yfinance"]:
            if name == "ibkr":
                quotes = await self._from_ib(tickers)
            elif name == "yfinance":
                quotes = await self._from_yf(tickers)
            if len(quotes) >= max(1, int(len(tickers) * 0.5)):
                break
        missing = [t for t in tickers if t not in quotes]
        if missing:
            fb = self._from_bars(missing)
            quotes.update(fb)
        for t, q in quotes.items():
            if q.last and q.last > 0:
                self._quote_cache[t] = q
        if allow_stale:
            for t in tickers:
                if t not in quotes and t in self._quote_cache:
                    quotes[t] = self._quote_cache[t]
        return quotes

    # ------------------------------------------------------------------- FX
    async def fx_to_usd(self, currency: str) -> float:
        currency = (currency or "USD").upper()
        if currency == "USD":
            return 1.0
        cached = self._fx_cache.get(currency)
        if cached and time.time() - cached[1] < 600:
            return cached[0]
        rate = await asyncio.to_thread(self._fx_sync, currency)
        if rate is None:
            rate = cached[0] if cached else 1.0
        self._fx_cache[currency] = (rate, time.time())
        return rate

    def _fx_sync(self, currency: str) -> float | None:
        import yfinance as yf

        for pair in (f"{currency}USD=X", f"USD{currency}=X"):
            try:
                raw = yf.download(pair, period="5d", interval="1d", progress=False, auto_adjust=False)
                if raw is None or raw.empty:
                    continue
                close = raw["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                series = pd.Series(close).dropna()
                if series.empty:
                    continue
                px = float(series.iloc[-1])
                if px <= 0:
                    continue
                return px if pair.startswith(currency) else 1.0 / px
            except Exception:  # noqa: BLE001
                continue
        return None

    def estimated_spread_bps(self, ticker: str, price: float) -> float:
        """Conservative spread estimate when no book is available."""
        if price <= 0:
            return 10.0
        tick_bps = (0.01 / price) * 10_000
        return max(2.0, min(50.0, tick_bps * 2))


def apply_spread_estimate(quotes: dict[str, Quote], svc: QuoteService) -> None:
    for q in quotes.values():
        if q.spread_bps is None and q.price:
            bps = svc.estimated_spread_bps(q.ticker, q.price)
            half = q.price * bps / 20_000
            q.bid = q.price - half
            q.ask = q.price + half
