"""Bar providers: IBKR historical (preferred) with yfinance fallback.

The active chain is decided by a capability probe at startup, because an
unentitled account answers historical requests with error 162/420.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

import pandas as pd

from tradebot.core.clock import now_utc, to_utc
from tradebot.core.logging import get_logger
from tradebot.ingest.market import store

log = get_logger("market.providers")


class BarProvider(Protocol):
    name: str

    async def available(self) -> bool: ...

    async def fetch(
        self, tickers: list[str], bar_minutes: int, days: int, kind: str = "intraday"
    ) -> dict[str, pd.DataFrame]: ...


# --------------------------------------------------------------------- IBKR
class IBPacer:
    """Respects IB's historical-data pacing limits."""

    def __init__(self, max_per_10min: int = 55, min_gap_same_contract: float = 1.0) -> None:
        self.max_per_10min = max_per_10min
        self.min_gap = min_gap_same_contract
        self._times: deque[float] = deque()
        self._last: dict[str, float] = {}

    async def wait(self, key: str) -> None:
        now = time.monotonic()
        while self._times and now - self._times[0] > 600:
            self._times.popleft()
        if len(self._times) >= self.max_per_10min:
            sleep_for = 600 - (now - self._times[0]) + 0.5
            log.warning("ib pacer sleeping", seconds=round(sleep_for, 1))
            await asyncio.sleep(max(1.0, sleep_for))
            now = time.monotonic()
        last = self._last.get(key)
        if last is not None and now - last < self.min_gap:
            await asyncio.sleep(self.min_gap - (now - last))
        self._times.append(time.monotonic())
        self._last[key] = time.monotonic()


class IBBarProvider:
    name = "ibkr"

    def __init__(self, client) -> None:  # client: tradebot.execution.ib_client.IBClient
        self.client = client
        self.pacer = IBPacer()
        self._failed = False
        self._fail_reason = ""

    async def available(self) -> bool:
        if self._failed or self.client is None:
            return False
        if not self.client.is_connected:
            return False
        # A competing session / missing entitlement means every request will hang
        # until timeout, so refuse the provider outright.
        return bool(self.client.market_data_ok)

    async def fetch(
        self, tickers: list[str], bar_minutes: int, days: int, kind: str = "intraday"
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        if not await self.available():
            return out
        bar_size = f"{bar_minutes} mins" if kind == "intraday" else "1 day"
        duration = f"{max(1, min(days, 365))} D"
        for ticker in tickers:
            contract = await self.client.qualify(ticker)
            if contract is None:
                continue
            await self.pacer.wait(ticker)
            try:
                bars = await asyncio.wait_for(
                    self.client.ib.reqHistoricalDataAsync(
                        contract,
                        endDateTime="",
                        durationStr=duration,
                        barSizeSetting=bar_size,
                        whatToShow="TRADES",
                        useRTH=True,
                        formatDate=2,
                    ),
                    timeout=45,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("ib bars failed", ticker=ticker, error=str(exc)[:160])
                continue
            if not bars:
                continue
            df = pd.DataFrame(
                [
                    {
                        "bar_ts": pd.Timestamp(b.date).tz_localize("UTC")
                        if pd.Timestamp(b.date).tzinfo is None
                        else pd.Timestamp(b.date).tz_convert("UTC"),
                        "open": b.open,
                        "high": b.high,
                        "low": b.low,
                        "close": b.close,
                        "volume": b.volume,
                    }
                    for b in bars
                ]
            ).set_index("bar_ts")
            out[ticker] = store.normalize(df)
        return out

    def mark_failed(self, reason: str) -> None:
        self._failed = True
        self._fail_reason = reason


# ----------------------------------------------------------------- yfinance
class YFinanceBarProvider:
    name = "yfinance"

    def __init__(self) -> None:
        self._down = False

    async def available(self) -> bool:
        return not self._down

    async def fetch(
        self, tickers: list[str], bar_minutes: int, days: int, kind: str = "intraday"
    ) -> dict[str, pd.DataFrame]:
        return await asyncio.to_thread(self._fetch_sync, tickers, bar_minutes, days, kind)

    def _fetch_sync(
        self, tickers: list[str], bar_minutes: int, days: int, kind: str
    ) -> dict[str, pd.DataFrame]:
        import yfinance as yf

        out: dict[str, pd.DataFrame] = {}
        interval = f"{bar_minutes}m" if kind == "intraday" else "1d"
        if kind == "intraday":
            # yfinance caps 15m history at ~60 calendar days
            period = f"{min(max(days, 5), 59)}d"
        else:
            period = f"{min(max(days, 30), 365 * 3)}d"

        for chunk in [tickers[i : i + 25] for i in range(0, len(tickers), 25)]:
            try:
                raw = yf.download(
                    chunk,
                    interval=interval,
                    period=period,
                    group_by="ticker",
                    auto_adjust=False,
                    threads=True,
                    progress=False,
                    prepost=False,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("yfinance chunk failed", error=str(exc)[:160], n=len(chunk))
                continue
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                for t in chunk:
                    if t not in raw.columns.get_level_values(0):
                        continue
                    sub = raw[t].dropna(how="all")
                    if sub.empty:
                        continue
                    norm = store.normalize(sub)
                    out[t] = store.to_session_close(norm) if kind == "daily" else norm
            else:
                t = chunk[0]
                sub = raw.dropna(how="all")
                if not sub.empty:
                    norm = store.normalize(sub)
                    out[t] = store.to_session_close(norm) if kind == "daily" else norm
        return out


# ------------------------------------------------------------------ service
@dataclass
class ProviderHealth:
    name: str
    ok: bool
    detail: str = ""
    checked_at: datetime = field(default_factory=now_utc)
    latency_ms: int = 0


class BarService:
    """Incremental bar cache in front of the provider chain."""

    def __init__(self, cfg, ib_client=None) -> None:  # cfg: AppConfig
        self.cfg = cfg
        self.providers: dict[str, BarProvider] = {"yfinance": YFinanceBarProvider()}
        if ib_client is not None:
            self.providers["ibkr"] = IBBarProvider(ib_client)
        self.health: dict[str, ProviderHealth] = {}
        self._cache: dict[tuple[str, int, str], datetime] = {}
        self._failures: dict[str, int] = {}

    def order(self) -> list[str]:
        wanted = self.cfg.data.bar_provider_order or ["yfinance"]
        return [p for p in wanted if p in self.providers]

    async def probe(self, tickers: list[str] | None = None) -> list[ProviderHealth]:
        tickers = tickers or ["AAPL"]
        results: list[ProviderHealth] = []
        for name in self.order():
            prov = self.providers[name]
            t0 = time.monotonic()
            try:
                ok = await prov.available()
                detail = ""
                if ok:
                    data = await prov.fetch(tickers[:1], self.cfg.data.bar_minutes, 3, "intraday")
                    ok = any(not df.empty for df in data.values())
                    detail = f"{sum(len(d) for d in data.values())} bars"
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, f"{type(exc).__name__}: {exc}"[:160]
            if not ok and name == "ibkr" and self.providers.get("ibkr") is not None:
                st = self.providers["ibkr"].client.status()  # type: ignore[attr-defined]
                detail = detail or ("no market data (competing session)" if st.get("competing_session")
                                    else (st.get("last_error") or "unavailable"))
            h = ProviderHealth(name=name, ok=ok, detail=detail,
                               latency_ms=int((time.monotonic() - t0) * 1000))
            self.health[name] = h
            results.append(h)
        return results

    def active_provider(self, kind: str = "intraday") -> str:
        for name in self.order():
            h = self.health.get(name)
            if h is not None and h.ok:
                return name
        return "yfinance"

    async def get_bars(
        self,
        tickers: list[str],
        bar_minutes: int | None = None,
        kind: str = "intraday",
        force: bool = False,
        days: int | None = None,
        budget_seconds: float | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch bars for `tickers`, filling only what is missing or stale.

        `budget_seconds` bounds the whole operation: providers are tried in order
        and the first one that returns usable data wins, so a slow provider can
        never consume the pipeline's stage budget. Partial results are kept.
        """
        started = time.monotonic()

        def _left() -> float:
            if budget_seconds is None:
                return float("inf")
            return budget_seconds - (time.monotonic() - started)

        bar_minutes = bar_minutes or self.cfg.data.bar_minutes
        days = days or (
            self.cfg.data.intraday_history_days if kind == "intraday" else self.cfg.data.daily_lookback_days
        )
        out: dict[str, pd.DataFrame] = {}
        missing: list[str] = []
        refresh_after = timedelta(minutes=self.cfg.data.bars_refresh_minutes)

        for t in tickers:
            key = (t, bar_minutes, kind)
            df = store.read(self.active_provider(kind), bar_minutes, t, kind)
            if df.empty:
                # try any provider that has cached data
                for name in self.order():
                    df = store.read(name, bar_minutes, t, kind)
                    if not df.empty:
                        break
            if df.empty:
                missing.append(t)
                out[t] = df
                continue
            age = now_utc() - to_utc(df.index[-1].to_pydatetime())
            stale = age > refresh_after
            last_fetch = self._cache.get(key)
            recently_fetched = last_fetch is not None and now_utc() - last_fetch < refresh_after
            if force or (stale and not recently_fetched):
                missing.append(t)
            out[t] = df

        if not missing:
            return out

        fetched_any = False
        for name in self.order():
            if _left() <= 5:
                log.warning("bar fetch budget exhausted", provider=name,
                            missing=len([t for t in missing if out.get(t) is None or out[t].empty]))
                break
            prov = self.providers[name]
            if not await prov.available():
                prev = self.health.get(name)
                detail = prev.detail if prev else "unavailable"
                self.health[name] = ProviderHealth(name=name, ok=False, detail=detail)
                continue
            try:
                data = await asyncio.wait_for(
                    prov.fetch(missing, bar_minutes, days, kind),
                    timeout=min(90.0, _left()) if budget_seconds is not None else 90.0,
                )
            except Exception as exc:  # noqa: BLE001
                self._failures[name] = self._failures.get(name, 0) + 1
                log.warning("provider fetch failed", provider=name, error=str(exc)[:200])
                self.health[name] = ProviderHealth(name=name, ok=False, detail=str(exc)[:160])
                if self._failures[name] >= 3 and hasattr(prov, "mark_failed"):
                    prov.mark_failed(str(exc)[:120])  # type: ignore[attr-defined]
                continue
            if not data:
                continue
            fetched_any = True
            for t, df in data.items():
                if df.empty:
                    continue
                full = store.append(name, bar_minutes, t, df, kind)
                out[t] = full
                self._cache[(t, bar_minutes, kind)] = now_utc()
            self.health[name] = ProviderHealth(name=name, ok=True, detail=f"{len(data)} tickers")
            break  # first working provider wins for this batch

        if not fetched_any:
            log.warning("no provider returned bars", n=len(missing))
        return out

    async def get_daily(self, tickers: list[str], days: int | None = None) -> dict[str, pd.DataFrame]:
        return await self.get_bars(tickers, bar_minutes=1440, kind="daily", days=days)

    def cached_bars(self, ticker: str, bar_minutes: int | None = None, kind: str = "intraday") -> pd.DataFrame:
        bar_minutes = bar_minutes or self.cfg.data.bar_minutes
        for name in self.order():
            df = store.read(name, bar_minutes, ticker, kind)
            if not df.empty:
                return df
        return pd.DataFrame(columns=store.COLUMNS)
