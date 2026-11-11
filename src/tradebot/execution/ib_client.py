"""IB Gateway client: connection lifecycle, contract cache, data health, errors."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from tradebot.core.clock import now_utc
from tradebot.core.logging import get_logger
from tradebot.core.symbols import ib_symbol
from tradebot.execution.ib_compat import IB, Stock

log = get_logger("ib")

# Error codes we care about (see PLAN §12.6)
INFO_CODES = {2104, 2106, 2107, 2108, 2158, 10167, 2119, 2109, 10197, 10168, 399, 2100}
CONNECTIVITY_CODES = {1100, 1101, 1102, 2110}
NO_DATA_CODES = {354, 10168, 10197, 10089, 10090}
PERMISSION_CODES = {162, 420, 200}


@dataclass
class IBNotice:
    code: int
    message: str
    ts: datetime = field(default_factory=now_utc)
    severity: str = "info"

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "ts": self.ts.isoformat(),
                "severity": self.severity}


class IBClient:
    """Owns the single IB connection used by the app."""

    def __init__(self, cfg, env) -> None:
        self.cfg = cfg
        self.env = env
        self.ib = IB()
        self.connected = False
        self.connecting = False
        self.market_data_ok = False
        self.competing_session = False
        self.account = env.IB_ACCOUNT or ""
        self.contracts: dict[str, Any] = {}
        self.min_ticks: dict[str, float] = {}
        self.notices: list[IBNotice] = []
        self.last_error = ""
        self.last_connect_at: datetime | None = None
        self.reconnect_attempts = 0
        self._reconnect_task: asyncio.Task | None = None
        self._order_handlers: list[Callable] = []
        self._fill_handlers: list[Callable] = []
        self._status_handlers: list[Callable] = []
        self._stopping = False
        self._lock = asyncio.Lock()
        self._install_handlers()

    @property
    def is_connected(self) -> bool:
        """True only when the socket is up (checked against ib_async as well)."""
        try:
            return bool(self.connected and self.ib.isConnected())
        except Exception:  # noqa: BLE001
            return bool(self.connected)

    # ------------------------------------------------------------- handlers
    def _install_handlers(self) -> None:
        self.ib.errorEvent += self._on_error
        self.ib.disconnectedEvent += self._on_disconnect
        self.ib.connectedEvent += self._on_connect
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_exec
        self.ib.commissionReportEvent += self._on_commission

    def on_order_status(self, fn: Callable) -> None:
        self._order_handlers.append(fn)

    def on_fill(self, fn: Callable) -> None:
        self._fill_handlers.append(fn)

    def on_connection(self, fn: Callable) -> None:
        self._status_handlers.append(fn)

    def _on_error(self, req_id, code, message, contract=None) -> None:  # noqa: ANN001
        code = int(code)
        text = str(message)[:300]
        severity = "info"
        if code in CONNECTIVITY_CODES:
            severity = "warn"
        elif code in NO_DATA_CODES:
            severity = "warn"
            self.market_data_ok = False
            if code == 10197:
                self.competing_session = True
        elif code in PERMISSION_CODES:
            severity = "error" if code in (200,) else "warn"
        elif code in (201, 202, 203, 399):
            severity = "warn"
        self.last_error = f"{code}: {text}"
        self.notices.append(IBNotice(code=code, message=text, severity=severity))
        self.notices = self.notices[-200:]
        if code not in INFO_CODES:
            log.info("ib notice", code=code, message=text[:160])

    def _on_connect(self) -> None:
        self.connected = True
        self.reconnect_attempts = 0
        log.info("ib connected", account=self.account)
        for fn in self._status_handlers:
            with contextlib.suppress(Exception):
                fn(True)

    def _on_disconnect(self) -> None:
        self.connected = False
        self.market_data_ok = False
        log.warning("ib disconnected")
        for fn in self._status_handlers:
            with contextlib.suppress(Exception):
                fn(False)
        if not self._stopping and self._reconnect_task is None:
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    def _on_order_status(self, trade) -> None:  # noqa: ANN001
        for fn in self._order_handlers:
            with contextlib.suppress(Exception):
                fn(trade)

    def _on_exec(self, trade, fill) -> None:  # noqa: ANN001
        for fn in self._fill_handlers:
            with contextlib.suppress(Exception):
                fn(trade, fill)

    def _on_commission(self, trade, fill, report) -> None:  # noqa: ANN001
        for fn in self._fill_handlers:
            with contextlib.suppress(Exception):
                fn(trade, fill, report)

    # ------------------------------------------------------------ lifecycle
    async def connect(self, retries: int = 3) -> bool:
        if self.connected:
            return True
        async with self._lock:
            if self.connected:
                return True
            self.connecting = True
            try:
                for attempt in range(retries):
                    try:
                        await self.ib.connectAsync(
                            self.env.IB_HOST, self.env.IB_PORT,
                            clientId=self.env.IB_CLIENT_ID, timeout=20,
                        )
                        self.connected = True
                        self.last_connect_at = now_utc()
                        accounts = self.ib.managedAccounts()
                        if accounts and not self.account:
                            self.account = accounts[0]
                        self.ib.reqMarketDataType(int(self.env.IB_MARKET_DATA_TYPE))
                        log.info("ib connect ok", host=self.env.IB_HOST, port=self.env.IB_PORT,
                                 account=self.account, accounts=accounts)
                        return True
                    except Exception as exc:  # noqa: BLE001
                        self.last_error = f"{type(exc).__name__}: {exc}"[:200]
                        log.warning("ib connect failed", attempt=attempt + 1, error=self.last_error)
                        await asyncio.sleep(min(2 ** attempt, 8))
                return False
            finally:
                self.connecting = False

    async def _reconnect_loop(self) -> None:
        delay = 2.0
        try:
            while not self._stopping and not self.connected:
                self.reconnect_attempts += 1
                await asyncio.sleep(delay)
                delay = min(delay * 1.8, 60.0)
                log.info("ib reconnect attempt", attempt=self.reconnect_attempts)
                if await self.connect(retries=1):
                    await self.after_reconnect()
                    return
        finally:
            self._reconnect_task = None

    async def after_reconnect(self) -> None:
        self.ib.reqMarketDataType(int(self.env.IB_MARKET_DATA_TYPE))
        self.contracts.clear()
        await self.qualify_many(["SPY"])
        for fn in self._status_handlers:
            with contextlib.suppress(Exception):
                fn(True)

    async def disconnect(self) -> None:
        self._stopping = True
        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            with contextlib.suppress(Exception):
                await self._reconnect_task
        if self.connected:
            self.ib.disconnect()
        self.connected = False

    # -------------------------------------------------------------- contracts
    async def qualify(self, ticker: str):
        if ticker in self.contracts:
            return self.contracts[ticker]
        if not self.connected:
            return None
        try:
            found = await asyncio.wait_for(
                self.ib.qualifyContractsAsync(Stock(ib_symbol(ticker), "SMART", "USD")), timeout=20
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("qualify failed", ticker=ticker, error=str(exc)[:120])
            return None
        if not found:
            return None
        contract = found[0]
        self.contracts[ticker] = contract
        try:
            details = await asyncio.wait_for(
                self.ib.reqContractDetailsAsync(contract), timeout=15
            )
            if details:
                self.min_ticks[ticker] = float(details[0].minTick or 0.01)
        except Exception:  # noqa: BLE001
            self.min_ticks.setdefault(ticker, 0.01)
        return contract

    async def qualify_many(self, tickers: list[str]) -> dict[str, Any]:
        out = {}
        for t in tickers:
            c = await self.qualify(t)
            if c is not None:
                out[t] = c
        return out

    def min_tick(self, ticker: str) -> float:
        return self.min_ticks.get(ticker, 0.01)

    def round_to_tick(self, ticker: str, price: float) -> float:
        """Snap a price to a valid increment for this contract.

        IB occasionally reports minTick=0.005 for a primary exchange while the
        SMART route still demands whole cents, which comes back as error 110
        ("price does not conform to the minimum price variation"). For stocks
        above $1 the tick is therefore never finer than one cent.
        """
        tick = self.min_tick(ticker)
        if price >= 1.0:
            tick = max(tick, 0.01)
        if tick <= 0:
            return round(price, 2)
        return round(round(price / tick) * tick, 6)

    # ------------------------------------------------------------- market data
    async def probe_market_data(self, ticker: str = "SPY") -> bool:
        """Verify that quotes actually arrive (delayed counts)."""
        contract = await self.qualify(ticker)
        if contract is None:
            self.market_data_ok = False
            return False
        self.ib.reqMarketDataType(int(self.env.IB_MARKET_DATA_TYPE))
        try:
            tickers = await asyncio.wait_for(self.ib.reqTickersAsync(contract), timeout=20)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            self.market_data_ok = False
            return False
        await asyncio.sleep(2.5)
        ok = False
        for t in tickers:
            vals = [getattr(t, a, None) for a in
                    ("bid", "ask", "last", "close", "delayedBid", "delayedAsk", "delayedLast", "delayedClose")]
            ok = any(isinstance(v, (int, float)) and v == v and v > 0 for v in vals)
        self.market_data_ok = ok
        if not ok:
            self.competing_session = any(n.code == 10197 for n in self.notices[-20:])
        return ok

    # ----------------------------------------------------------------- account
    async def account_summary(self) -> list:
        if not self.connected:
            return []
        try:
            return await asyncio.wait_for(self.ib.accountSummaryAsync(), timeout=20)
        except Exception as exc:  # noqa: BLE001
            log.warning("account summary failed", error=str(exc)[:120])
            return []

    def positions(self) -> list:
        return list(self.ib.positions()) if self.connected else []

    def portfolio(self) -> list:
        return list(self.ib.portfolio()) if self.connected else []

    async def open_orders(self) -> list:
        if not self.connected:
            return []
        try:
            return await asyncio.wait_for(self.ib.reqAllOpenOrdersAsync(), timeout=20)
        except Exception:  # noqa: BLE001
            return []

    async def server_time(self) -> datetime | None:
        if not self.connected:
            return None
        try:
            return await asyncio.wait_for(self.ib.reqCurrentTimeAsync(), timeout=10)
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        return {
            "connected": self.connected and self.ib.isConnected(),
            "connecting": self.connecting,
            "host": self.env.IB_HOST,
            "port": self.env.IB_PORT,
            "client_id": self.env.IB_CLIENT_ID,
            "account": self.account,
            "market_data_ok": self.market_data_ok,
            "competing_session": self.competing_session,
            "market_data_type": int(self.env.IB_MARKET_DATA_TYPE),
            "last_error": self.last_error,
            "last_connect_at": self.last_connect_at.isoformat() if self.last_connect_at else None,
            "reconnect_attempts": self.reconnect_attempts,
            "notices": [n.as_dict() for n in self.notices[-12:]],
            "library": "ib_async",
        }

    def uptime_s(self) -> float:
        if not self.last_connect_at:
            return 0.0
        return (now_utc() - self.last_connect_at).total_seconds()
