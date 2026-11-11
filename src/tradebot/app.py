"""Application state: owns services, scheduler, monitor loop and shared caches."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import func, select

from tradebot.core.clock import market_clock, now_utc, to_utc
from tradebot.core.events import bus
from tradebot.core.logging import get_logger, setup_logging
from tradebot.core.pipeline import Pipeline
from tradebot.core.types import AccountState, PositionState
from tradebot.db.models import AccountSnapshot, Alert, PositionRow, Trade
from tradebot.db.session import init_db, session_scope, state_all, state_get, state_set
from tradebot.execution.account import read_account
from tradebot.execution.ib_client import IBClient
from tradebot.execution.order_manager import OrderManager, position_state_from_row
from tradebot.execution.reconcile import reconcile
from tradebot.features.builder import select_feature_columns
from tradebot.ingest.market.providers import BarService
from tradebot.ingest.market.quotes import QuoteService
from tradebot.ingest.news.service import NewsService
from tradebot.ingest.universe.tradingview import UniverseService
from tradebot.models.predict import Predictor
from tradebot.risk.breakers import CooldownTracker
from tradebot.sentiment.finbert import FinBert
from tradebot.settings import AppConfig, ensure_dirs, get_env, load_config
from tradebot.strategy import reason_codes as rc

log = get_logger("app")


class App:
    def __init__(self, cfg: AppConfig | None = None) -> None:
        ensure_dirs()
        setup_logging()
        self.env = get_env()
        self.cfg = cfg or load_config()
        self.ib_client = IBClient(self.cfg, self.env)
        self.orders = OrderManager(self.cfg, self.env, self.ib_client)
        self.bars = BarService(self.cfg, self.ib_client)
        self.quotes = QuoteService(self.cfg, self.ib_client, self.bars)
        self.universe = UniverseService(self.cfg)
        self.news = NewsService(self.cfg, self.env)
        self.finbert = FinBert(self.cfg)
        self.predictor = Predictor(self.cfg)
        self.pipeline = Pipeline(self)
        self.cooldowns = CooldownTracker()
        self.decay_counts: dict[str, int] = {}
        self.started_at = now_utc()
        self._peak_equity = 0.0
        self._data_ok = True
        self._data_detail = ""
        self._bars_ok = 0
        self._bars_total = 0
        self._scheduler = None
        self._monitor_task: asyncio.Task | None = None
        self._account_cache: AccountState | None = None
        self._account_cache_at: datetime | None = None
        self._positions_cache: list[PositionState] = []
        self._positions_cache_at: datetime | None = None
        self._paused = False
        self._monitor_interval = self.cfg.pipeline.position_monitor_seconds

    # ------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        await init_db()
        await self.apply_db_overrides()
        await self.ib_client.connect()
        if self.ib_client.connected:
            ok = await self.ib_client.probe_market_data()
            log.info("market data probe", ok=ok, competing=self.ib_client.competing_session)
            await reconcile(self.ib_client)
            await self.refresh_quotes_cache()
        await self.bars.probe()
        loaded = await self.predictor.load()
        log.info("predictor loaded", ok=loaded, model=self.predictor.model_id)
        if self.cfg.sentiment.enabled:
            asyncio.create_task(self.finbert.ensure_loaded())
        self.start_scheduler()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        log.info("app started", mode=self.cfg.mode, ib=self.ib_client.connected)

    async def stop(self) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        await self.news.close()
        await self.ib_client.disconnect()

    # ---------------------------------------------------------------- config
    async def apply_db_overrides(self) -> None:
        from tradebot.settings import set_path

        overrides = await state_get("settings_overrides", {}) or {}
        if not overrides:
            return
        data = self.cfg.model_dump()
        for key, value in overrides.items():
            set_path(data, key, value)
        try:
            self.cfg = AppConfig(**data)
            self.ib_client.cfg = self.cfg
            self.orders.cfg = self.cfg
            self.bars.cfg = self.cfg
            self.quotes.cfg = self.cfg
            self.universe.cfg = self.cfg
            self.news.cfg = self.cfg
            self.finbert.cfg = self.cfg
            self.predictor.cfg = self.cfg
        except Exception as exc:  # noqa: BLE001
            log.error("invalid settings overrides ignored", error=str(exc)[:200])

    async def reload_config(self, overrides: dict) -> None:
        await state_set("settings_overrides", overrides)
        await self.apply_db_overrides()
        self.reschedule(self.cfg.pipeline.interval_minutes)

    # ------------------------------------------------------------- scheduler
    def start_scheduler(self) -> None:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger

        self._scheduler = AsyncIOScheduler(timezone="UTC")
        self._scheduler.add_job(
            self._scheduled_run,
            IntervalTrigger(minutes=max(3, self.cfg.pipeline.interval_minutes)),
            id="pipeline",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=300,
        )
        self._scheduler.start()
        log.info("scheduler started", interval_min=self.cfg.pipeline.interval_minutes)

    async def _scheduled_run(self) -> None:
        if self._paused:
            return
        try:
            await self.pipeline.run(trigger="scheduler")
        except Exception as exc:  # noqa: BLE001
            log.error("scheduled run failed", error=str(exc)[:300])

    def reschedule(self, minutes: int) -> None:
        if self._scheduler is None:
            return
        from apscheduler.triggers.interval import IntervalTrigger

        self._scheduler.reschedule_job(
            "pipeline", trigger=IntervalTrigger(minutes=max(3, minutes))
        )

    def pause(self) -> None:
        self._paused = True
        if self._scheduler is not None:
            self._scheduler.pause_job("pipeline")

    def resume(self) -> None:
        self._paused = False
        if self._scheduler is not None:
            self._scheduler.resume_job("pipeline")

    @property
    def paused(self) -> bool:
        return self._paused

    def next_run(self) -> datetime | None:
        if self._scheduler is None:
            return None
        job = self._scheduler.get_job("pipeline")
        if not job or not job.next_run_time:
            return None
        ts = job.next_run_time
        return ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts

    # ------------------------------------------------------------- positions
    async def get_positions(self, use_cache: bool = False) -> list[PositionState]:
        if (
            use_cache
            and self._positions_cache_at is not None
            and now_utc() - self._positions_cache_at < timedelta(seconds=5)
        ):
            return list(self._positions_cache)

        ib_positions = {}
        if self.ib_client.connected:
            for p in self.ib_client.positions():
                ticker = p.contract.symbol.replace(" ", ".")
                ib_positions[ticker] = (float(p.position), float(p.avgCost))

        out: list[PositionState] = []
        async with session_scope() as s:
            rows = (
                await s.execute(select(PositionRow).where(PositionRow.status == "open"))
            ).scalars().all()
            db_map = {r.ticker: r for r in rows}
            for ticker, (qty, avg) in ib_positions.items():
                row = db_map.get(ticker)
                if row is not None:
                    st = position_state_from_row(row)
                    st.qty = qty
                    st.avg_cost = avg
                else:
                    st = PositionState(ticker=ticker, qty=qty, avg_cost=avg, external=True)
                if st.entry_ts is not None:
                    st.bars_held = max(
                        0,
                        int((now_utc() - to_utc(st.entry_ts)).total_seconds()
                            // (self.cfg.data.bar_minutes * 60)),
                    )
                out.append(st)
            for ticker, row in db_map.items():
                if ticker in ib_positions:
                    continue
                st = position_state_from_row(row)
                out.append(st)

        for st in out:
            q = self.quotes._quote_cache.get(st.ticker)
            if q and q.price:
                st.last = q.price
        self._positions_cache = out
        self._positions_cache_at = now_utc()
        return out

    async def refresh_quotes_cache(self, tickers: list[str] | None = None) -> dict:
        if tickers is None:
            positions = await self.get_positions()
            tickers = [p.ticker for p in positions if abs(p.qty) > 0]
        if not tickers:
            return {}
        quotes = await self.quotes.get_quotes(tickers, allow_stale=True)
        return quotes

    # --------------------------------------------------------------- account
    async def get_account(self, use_cache: bool = True) -> AccountState:
        if (
            use_cache
            and self._account_cache is not None
            and self._account_cache_at is not None
            and now_utc() - self._account_cache_at < timedelta(seconds=20)
        ):
            return self._account_cache
        if self.ib_client.connected:
            state = await read_account(self.ib_client, self.quotes)
            if state.net_liq > 0:
                self._account_cache = state
                self._account_cache_at = now_utc()
                return state
        async with session_scope() as s:
            res = await s.execute(
                select(AccountSnapshot).order_by(AccountSnapshot.ts.desc()).limit(1)
            )
            row = res.scalars().first()
            if row is not None:
                state = AccountState(
                    currency=row.currency, net_liq=row.net_liq, cash=row.cash,
                    available_funds=row.available_funds, buying_power=row.buying_power,
                    excess_liquidity=row.excess_liquidity,
                    gross_position_value=row.gross_position_value,
                    realized_pnl=row.realized_pnl, unrealized_pnl=row.unrealized_pnl,
                    daily_pnl=row.daily_pnl, fx_to_usd=row.fx_to_usd,
                )
                return state
        return self._account_cache or AccountState()

    async def peak_equity(self) -> float:
        if self._peak_equity > 0:
            return self._peak_equity
        stored = await state_get("peak_equity", 0.0)
        self._peak_equity = float(stored or 0.0)
        return self._peak_equity

    async def update_peak_equity(self, account: AccountState | None) -> None:
        if account is None or account.net_liq <= 0:
            return
        if account.net_liq > self._peak_equity:
            self._peak_equity = account.net_liq
            await state_set("peak_equity", self._peak_equity)

    async def trades_today(self) -> int:
        from tradebot.core.clock import ET

        today = now_utc().astimezone(ET).date()
        async with session_scope() as s:
            res = await s.execute(select(Trade.created_at))
            rows = res.scalars().all()
        return sum(1 for ts in rows if ts and to_utc(ts).astimezone(ET).date() == today)

    async def consecutive_losses(self) -> int:
        async with session_scope() as s:
            res = await s.execute(select(Trade).order_by(Trade.created_at.desc()).limit(10))
            trades = res.scalars().all()
        count = 0
        for t in trades:
            if t.pnl < 0:
                count += 1
            else:
                break
        return count

    # ----------------------------------------------------------------- health
    def set_data_health(self, ok: int, total: int) -> None:
        self._bars_ok, self._bars_total = ok, total
        if total == 0:
            self._data_ok, self._data_detail = False, "no universe"
            return
        ratio = ok / total
        self._data_ok = ratio >= 0.7
        self._data_detail = f"{ok}/{total} tickers with fresh bars"

    def data_health(self) -> tuple[bool, str]:
        return self._data_ok, self._data_detail

    def model_health(self) -> tuple[bool, str]:
        if self.predictor.ready:
            return True, ""
        return False, "no trained model (bootstrap mode)"

    def feature_schema_hash(self) -> str:
        if self.predictor.bundle:
            return str(self.predictor.bundle.get("schema_hash", ""))
        return ""

    # ----------------------------------------------------------------- alerts
    async def alert(self, level: str, kind: str, message: str, payload: dict | None = None) -> None:
        async with session_scope() as s:
            # the same warning fires on every run; keep the feed readable
            recent = (await s.execute(
                select(Alert).where(Alert.message == message[:1000])
                .order_by(Alert.id.desc()).limit(1)
            )).scalars().first()
            if recent is not None and recent.ts is not None:
                age = (now_utc() - to_utc(recent.ts)).total_seconds()
                if age < 6 * 3600:
                    recent.ts = now_utc()
                    return
            s.add(Alert(level=level, kind=kind, message=message[:1000], payload=payload))
        await bus.publish("alert", level=level, kind=kind, message=message)

    async def persist_connector_health(self, batch) -> None:
        if batch is None or not getattr(batch, "health", None):
            return
        from tradebot.db.models import ConnectorHealth

        async with session_scope() as s:
            for h in batch.health:
                row = await s.get(ConnectorHealth, h["name"])
                if row is None:
                    row = ConnectorHealth(name=h["name"])
                    s.add(row)
                row.items_last_run = int(h.get("items_last_run", 0))
                row.latency_ms = int(h.get("latency_ms", 0))
                row.consecutive_failures = int(h.get("consecutive_failures", 0))
                row.enabled = bool(h.get("enabled", True))
                row.total_runs = int(h.get("total_runs", 0))
                row.total_items = int(h.get("total_items", 0))
                if h.get("last_error"):
                    row.last_error = h["last_error"][:400]
                    row.last_error_at = now_utc()
                if h.get("items_last_run", 0) > 0:
                    row.last_success = now_utc()

    # ---------------------------------------------------------------- monitor
    async def _monitor_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._monitor_interval)
                await self.monitor_tick()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                log.error("monitor tick failed", error=str(exc)[:200])

    async def monitor_tick(self) -> dict:
        out = {"quotes": 0, "trailing": [], "naked": [], "account": False}
        positions = await self.get_positions()
        longs = [p for p in positions if p.qty > 0]
        active = [p for p in positions if abs(p.qty) > 0]
        if active and self.ib_client.connected:
            quotes = await self.quotes.get_quotes([p.ticker for p in active], allow_stale=True)
            out["quotes"] = len(quotes)
            for p in longs:
                q = quotes.get(p.ticker)
                if q and q.price:
                    p.last = q.price
                    if p.highest_close is None or q.price > p.highest_close:
                        p.highest_close = q.price
                from tradebot.risk.sizing import new_trailing_stop

                atr = self._atr_for(p.ticker)
                new_sl = new_trailing_stop(p, atr, self.cfg)
                if new_sl is not None and self.cfg.mode in ("paper", "live"):
                    res = await self.orders.modify_stops(p.ticker, sl=new_sl)
                    if res.get("ok"):
                        out["trailing"].append({"ticker": p.ticker, "sl": new_sl})
                        await self.alert("info", "trailing", f"{p.ticker} stop trailed to {new_sl}")
            if self.cfg.mode in ("paper", "live"):
                out["naked"] = await self.orders.watchdog()
        acct = await self.get_account(use_cache=False)
        if acct.net_liq > 0:
            async with session_scope() as s:
                s.add(
                    AccountSnapshot(
                        ts=now_utc(), currency=acct.currency, net_liq=acct.net_liq, cash=acct.cash,
                        available_funds=acct.available_funds, buying_power=acct.buying_power,
                        excess_liquidity=acct.excess_liquidity,
                        gross_position_value=acct.gross_position_value,
                        realized_pnl=acct.realized_pnl, unrealized_pnl=acct.unrealized_pnl,
                        daily_pnl=acct.daily_pnl, fx_to_usd=acct.fx_to_usd,
                    )
                )
            out["account"] = True
            await self.update_peak_equity(acct)
        if longs:
            await bus.publish(
                "position.update",
                positions=[
                    {"ticker": p.ticker, "qty": p.qty, "last": p.last, "sl": p.sl, "tp": p.tp,
                     "unrealized": round(p.unrealized_pnl, 2)}
                    for p in longs
                ],
            )
        await bus.publish("account.update", net_liq=acct.net_liq, daily_pnl=acct.daily_pnl,
                          currency=acct.currency)
        return out

    def _atr_for(self, ticker: str) -> float:
        df = self.bars.cached_bars(ticker)
        if df.empty or len(df) < 20:
            return 0.0
        from tradebot.ingest.market.indicators import atr_series

        s = atr_series(df, 14)
        return float(s.iloc[-1]) if len(s) and np.isfinite(s.iloc[-1]) else 0.0

    def update_decay_counts(self, signals: dict, positions: list[PositionState]) -> None:
        for p in positions:
            sig = signals.get(p.ticker)
            if sig is not None and sig.p_win < self.cfg.strategy.exit_threshold_p_win:
                self.decay_counts[p.ticker] = self.decay_counts.get(p.ticker, 0) + 1
            else:
                self.decay_counts.pop(p.ticker, None)

    # ------------------------------------------------------------------ status
    async def status(self) -> dict:
        clock = market_clock()
        acct = await self.get_account()
        positions = await self.get_positions()
        kill = bool(await state_get("kill_switch", False))
        return {
            "mode": self.cfg.mode,
            "approval_mode": self.cfg.approval_mode,
            "paused": self._paused,
            "kill_switch": kill,
            "ib": self.ib_client.status(),
            "market": {
                "open": clock.is_open,
                "session": clock.session_label,
                "open_at": clock.session_open.isoformat() if clock.session_open else None,
                "close_at": clock.session_close.isoformat() if clock.session_close else None,
                "next_open": clock.next_open.isoformat(),
                "minutes_to_close": clock.minutes_to_close,
            },
            "next_run": self.next_run().isoformat() if self.next_run() else None,
            "interval_minutes": self.cfg.pipeline.interval_minutes,
            "account": {
                "currency": acct.currency, "net_liq": acct.net_liq, "cash": acct.cash,
                "available_funds": acct.available_funds, "buying_power": acct.buying_power,
                "excess_liquidity": acct.excess_liquidity,
                "gross_position_value": acct.gross_position_value,
                "realized_pnl": acct.realized_pnl, "unrealized_pnl": acct.unrealized_pnl,
                "daily_pnl": acct.daily_pnl, "daily_pnl_pct": acct.daily_pnl_pct,
                "fx_to_usd": acct.fx_to_usd, "net_liq_usd": acct.net_liq_usd,
                "excess_liq_pct": acct.excess_liq_pct,
            },
            "positions": len([p for p in positions if p.qty > 0]),
            "max_positions": self.cfg.strategy.max_positions,
            "model": self.predictor.status(),
            "finbert": self.finbert.status(),
            "bars": {"ok": self._bars_ok, "total": self._bars_total,
                     "health": [{"name": h.name, "ok": h.ok, "detail": h.detail}
                                for h in self.bars.health.values()]},
            "connectors": self.news.health_snapshot(),
            "pipeline": {
                "running": self.pipeline.running,
                "last_run_id": self.pipeline.last_run_id,
                "last_error": self.pipeline.last_error,
                "last_started": self.pipeline.last_started.isoformat() if self.pipeline.last_started else None,
                "last_finished": self.pipeline.last_finished.isoformat() if self.pipeline.last_finished else None,
                "last_duration_ms": self.pipeline.last_duration_ms,
            },
            "uptime_s": (now_utc() - self.started_at).total_seconds(),
        }

    # --------------------------------------------------------------- controls
    async def kill_switch(self, reason: str = "manual") -> dict:
        await state_set("kill_switch", True)
        await state_set("kill_reason", reason)
        results = []
        if self.cfg.mode in ("paper", "live"):
            results = await self.orders.close_all(rc.SELL_KILLSWITCH)
        await self.alert("critical", "killswitch", f"Kill switch engaged ({reason})")
        self.pause()
        return {"ok": True, "closed": [r.ticker for r in results]}

    async def rearm(self) -> dict:
        await state_set("kill_switch", False)
        await state_set("drawdown_halt", False)
        await state_set("peak_equity", (await self.get_account()).net_liq)
        self._peak_equity = (await self.get_account()).net_liq
        self.resume()
        await self.alert("info", "rearm", "Breakers re-armed manually")
        return {"ok": True}

    async def set_mode(self, mode: str) -> dict:
        if mode not in ("off", "dry_run", "paper", "live"):
            raise ValueError("invalid mode")
        if mode == "live":
            if not self.env.ALLOW_LIVE or self.env.IB_PORT != 4001:
                raise ValueError("LIVE requires ALLOW_LIVE=1 and IB_PORT=4001 in .env")
        overrides = await state_get("settings_overrides", {}) or {}
        overrides["mode"] = mode
        await state_set("settings_overrides", overrides)
        self.cfg.mode = mode  # type: ignore[assignment]
        await self.alert("warning", "mode", f"Trading mode changed to {mode.upper()}")
        return {"ok": True, "mode": mode}

    async def state_get(self, key: str, default=None):
        return await state_get(key, default)
