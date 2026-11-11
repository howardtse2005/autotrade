"""Pipeline orchestrator: S0 pre-flight -> S7 persist, with per-stage telemetry."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import func, select

from tradebot.core.clock import MarketClock, market_clock, now_utc
from tradebot.core.events import bus
from tradebot.core.logging import get_logger
from tradebot.core.types import (
    AccountState,
    BreakerState,
    Quote,
    StageResult,
    TickerSignal,
    UniverseRow,
)
from tradebot.db.models import (
    AccountSnapshot,
    Prediction,
    PositionRow,
    QuoteRow,
    Run,
    RunEvent,
    Signal,
    Trade,
    UniverseSnapshot,
)
from tradebot.db.session import session_scope
from tradebot.execution.order_manager import position_state_from_row
from tradebot.features.builder import build_ticker_frame
from tradebot.risk.breakers import BreakerInputs, CooldownTracker, evaluate
from tradebot.sentiment.aggregate import latest_features
from tradebot.strategy import reason_codes as rc
from tradebot.strategy.decide import PortfolioState, decide

log = get_logger("pipeline")


@dataclass
class RunContext:
    run_id: int
    trigger: str
    started: datetime = field(default_factory=now_utc)
    stages: list[StageResult] = field(default_factory=list)
    clock: MarketClock | None = None
    universe: list[UniverseRow] = field(default_factory=list)
    # every screened name available to this run, and how many we have evaluated
    pool: list[UniverseRow] = field(default_factory=list)
    cursor: int = 0
    bars: dict = field(default_factory=dict)
    daily: dict = field(default_factory=dict)
    quotes: dict[str, Quote] = field(default_factory=dict)
    news_batch: object | None = None
    signals: dict[str, TickerSignal] = field(default_factory=dict)
    bundle: object | None = None
    account: AccountState | None = None
    breaker: object | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    funnel: dict = field(default_factory=dict)
    model_id: str = ""
    bootstrap: bool = True


class Pipeline:
    def __init__(self, app) -> None:
        self.app = app  # tradebot.web.app.AppState holder
        self._lock = asyncio.Lock()
        self.running = False
        self.last_run_id: int | None = None
        self.last_error = ""
        self.last_started: datetime | None = None
        self.last_finished: datetime | None = None
        self.last_duration_ms = 0

    # ------------------------------------------------------------------ utils
    def _cfg(self):
        return self.app.cfg

    async def _event(self, run_id: int, stage: str, message: str, level: str = "info",
                     payload: dict | None = None) -> None:
        async with session_scope() as s:
            s.add(RunEvent(run_id=run_id, stage=stage, level=level, message=message[:2000],
                           payload=payload))
        await bus.publish("run.stage", run_id=run_id, stage=stage, level=level, message=message)

    async def _stage(self, ctx: RunContext, name: str, fn, timeout: float) -> StageResult:
        res = StageResult(name=name, started_at=now_utc())
        t0 = time.time()
        try:
            detail = await asyncio.wait_for(fn(ctx), timeout=timeout)
            res.status = "ok"
            res.detail = detail or ""
        except asyncio.TimeoutError:
            res.status = "timeout"
            res.detail = f"timeout after {timeout}s"
            ctx.errors.append(f"{name}: {res.detail}")
            await self._event(ctx.run_id, name, res.detail, "error")
        except Exception as exc:  # noqa: BLE001
            res.status = "error"
            res.detail = f"{type(exc).__name__}: {exc}"[:300]
            ctx.errors.append(f"{name}: {res.detail}")
            log.error("stage failed", stage=name, error=res.detail)
            await self._event(ctx.run_id, name, res.detail, "error")
        res.duration_ms = int((time.time() - t0) * 1000)
        ctx.stages.append(res)
        await bus.publish("run.stage", run_id=ctx.run_id, stage=name, status=res.status,
                          duration_ms=res.duration_ms, detail=res.detail)
        return res

    # ------------------------------------------------------------------- main
    async def run(self, trigger: str = "scheduler", force: bool = False) -> dict:
        if self._lock.locked() and not force:
            return {"status": "busy"}
        async with self._lock:
            self.running = True
            cfg = self._cfg()
            run_id = await self._create_run(trigger)
            ctx = RunContext(run_id=run_id, trigger=trigger)
            self.last_run_id = run_id
            self.last_started = ctx.started
            await bus.publish("run.start", run_id=run_id, trigger=trigger, mode=cfg.mode)
            try:
                await self._stage(ctx, "S0_preflight", self._s0, 20)
                if ctx.stages and ctx.stages[-1].status != "ok" and ctx.stages[-1].status != "warn":
                    return await self._finish(ctx, "skipped", ctx.stages[-1].detail)

                await self._stage(ctx, "S1_universe", self._s1, 30)
                await self._stage(ctx, "S2_ingest", self._s2, 240)
                await self._stage(ctx, "S3_sentiment", self._s3, cfg.sentiment.score_budget_seconds + 60)
                await self._stage(ctx, "S4_predict", self._s4, 90)
                await self._stage(ctx, "S5_strategy", self._s5, 20)
                # keep searching until the book is full (or the pool runs out)
                rounds = 0
                blocked = bool(
                    ctx.breaker is not None and ctx.breaker.entries_blocked
                )
                while (
                    not blocked
                    and rounds < int(cfg.universe.max_expand_rounds)
                    and ctx.cursor < len(ctx.pool)
                    and ctx.funnel.get("selected", 0) < ctx.funnel.get("free_slots", 0)
                ):
                    rounds += 1
                    res = await self._stage(
                        ctx, f"S5b_expand_{rounds}", self._expand, 180
                    )
                    if res.status not in ("ok", "warn"):
                        break
                positions = await self.app.get_positions()
                self.app.update_decay_counts(ctx.signals, positions)
                await self._stage(ctx, "S6_execute", self._s6, 120)
                await self._stage(ctx, "S7_persist", self._s7, 30)
                status = "ok" if not ctx.errors else "partial"
                return await self._finish(ctx, status, "")
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"[:400]
                log.exception("pipeline crashed")
                return await self._finish(ctx, "error", self.last_error)
            finally:
                self.running = False

    async def _create_run(self, trigger: str) -> int:
        cfg = self._cfg()
        async with session_scope() as s:
            row = Run(
                started_at=now_utc(), status="running", trigger=trigger, mode=cfg.mode,
                interval_min=cfg.pipeline.interval_minutes, config_hash=cfg.hash(),
                model_id=self.app.predictor.model_id or "",
            )
            s.add(row)
            await s.flush()
            return row.id

    async def _finish(self, ctx: RunContext, status: str, error: str) -> dict:
        cfg = self._cfg()
        finished = now_utc()
        duration = int((finished - ctx.started).total_seconds() * 1000)
        summary = {
            "funnel": ctx.funnel,
            "signals": len(ctx.signals),
            "orders": len(getattr(ctx.bundle, "intents", []) or []),
            "model_id": ctx.model_id,
            "bootstrap": ctx.bootstrap,
            "warnings": ctx.warnings[:8],
        }
        async with session_scope() as s:
            row = await s.get(Run, ctx.run_id)
            if row is not None:
                row.finished_at = finished
                row.status = status
                row.duration_ms = duration
                row.stage_timings = [
                    {"name": st.name, "status": st.status, "ms": st.duration_ms,
                     "detail": st.detail} for st in ctx.stages
                ]
                row.funnel = ctx.funnel
                row.summary = summary
                row.error = error or None
                row.model_id = ctx.model_id or row.model_id
        self.last_finished = finished
        self.last_duration_ms = duration
        await bus.publish("run.done", run_id=ctx.run_id, status=status, duration_ms=duration,
                          summary=summary)
        log.info("run finished", run_id=ctx.run_id, status=status, ms=duration)
        del cfg
        return {"status": status, "run_id": ctx.run_id, "duration_ms": duration,
                "funnel": ctx.funnel, "error": error}

    # -------------------------------------------------------------------- S0
    async def _s0(self, ctx: RunContext) -> str:
        cfg = self._cfg()
        app = self.app
        if cfg.mode == "off":
            raise RuntimeError("mode is OFF")
        ctx.clock = market_clock()
        await self._event(ctx.run_id, "S0", f"market {'open' if ctx.clock.is_open else 'closed'} "
                                            f"({ctx.clock.session_label})")
        if cfg.pipeline.market_hours_only and not ctx.clock.is_open and not cfg.pipeline.research_when_closed:
            raise RuntimeError("market closed and research mode disabled")

        # intents that claim to be working must still exist at IB
        try:
            stale = await app.orders.reconcile_intents()
            if stale:
                ctx.funnel["stale_intents"] = stale
        except Exception as exc:  # noqa: BLE001
            log.warning("reconcile_intents failed", error=str(exc)[:160])

        # a new run supersedes every earlier pending approval
        try:
            expired = await app.orders.expire_stale(ctx.run_id)
            if expired:
                ctx.funnel["expired_approvals"] = expired
                await self._event(ctx.run_id, "S0", f"{expired} stale approvals expired", "warn")
        except Exception as exc:  # noqa: BLE001
            log.warning("expire_stale failed", error=str(exc)[:160])

        ctx.account = await app.get_account()
        positions = await app.get_positions()
        day_pnl_pct = ctx.account.daily_pnl_pct
        peak = await app.peak_equity()
        trades_today = await app.trades_today()
        consecutive = await app.consecutive_losses()
        data_ok, data_detail = app.data_health()
        model_ok, model_detail = app.model_health()
        manual_halt = bool(await app.state_get("kill_switch", False))
        dd_halt = bool(await app.state_get("drawdown_halt", False))

        br = evaluate(
            BreakerInputs(
                account=ctx.account, day_pnl_pct=day_pnl_pct, peak_equity=peak,
                trades_today=trades_today, consecutive_losses=consecutive,
                data_ok=data_ok, data_detail=data_detail, model_ok=model_ok,
                model_detail=model_detail, manual_halt=manual_halt, drawdown_halt=dd_halt,
            ),
            cfg,
        )
        ctx.breaker = br
        if br.reasons:
            await self._event(ctx.run_id, "S0", "; ".join(br.reasons), "warn")
            ctx.warnings.extend(br.reasons)
        if cfg.mode in ("paper", "live") and not app.ib_client.connected:
            raise RuntimeError("IB Gateway not connected")
        ctx.funnel["positions"] = len(positions)
        return f"{len(positions)} positions, breakers {'BLOCKED' if br.entries_blocked else 'ok'}"

    # -------------------------------------------------------------------- S1
    async def _s1(self, ctx: RunContext) -> str:
        cfg = self._cfg()
        held = [p.ticker for p in await self.app.get_positions() if p.qty > 0]
        # Size the first pass by the number of open slots: evaluating a fixed 50
        # names and then filtering down is what left the book half empty.
        free = max(1, int(cfg.strategy.max_positions) - len(held))
        size = min(
            int(cfg.universe.max_candidates),
            max(int(cfg.universe.size), free * int(cfg.universe.names_per_slot)),
        )
        rows = await self.app.universe.get(held=held, limit=int(cfg.universe.max_candidates))
        # keep the whole screened pool so later rounds can pull further tranches
        ctx.pool = list(rows)
        ctx.universe = list(rows[:size])
        ctx.cursor = min(size, len(rows))
        rows = ctx.universe
        if self.app.universe.stale:
            ctx.warnings.append(f"universe stale ({self.app.universe.last_source})")
        async with session_scope() as s:
            for r in rows:
                s.add(
                    UniverseSnapshot(
                        run_id=ctx.run_id, ticker=r.ticker, exchange=r.exchange, price=r.price,
                        volume=r.volume, change_pct=r.change_pct, rel_volume=r.rel_volume,
                        market_cap=r.market_cap, dollar_volume=r.dollar_volume, atr_pct=r.atr_pct,
                        sector=r.sector, industry=r.industry, screen_score=r.screen_score,
                        rank=r.rank, is_held=r.is_held, stale=self.app.universe.stale,
                        source=self.app.universe.last_source,
                    )
                )
        return (f"{len(rows)}/{len(ctx.pool)} tickers from "
                f"{self.app.universe.last_source}")

    # -------------------------------------------------------------------- S2
    async def _s2(self, ctx: RunContext) -> str:
        cfg = self._cfg()
        app = self.app
        tickers = [r.ticker for r in ctx.universe]
        benchmarks = cfg.universe.benchmarks or ["SPY"]

        async def news_task():
            return await app.news.collect(ctx.universe, ctx.run_id)

        async def market_task():
            # S2 has a 150 s budget; leave room for quotes and the daily context.
            bars = await app.bars.get_bars(tickers, budget_seconds=90)
            daily = await app.bars.get_daily(tickers + benchmarks + ["^VIX"])
            quotes = await app.quotes.get_quotes(tickers)
            from tradebot.ingest.market.quotes import apply_spread_estimate

            apply_spread_estimate(quotes, app.quotes)
            return bars, daily, quotes

        news_res, market_res = await asyncio.gather(
            news_task(), market_task(), return_exceptions=True
        )
        if isinstance(news_res, Exception):
            ctx.errors.append(f"news: {news_res}")
            await self._event(ctx.run_id, "S2a", str(news_res)[:200], "error")
        else:
            ctx.news_batch = news_res
            ctx.funnel["news_items"] = len(news_res.items)
            ctx.funnel["coverage_pct"] = round(news_res.coverage_pct, 1)
        if isinstance(market_res, Exception):
            raise RuntimeError(f"market data: {market_res}")
        ctx.bars, ctx.daily, ctx.quotes = market_res
        ctx.funnel["bars"] = len(ctx.bars)
        ctx.funnel["quotes"] = len(ctx.quotes)

        missing = [t for t, df in ctx.bars.items() if df.empty]
        if missing:
            ctx.warnings.append(f"{len(missing)} tickers without bars")
        from sqlalchemy import delete

        async with session_scope() as s:
            await s.execute(delete(QuoteRow).where(QuoteRow.run_id == ctx.run_id))
            for t, q in ctx.quotes.items():
                s.add(
                    QuoteRow(
                        run_id=ctx.run_id, ticker=t, ts=q.ts, bid=q.bid, ask=q.ask, last=q.last,
                        bid_size=q.bid_size, ask_size=q.ask_size, spread_bps=q.spread_bps,
                        source=q.source, market_data_type=q.market_data_type,
                    )
                )
        app.set_data_health(len(ctx.bars) - len(missing), len(tickers))
        return (f"{ctx.funnel.get('news_items', 0)} news, {len(ctx.quotes)} quotes, "
                f"{len(ctx.bars) - len(missing)}/{len(tickers)} bar series")

    # -------------------------------------------------------------------- S3
    async def _s3(self, ctx: RunContext) -> str:
        cfg = self._cfg()
        batch = ctx.news_batch
        if batch is None or not getattr(batch, "items", None):
            ctx.scores = {}
            return "no news to score"
        order = self._priority_order(ctx)
        pairs = []
        for t in order:
            for item in batch.by_ticker.get(t, []):
                if item.item_id:
                    pairs.append((item.item_id, item.text))
        scores = await self.app.finbert.score_texts(pairs, cfg.sentiment.score_budget_seconds)
        ctx.scores = scores
        scored = sum(1 for p in pairs if p[0] in scores)
        ctx.funnel["news_scored"] = scored
        ctx.funnel["news_pending"] = max(0, len(pairs) - scored)
        await self._persist_sentiment(ctx, batch, scores)
        return f"{scored}/{len(pairs)} texts scored"

    def _priority_order(self, ctx: RunContext) -> list[str]:
        held = [p.ticker for p in (ctx.universe or []) if p.is_held]
        ranked = [r.ticker for r in sorted(ctx.universe, key=lambda r: r.rank)]
        rest = [r.ticker for r in ctx.universe if r.ticker not in held and r.ticker not in ranked[:15]]
        return held + ranked[:15] + rest

    async def _persist_sentiment(self, ctx: RunContext, batch, scores) -> None:
        from tradebot.db.models import NewsItem

        async with session_scope() as s:
            for item in batch.items:
                sc = scores.get(item.item_id or "")
                if sc is None:
                    continue
                res = await s.execute(
                    select(NewsItem).where(NewsItem.item_hash == item.item_id).limit(1)
                )
                row = res.scalars().first()
                if row is not None and row.p_pos is None:
                    row.p_pos = sc.p_pos
                    row.p_neg = sc.p_neg
                    row.p_neu = sc.p_neu
                    row.label = sc.label

    # -------------------------------------------------------------------- S4
    async def _s4(self, ctx: RunContext) -> str:
        cfg = self._cfg()
        app = self.app
        batch = ctx.news_batch
        scores = getattr(ctx, "scores", {})
        benchmarks = cfg.universe.benchmarks or ["SPY"]
        spy = ctx.daily.get(benchmarks[0]) if benchmarks else None
        vix = ctx.daily.get("^VIX")

        rows: list[pd.DataFrame] = []
        sector_map = {r.ticker: r.sector for r in ctx.universe}
        held_map = {r.ticker: r.is_held for r in ctx.universe}
        pending_frac = 0.0
        if batch is not None and getattr(batch, "items", None):
            total = len(batch.items)
            scored = sum(1 for i in batch.items if i.item_id in scores)
            pending_frac = 1 - (scored / total) if total else 0.0

        for row in ctx.universe:
            t = row.ticker
            bars = ctx.bars.get(t)
            if bars is None or bars.empty:
                continue
            try:
                f = build_ticker_frame(
                    t, bars, cfg, daily=ctx.daily.get(t), spy=spy, vix=vix,
                    sentiment=None, with_labels=False,
                )
            except Exception as exc:  # noqa: BLE001
                ctx.warnings.append(f"{t}: features failed ({str(exc)[:60]})")
                continue
            if f.empty:
                continue
            last = f.iloc[[-1]].copy()
            if batch is not None:
                items = batch.by_ticker.get(t, [])
                if items:
                    # sentiment is evaluated as of *now* (news arrives after the bar close)
                    sent = latest_features(items, scores, now_utc(), cfg, pending_frac=pending_frac)
                    for k, v in sent.items():
                        last[k] = v
            rows.append(last)

        if not rows:
            raise RuntimeError("no feature rows built")

        frame = pd.concat(rows)
        frame["ticker"] = frame["ticker"].astype(str)
        from tradebot.features.builder import add_cross_section

        frame = add_cross_section(frame, cfg)
        ctx.feature_frame = frame
        schema_hash = app.feature_schema_hash()
        preds = app.predictor.predict(frame, schema_hash=schema_hash)
        ctx.model_id = preds.model_id
        ctx.bootstrap = preds.bootstrap
        if preds.error:
            ctx.warnings.append(f"predictor: {preds.error}")
        pdf = preds.frame

        # Percentile of p_win inside *this* cross-section: the entry gate is
        # relative, because the absolute level of p_win moves with the regime.
        min_names = int(getattr(cfg.strategy, "min_cross_section_names", 10))
        if "p_win" in pdf.columns and len(pdf) >= min_names:
            pdf = pdf.copy()
            pdf["p_win_pct"] = pdf["p_win"].rank(pct=True, method="average")
            pdf["p_entry_pct"] = pdf["p_entry"].rank(pct=True, method="average")
        else:
            pdf = pdf.copy()
            pdf["p_win_pct"] = np.nan
            pdf["p_entry_pct"] = np.nan
        ctx.funnel["cross_section_names"] = int(len(pdf))
        ctx.funnel["candidates_evaluated"] = int(len(ctx.universe))
        ctx.funnel["candidate_pool"] = int(len(ctx.pool))
        if len(pdf) < min_names:
            ctx.warnings.append(
                f"cross-section has only {len(pdf)} names — falling back to absolute thresholds"
            )

        # structural liquidity from the universe screen (price x 10-day avg volume)
        liq = {u.ticker: float(u.dollar_volume or 0) for u in (ctx.universe or [])}

        signals: dict[str, TickerSignal] = {}
        for _, r in pdf.iterrows():
            t = str(r["ticker"])
            q = ctx.quotes.get(t)
            price = float(q.price) if q and q.price else float(r.get("close") or 0)
            if price <= 0:
                continue
            atr = float(r.get("atr14") or 0)
            spread = q.spread_bps if q else None
            age = q.age_seconds(now_utc()) if q else None
            sig = TickerSignal(
                ticker=t,
                bar_ts=r.name.to_pydatetime() if hasattr(r.name, "to_pydatetime") else now_utc(),
                price=price,
                atr=atr,
                atr_pct=float(r.get("atr_pct") or 0),
                p_win=float(r.get("p_win") or 0.5),
                p_win_pct=(
                    float(r["p_win_pct"]) if r.get("p_win_pct") is not None
                    and not pd.isna(r.get("p_win_pct")) else None
                ),
                expected_r=float(r.get("expected_r") if r.get("expected_r") is not None
                                 else _expected_r(r, cfg)),
                vol_hat=float(r.get("vol_hat") or 0.01),
                p_entry=float(r.get("p_entry") or 0.5),
                p_entry_pct=(
                    float(r["p_entry_pct"]) if r.get("p_entry_pct") is not None
                    and not pd.isna(r.get("p_entry_pct")) else None
                ),
                disagreement=float(r.get("disagreement") or 0),
                ml_score=float(r.get("ml_score") or 50),
                p_up_trees=float(r.get("p_up_trees") or 0.5),
                p_down_trees=float(r.get("p_down_trees") or 0.5),
                p_up_lin=float(r.get("p_up_lin") or 0.5),
                exp_ret=float(r.get("exp_ret") or 0),
                sent_ewm_240=_nan0(r.get("sent_ewm_240m")),
                sent_mean_60=_nan0(r.get("sent_mean_60m")),
                sent_max_neg_240=_nan0(r.get("sent_max_neg_240m")),
                news_count_240=int(_nan0(r.get("news_count_240m"))),
                news_neg_240=int(_nan0(r.get("news_neg_240m"))),
                news_pending_frac=float(_nan0(r.get("news_pending_frac"))),
                rel_volume=float(_nan0(r.get("rel_volume"), 1.0)),
                dollar_volume=float(_nan0(r.get("dollar_volume"))),
                avg_dollar_volume=liq.get(t, 0.0),
                spread_bps=spread,
                quote_age_s=age,
                sector=sector_map.get(t, "Unknown"),
                bootstrap=preds.bootstrap,
                shap_top3=r.get("shap_top3") or [],
            )
            signals[t] = sig
        ctx.signals = signals
        ctx.funnel["signals"] = len(signals)
        await self._persist_predictions(ctx, pdf)
        return f"{len(signals)} signals, model={preds.model_id or 'BOOTSTRAP'}"

    async def _persist_predictions(self, ctx: RunContext, pdf: pd.DataFrame) -> None:
        from sqlalchemy import delete

        async with session_scope() as s:
            await s.execute(delete(Prediction).where(Prediction.run_id == ctx.run_id))
            for _, r in pdf.iterrows():
                bar_ts = r.name.to_pydatetime() if hasattr(r.name, "to_pydatetime") else now_utc()
                s.add(
                    Prediction(
                        run_id=ctx.run_id, ticker=str(r["ticker"]), bar_ts=bar_ts,
                        model_id=ctx.model_id or "bootstrap",
                        p_win=float(r.get("p_win") or 0),
                        expected_r=float(r.get("expected_r") if r.get("expected_r") is not None
                                         else (r.get("exp_ret") or 0)),
                        vol_hat=float(r.get("vol_hat") or 0), p_entry=float(r.get("p_entry") or 0),
                        disagreement=float(r.get("disagreement") or 0),
                        ml_score=float(r.get("ml_score") or 50),
                        p_up_trees=float(r.get("p_up_trees") or 0),
                        p_down_trees=float(r.get("p_down_trees") or 0),
                        p_up_lin=float(r.get("p_up_lin") or 0),
                        exp_ret=float(r.get("exp_ret") or 0),
                        shap_top3=r.get("shap_top3") or [],
                    )
                )

    # -------------------------------------------------------------------- S5
    async def _s5(self, ctx: RunContext) -> str:
        cfg = self._cfg()
        app = self.app
        positions = await app.get_positions()
        for p in positions:
            q = ctx.quotes.get(p.ticker)
            if q and q.price:
                p.last = q.price
            if p.highest_close is None or (p.last and p.last > p.highest_close):
                p.highest_close = p.last

        state = PortfolioState(
            positions=positions,
            account=ctx.account or AccountState(),
            clock=ctx.clock,
            trades_today=await app.trades_today(),
            day_pnl_pct=ctx.account.daily_pnl_pct if ctx.account else 0.0,
            peak_equity=await app.peak_equity(),
            breakers=ctx.breaker,
            cooldowns=app.cooldowns,
            consecutive_losses=await app.consecutive_losses(),
            decay_counts=app.decay_counts,
            mode=cfg.mode,
            bootstrap=ctx.bootstrap,
            run_id=ctx.run_id,
            allow_queued_entries=cfg.pipeline.queue_entries_outside_rth,
            reserved_usd=await app.orders.queued_buy_notional(),
            queued_tickers=await app.orders.queued_buy_tickers(),
        )
        thresholds = app.predictor.effective_thresholds()
        ctx.thresholds = thresholds
        bundle = decide(state, ctx.signals, cfg, universe_size=len(ctx.universe),
                        thresholds=thresholds)
        ctx.bundle = bundle
        ctx.funnel.update(bundle.funnel)
        ctx.funnel["buy_threshold"] = round(float(thresholds.get("buy", 0)), 4)
        ctx.funnel["threshold_source"] = thresholds.get("source", "config")
        ctx.funnel["reserved_usd"] = round(float(state.reserved_usd), 2)
        ctx.funnel["queued_slots"] = len(state.queued_tickers)
        ctx.funnel["threshold_mode"] = thresholds.get("mode", "absolute")
        ctx.funnel["min_bar_pct"] = round(float(thresholds.get("min_bar_pct") or 0), 4)
        edge = app.predictor.thresholds.get("stats", {})
        if edge and not app.predictor.thresholds.get("edge_verified", False):
            msg = (f"Model edge unverified: top-decile OOF expectancy "
                   f"{edge.get('top_decile_mean_r', 0):+.2f}R over {edge.get('top_decile_trades', 0)} trades")
            if msg not in ctx.warnings:
                ctx.warnings.append(msg)
            ctx.funnel["edge_verified"] = False
        else:
            ctx.funnel["edge_verified"] = bool(edge)
        await self._persist_signals(ctx, bundle)
        return (f"{bundle.funnel.get('selected', 0)} selected, "
                f"{len(bundle.intents)} intents, {bundle.funnel.get('exits', 0)} exits")

    async def _expand(self, ctx: RunContext) -> str:
        """Evaluate the next tranche of the screened pool.

        The run keeps pulling names while the book still has empty slots, so a
        quiet day cannot starve the selection down to one candidate.
        """
        cfg = self._cfg()
        step = int(cfg.universe.expand_step)
        start = ctx.cursor
        if start >= len(ctx.pool):
            return "candidate pool exhausted"
        add = ctx.pool[start : start + step]
        if not add:
            return "candidate pool exhausted"
        ctx.cursor = start + len(add)
        ctx.universe = list(ctx.universe) + list(add)
        ctx.funnel["expansion_rounds"] = int(ctx.funnel.get("expansion_rounds", 0)) + 1
        ctx.funnel["candidates_evaluated"] = len(ctx.universe)

        await self._s2(ctx)
        await self._s3(ctx)
        await self._s4(ctx)
        await self._s5(ctx)
        return (f"+{len(add)} names -> {ctx.funnel.get('selected', 0)}/"
                f"{ctx.funnel.get('free_slots', 0)} slots filled")

    async def _persist_signals(self, ctx: RunContext, bundle) -> None:
        from sqlalchemy import delete

        async with session_scope() as s:
            await s.execute(delete(Signal).where(Signal.run_id == ctx.run_id))
            for d in bundle.decisions:
                s.add(
                    Signal(
                        run_id=ctx.run_id, ticker=d.ticker, action=d.action,
                        reason_codes=d.reason_codes, rank=d.rank, rank_score=d.rank_score,
                        planned_qty=d.planned_qty, planned_sl=d.planned_sl, planned_tp=d.planned_tp,
                        detail=d.detail or None, ts=now_utc(),
                    )
                )

    # -------------------------------------------------------------------- S6
    async def _s6(self, ctx: RunContext) -> str:
        cfg = self._cfg()
        app = self.app
        if ctx.bundle is None:
            return "no bundle"
        intents = ctx.bundle.intents or []

        # 1) price the marketable limits from the live quote
        for intent in intents:
            q = ctx.quotes.get(intent.ticker)
            queued = bool((intent.meta or {}).get("queued"))
            if intent.side == "BUY" and (intent.order_type == "LMT_MARKETABLE" or queued):
                if q and q.price:
                    bps = (cfg.execution.queued_limit_buffer_bps if queued
                           else cfg.execution.slippage_buffer_bps)
                    buf = bps / 10_000
                    # a queued entry must be a limit order: market orders are not
                    # accepted outside RTH
                    intent.limit_price = round(q.price * (1 + buf), 4)
                    if queued:
                        intent.order_type = "LMT_QUEUED"

        # 2) outside RTH the queue must track this decision, not the one that
        #    created the order: re-price / resize / cancel, and keep the total
        #    queued notional inside available cash.
        clock = ctx.clock or market_clock()
        synced: dict | None = None
        if not clock.is_open and cfg.pipeline.queue_entries_outside_rth:
            buys = [i for i in intents if i.side == "BUY"]
            try:
                synced = await app.orders.sync_queued_entries(
                    buys,
                    available_usd=(ctx.account.available_usd if ctx.account else None),
                )
                ctx.funnel["queued_sync"] = synced
            except Exception as exc:  # noqa: BLE001
                ctx.warnings.append(f"queue sync: {str(exc)[:120]}")
                log.warning("queue sync failed", error=str(exc)[:160])

        # A run with no candidates still has to reconcile the standing queue,
        # otherwise stale or over-committed brackets survive to the open.
        if not intents:
            if synced:
                return (f"queue reconciled: cancelled {synced.get('cancelled', 0)}, "
                        f"modified {synced.get('modified', 0)}, "
                        f"queued ${synced.get('queued_notional', 0):,.0f}")
            return "no intents"

        # 3) submit whatever is not already working
        results = []
        for intent in intents:
            res = await app.orders.submit(intent)
            results.append(f"{res.side} {res.ticker} x{res.qty} -> {res.status}")
            await bus.publish("order.update", ticker=res.ticker, side=res.side, qty=res.qty,
                              status=res.status, run_id=ctx.run_id)
        ctx.funnel["orders_submitted"] = len(results)
        return "; ".join(results[:6])

    # -------------------------------------------------------------------- S7
    async def _s7(self, ctx: RunContext) -> str:
        app = self.app
        acct = ctx.account
        if acct is not None:
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
        await app.update_peak_equity(acct)
        await app.persist_connector_health(ctx.news_batch)
        for w in ctx.warnings[:10]:
            await app.alert("warning", "pipeline", w)
        return "persisted"


def _nan0(value, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:  # noqa: BLE001
        return default
    if not np.isfinite(v):
        return default
    return v


def _expected_r(row: pd.Series, cfg) -> float:
    p = float(row.get("p_win") or 0.5)
    ratio = cfg.risk.tp_atr / max(cfg.risk.sl_atr, 1e-6)
    return p * ratio - (1 - p)


async def count_trades_today() -> int:
    from tradebot.core.clock import ET

    today = now_utc().astimezone(ET).date()
    async with session_scope() as s:
        res = await s.execute(select(Trade).where(func.date(Trade.created_at) == str(today)))
        return len(res.scalars().all())
