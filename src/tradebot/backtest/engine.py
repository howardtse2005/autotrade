"""Vectorised backtest: predict the panel, then run the shared decision function.

The simulation is bar-by-bar and portfolio-constrained (max positions, cash,
sector caps, cooldowns), so results are directly comparable to live behaviour.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradebot.core.clock import now_utc
from tradebot.core.logging import get_logger
from tradebot.core.types import AccountState, PositionState, TickerSignal
from tradebot.db.models import Backtest
from tradebot.db.session import session_scope
from tradebot.features.builder import build_panel
from tradebot.risk.breakers import CooldownTracker
from tradebot.risk.sizing import compute_stops, size_position
from tradebot.strategy import reason_codes as rc
from tradebot.strategy.decide import PortfolioState, decide

log = get_logger("backtest")


@dataclass
class SimPosition:
    ticker: str
    qty: int
    entry_price: float
    entry_ts: pd.Timestamp
    sl: float
    tp: float
    risk_per_share: float
    entry_idx: int
    entry_cost: float = 0.0
    sector: str = "Unknown"
    highest: float = 0.0
    bars_held: int = 0


@dataclass
class SimResult:
    equity: list[tuple[str, float]] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _costs(cfg, price: float, qty: int, spread_bps: float | None) -> float:
    c = cfg.costs
    comm = max(c.commission_per_share * qty, c.commission_min)
    comm = min(comm, price * qty * c.commission_max_pct / 100)
    slip_bps = c.slippage_bps + (spread_bps / 2 if (c.half_spread_from_data and spread_bps) else 0)
    slip = price * qty * slip_bps / 10_000
    return comm + slip


def simulate(
    preds: pd.DataFrame,
    bars_by_ticker: dict[str, pd.DataFrame],
    cfg,
    initial_cash: float | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    thresholds: dict | None = None,
    expected_r_fn=None,
) -> SimResult:
    initial_cash = initial_cash or cfg.backtest.initial_cash
    lag = cfg.data.execution_lag_bars
    horizon = cfg.ml.horizon_bars

    # timestamp -> {ticker: row}
    pdf = preds.copy()
    pdf["bar_ts"] = pd.to_datetime(pdf["bar_ts"])
    if start is not None:
        pdf = pdf[pdf["bar_ts"] >= start]
    if end is not None:
        pdf = pdf[pdf["bar_ts"] <= end]
    if pdf.empty:
        return SimResult(metrics={"error": "no predictions in range"})

    times = np.sort(pdf["bar_ts"].unique())
    by_time = {t: g.set_index("ticker") for t, g in pdf.groupby("bar_ts")}
    index_maps = {t: {ts: i for i, ts in enumerate(df.index)} for t, df in bars_by_ticker.items()}
    arrays = {
        t: (df["open"].to_numpy(float), df["high"].to_numpy(float),
            df["low"].to_numpy(float), df["close"].to_numpy(float))
        for t, df in bars_by_ticker.items()
    }
    bar_index = {t: df.index for t, df in bars_by_ticker.items()}
    session_ids = {
        t: np.array([ts.tz_convert("America/New_York").date().toordinal() for ts in df.index])
        for t, df in bars_by_ticker.items()
    }

    cash = initial_cash
    positions: dict[str, SimPosition] = {}
    equity_curve: list[tuple[str, float]] = []
    trades: list[dict] = []
    cooldowns = CooldownTracker()
    equity_peak = initial_cash
    day_start_equity = initial_cash
    current_day = None
    halted_days: set = set()

    for t in times:
        ts = pd.Timestamp(t)
        day = ts.tz_convert("America/New_York").date()

        # ---- mark to market
        mtm = cash
        for p in positions.values():
            o, h, lo, c = arrays[p.ticker]
            idx = index_maps[p.ticker].get(ts)
            px = float(c[idx]) if idx is not None else p.entry_price
            mtm += p.qty * px

        if current_day != day:
            current_day = day
            day_start_equity = mtm
        equity_peak = max(equity_peak, mtm)
        day_pnl_pct = (mtm / day_start_equity - 1) * 100 if day_start_equity else 0.0

        # ---- exits on this bar (stop / target / time / EOD)
        for ticker in list(positions):
            p = positions[ticker]
            o, h, lo, c = arrays[ticker]
            idx = index_maps[ticker].get(ts)
            if idx is None:
                continue
            p.bars_held += 1
            p.highest = max(p.highest, float(h[idx]))
            exit_price = None
            reason = ""
            if lo[idx] <= p.sl and h[idx] >= p.tp:
                exit_price, reason = p.sl, rc.SELL_STOP  # conservative: stop first
            elif lo[idx] <= p.sl:
                exit_price = min(p.sl, float(o[idx])) if o[idx] < p.sl else p.sl
                reason = rc.SELL_STOP
            elif h[idx] >= p.tp:
                exit_price = max(p.tp, float(o[idx])) if o[idx] > p.tp else p.tp
                reason = rc.SELL_TP
            elif p.bars_held >= horizon:
                exit_price, reason = float(c[idx]), rc.SELL_TIME
            elif cfg.pipeline.flatten_eod:
                sess = session_ids[ticker]
                nxt = idx + 1
                if nxt >= len(sess) or sess[nxt] != sess[idx]:
                    exit_price, reason = float(c[idx]), rc.SELL_EOD
            if exit_price is not None:
                cost = _costs(cfg, exit_price, p.qty, None)
                pnl = (exit_price - p.entry_price) * p.qty - cost
                cash += p.qty * exit_price - cost
                r_mult = pnl / max(p.risk_per_share * p.qty, 1e-9)
                trades.append({
                    "ticker": ticker, "qty": p.qty, "entry_ts": str(p.entry_ts),
                    "entry_price": round(p.entry_price, 3), "exit_ts": str(ts),
                    "exit_price": round(exit_price, 3), "pnl": round(pnl, 2),
                    "pnl_pct": round((exit_price / p.entry_price - 1) * 100, 3),
                    "r_multiple": round(r_mult, 3), "exit_reason": reason,
                    "hold_bars": p.bars_held, "sector": p.sector,
                    "cost": round(cost + p.entry_cost, 4),
                })
                if reason == rc.SELL_STOP:
                    cooldowns.add(ticker, cfg.risk.cooldown_after_stop_minutes)
                positions.pop(ticker, None)

        equity_curve.append((str(ts), round(mtm, 2)))

        # ---- new entries
        if day_pnl_pct <= -abs(cfg.risk.daily_loss_limit_pct):
            halted_days.add(day)
        if day in halted_days:
            continue

        rows = by_time.get(t)
        if rows is None or rows.empty:
            continue
        # percentile over the full cross-section (held names included), exactly
        # as the live pipeline does, so the gate is evaluated on the same basis
        try:
            pct_map = rows["p_win"].rank(pct=True, method="average")
            entry_map = rows["p_entry"].rank(pct=True, method="average")
        except Exception:  # noqa: BLE001
            pct_map = None
            entry_map = None
        signals: dict[str, TickerSignal] = {}
        for ticker, r in rows.iterrows():
            if ticker in positions:
                continue
            o, h, lo, c = arrays.get(ticker, (None,) * 4)
            if o is None:
                continue
            idx = index_maps[ticker].get(ts)
            if idx is None:
                continue
            price = float(c[idx])
            atr = float(r.get("atr14") or 0)
            if price <= 0 or atr <= 0:
                continue
            signals[ticker] = TickerSignal(
                ticker=ticker, bar_ts=ts.to_pydatetime(), price=price, atr=atr,
                atr_pct=float(r.get("atr_pct") or 0),
                p_win=float(r.get("p_win") or 0.5),
                p_win_pct=(float(pct_map.get(ticker)) if pct_map is not None
                           and pct_map.get(ticker) is not None else None),
                expected_r=float(
                    expected_r_fn(np.array([float(r.get("p_win") or 0.5)]))[0]
                    if expected_r_fn is not None
                    else _expected_r(float(r.get("p_win") or 0.5), cfg)
                ),
                vol_hat=float(r.get("vol_hat") or 0.01),
                p_entry=float(r.get("p_entry") or 0.5),
                p_entry_pct=(float(entry_map.get(ticker)) if entry_map is not None
                             and entry_map.get(ticker) is not None else None),
                disagreement=float(r.get("disagreement") or 0),
                sent_ewm_240=_f(r.get("sent_ewm_240m")),
                sent_max_neg_240=_f(r.get("sent_max_neg_240m")),
                news_count_240=int(_f(r.get("news_count_240m"))),
                news_neg_240=int(_f(r.get("news_neg_240m"))),
                rel_volume=_f(r.get("rel_volume"), 1.0),
                dollar_volume=_f(r.get("dollar_volume")),
                avg_dollar_volume=_f(r.get("dollar_volume")) * 26.0,
                spread_bps=None,
                quote_age_s=0.0,
                sector=str(r.get("sector") or "Unknown"),
            )

        if not signals:
            continue

        pos_states = [
            PositionState(
                ticker=p.ticker, qty=p.qty, avg_cost=p.entry_price, last=float(
                    arrays[p.ticker][3][index_maps[p.ticker].get(ts, p.entry_idx)]
                ), sl=p.sl, tp=p.tp, bars_held=p.bars_held, sector=p.sector,
            )
            for p in positions.values()
        ]
        acct = AccountState(currency="USD", net_liq=mtm, cash=cash, available_funds=cash,
                            buying_power=cash * 2, excess_liquidity=mtm * 0.5, fx_to_usd=1.0)
        state = PortfolioState(
            positions=pos_states, account=acct, clock=None, breakers=None,
            cooldowns=cooldowns, mode="paper", run_id=None,
        )
        bundle = decide(state, signals, cfg, universe_size=len(signals), thresholds=thresholds)
        for intent in bundle.intents:
            if intent.side != "BUY" or intent.ticker in positions:
                continue
            if len(positions) >= cfg.strategy.max_positions:
                break
            ticker = intent.ticker
            o, h, lo, c = arrays[ticker]
            idx = index_maps[ticker].get(ts)
            if idx is None:
                continue
            fill_idx = idx + 1 + lag
            if fill_idx >= len(o):
                continue
            fill = float(o[fill_idx])
            qty = intent.qty
            cost = _costs(cfg, fill, qty, None)
            if qty * fill + cost > cash:
                qty = int((cash - cost) / fill)
            if qty <= 0:
                continue
            cash -= qty * fill + cost
            sl, tp, sl_dist, _tp_dist = compute_stops(signals[ticker], cfg)
            entry_ts = bar_index[ticker][fill_idx]
            positions[ticker] = SimPosition(
                ticker=ticker, qty=qty, entry_price=fill, entry_ts=pd.Timestamp(entry_ts),
                sl=sl, tp=tp, risk_per_share=max(sl_dist, 1e-6),
                entry_idx=fill_idx, sector=signals[ticker].sector, entry_cost=cost,
            )

    final_equity = cash + sum(
        p.qty * float(arrays[p.ticker][3][-1]) for p in positions.values()
    )
    metrics = _metrics(equity_curve, trades, initial_cash, final_equity, cfg)
    return SimResult(equity=equity_curve, trades=trades, metrics=metrics)


def _f(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:  # noqa: BLE001
        return default


def _expected_r(p: float, cfg) -> float:
    ratio = cfg.risk.tp_atr / max(cfg.risk.sl_atr, 1e-6)
    return p * ratio - (1 - p)


def _metrics(equity: list[tuple[str, float]], trades: list[dict], initial: float,
             final: float, cfg) -> dict:
    if not equity:
        return {"error": "no equity points"}
    vals = np.array([e[1] for e in equity], dtype=float)
    rets = np.diff(vals) / np.maximum(vals[:-1], 1e-9)
    total_ret = (final / initial - 1) * 100
    peak = np.maximum.accumulate(vals)
    dd = (vals - peak) / np.maximum(peak, 1e-9)
    max_dd = float(dd.min() * 100)
    # intraday-ish annualisation: 26 bars/day * 252 days
    ann = math.sqrt(26 * 252)
    sharpe = float(rets.mean() / rets.std() * ann) if rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std() * ann) if len(downside) > 1 and downside.std() > 0 else 0.0

    pnls = np.array([t["pnl"] for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = float(len(wins) / len(pnls) * 100) if len(pnls) else 0.0
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else (
        float("inf") if len(wins) else 0.0
    )
    rs = np.array([t["r_multiple"] for t in trades], dtype=float)
    exit_counts: dict[str, int] = {}
    for t in trades:
        exit_counts[t["exit_reason"]] = exit_counts.get(t["exit_reason"], 0) + 1
    return {
        "total_return_pct": round(total_ret, 2),
        "final_equity": round(final, 2),
        "initial_cash": round(initial, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "trades": len(trades),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 3) if math.isfinite(profit_factor) else None,
        "avg_r": round(float(rs.mean()), 3) if len(rs) else 0.0,
        "expectancy_r": round(float(rs.mean()), 3) if len(rs) else 0.0,
        "avg_hold_bars": round(float(np.mean([t["hold_bars"] for t in trades])), 1) if trades else 0.0,
        "total_costs": round(float(sum(t.get("cost", 0.0) for t in trades)), 2),
        "exit_breakdown": exit_counts,
        "bars": len(equity),
        "period_start": equity[0][0],
        "period_end": equity[-1][0],
    }


async def run_backtest(app, config: dict) -> int:
    """Build predictions for the universe, simulate, persist, return the backtest id."""
    cfg = app.cfg
    days = int(config.get("days", cfg.backtest.days))
    initial = float(config.get("initial_cash", cfg.backtest.initial_cash))
    use_model = bool(config.get("use_model", True))
    t0 = time.time()

    async with session_scope() as s:
        row = Backtest(status="running", config={"days": days, "initial_cash": initial,
                                                 "use_model": use_model})
        s.add(row)
        await s.flush()
        bt_id = row.id

    try:
        positions = await app.get_positions()
        held = [p.ticker for p in positions if p.qty > 0]
        universe = await app.universe.get(held=held)
        tickers = [r.ticker for r in universe]
        sector_map = {r.ticker: r.sector for r in universe}
        await app.bars.get_bars(tickers)
        await app.bars.get_daily(tickers + cfg.universe.benchmarks + ["^VIX"])
        bars = {t: app.bars.cached_bars(t) for t in tickers if not app.bars.cached_bars(t).empty}
        daily = {t: app.bars.cached_bars(t, 1440, "daily") for t in tickers}
        spy = app.bars.cached_bars(cfg.universe.benchmarks[0], 1440, "daily")
        vix = app.bars.cached_bars("^VIX", 1440, "daily")

        panel = await asyncio.to_thread(build_panel, bars, cfg, daily, spy, vix)
        if panel.frame.empty:
            raise RuntimeError("no feature rows")

        frame = panel.frame.copy()
        frame["bar_ts"] = pd.to_datetime(frame.index)
        frame["sector"] = frame["ticker"].map(sector_map).fillna("Unknown")
        frame = frame.reset_index(drop=True)

        cutoff = frame["bar_ts"].max() - pd.Timedelta(days=days)
        window = frame[frame["bar_ts"] >= cutoff].copy()
        if window.empty:
            raise RuntimeError("no data in the requested window")

        if use_model and app.predictor.ready:
            preds = app.predictor.predict_batch(window)
            warnings = []
            train_end = (app.predictor.meta or {}).get("train_end")
            if train_end and pd.Timestamp(train_end) >= window["bar_ts"].min():
                warnings.append(
                    f"Model trained through {pd.Timestamp(train_end).date()} — results before "
                    "that date are in-sample."
                )
        else:
            from tradebot.models.predict import bootstrap_predict

            preds = bootstrap_predict(window, cfg)
            preds["ticker"] = window["ticker"].to_numpy()
            preds["bar_ts"] = window["bar_ts"].to_numpy()
            preds["sector"] = window["sector"].to_numpy()
            warnings = ["Rule-based bootstrap signals (no champion model)"]

        preds["sector"] = window["sector"].to_numpy()
        result = await asyncio.to_thread(
            simulate, preds, bars, cfg, initial, window["bar_ts"].min(), window["bar_ts"].max(),
            app.predictor.effective_thresholds() if app.predictor.ready else None,
            app.predictor.expected_r if app.predictor.ready else None,
        )
        result.warnings.extend(warnings)
        result.metrics["duration_s"] = round(time.time() - t0, 1)
        result.metrics["warnings"] = result.warnings
        result.metrics["use_model"] = bool(use_model and app.predictor.ready)
        result.metrics["model_id"] = app.predictor.model_id if use_model else "bootstrap"

        async with session_scope() as s:
            row = await s.get(Backtest, bt_id)
            if row is not None:
                row.status = "ok"
                row.metrics = result.metrics
                row.equity_curve = [{"ts": ts, "equity": eq} for ts, eq in result.equity]
                row.trades = result.trades[:2000]
                row.duration_ms = int((time.time() - t0) * 1000)
        return bt_id
    except Exception as exc:  # noqa: BLE001
        log.exception("backtest failed")
        async with session_scope() as s:
            row = await s.get(Backtest, bt_id)
            if row is not None:
                row.status = "error"
                row.error = f"{type(exc).__name__}: {exc}"[:500]
                row.duration_ms = int((time.time() - t0) * 1000)
        raise
