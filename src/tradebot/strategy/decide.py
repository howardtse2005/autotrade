"""The single decision function shared by live trading, backtests and replay.

`decide()` is a pure function: no I/O, no clocks, no randomness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from tradebot.core.clock import MarketClock
from tradebot.core.types import (
    AccountState,
    Decision,
    DecisionBundle,
    OrderIntent,
    PositionState,
    TickerSignal,
)
from tradebot.strategy import reason_codes as rc
from tradebot.risk.breakers import BreakerEvaluation, CooldownTracker
from tradebot.risk.sizing import compute_stops, size_position


@dataclass
class PortfolioState:
    positions: list[PositionState] = field(default_factory=list)
    account: AccountState = field(default_factory=AccountState)
    clock: MarketClock | None = None
    trades_today: int = 0
    day_pnl_pct: float = 0.0
    peak_equity: float = 0.0
    breakers: BreakerEvaluation | None = None
    cooldowns: CooldownTracker = field(default_factory=CooldownTracker)
    consecutive_losses: int = 0
    decay_counts: dict[str, int] = field(default_factory=dict)
    mode: str = "dry_run"
    bootstrap: bool = False
    run_id: int | None = None
    allow_queued_entries: bool = False
    # cash already committed to working (queued) BUY orders - IBKR counts open
    # orders against buying power, so new sizing must leave room for them
    reserved_usd: float = 0.0
    # tickers that already have a working queued BUY (they occupy a slot)
    queued_tickers: set[str] = field(default_factory=set)

    @property
    def held(self) -> dict[str, PositionState]:
        return {p.ticker: p for p in self.positions if p.qty > 0 and not p.external}

    @property
    def open_count(self) -> int:
        return len(self.held)


def _z(values: list[float]) -> np.ndarray:
    if not values:
        return np.array([])
    a = np.asarray(values, dtype=float)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if len(a) < 3 or np.allclose(a, a[0]):
        return np.zeros(len(a))
    med = np.median(a)
    iqr = np.percentile(a, 75) - np.percentile(a, 25)
    scale = iqr / 1.349 if iqr > 1e-9 else (a.std() or 1.0)
    return np.clip((a - med) / max(scale, 1e-9), -3, 3)


def _rank_scores(signals: list[TickerSignal], cfg) -> np.ndarray:
    w = cfg.strategy.rank_weights
    p = _z([s.p_win for s in signals])
    e = _z([s.exp_r_per_vol for s in signals])
    sent = _z([s.sent_ewm_240 * s.confidence for s in signals])
    rv = _z([s.rel_volume for s in signals])
    total = sum(w.values()) or 1.0
    return (
        p * w.get("p_win", 0) + e * w.get("exp_r_per_vol", 0)
        + sent * w.get("sentiment", 0) + rv * w.get("rel_volume", 0)
    ) / total


def _entry_gates(
    sig: TickerSignal, state: PortfolioState, cfg, held: dict[str, PositionState],
    thresholds: dict | None = None, apply_pwin: bool = True,
) -> list[str]:
    codes: list[str] = []
    s, r, st = cfg.strategy, cfg.risk, state

    if sig.ticker in held:
        codes.append(rc.SKIP_ALREADY_HELD)
    if st.mode in ("paper", "live") and st.bootstrap:
        codes.append(rc.SKIP_MODEL_UNHEALTHY)
    if st.breakers is not None and st.breakers.entries_blocked:
        codes.append(rc.SKIP_BREAKER)
    if st.clock is not None and cfg.pipeline.market_hours_only:
        if not st.clock.is_open:
            # outside RTH: either block, or let the order wait at the broker for the open
            if not (st.allow_queued_entries and st.mode in ("paper", "live")):
                codes.append(rc.SKIP_MARKET_CLOSED)
        else:
            allowed, _phase = st.clock.allows_new_entries(
                cfg.pipeline.skip_open_minutes, cfg.pipeline.no_new_entries_last_minutes
            )
            if not allowed:
                codes.append(rc.SKIP_TIME_WINDOW)
    if st.cooldowns.blocked(sig.ticker):
        codes.append(rc.SKIP_COOLDOWN)
    if sig.earnings_flag and s.earnings_blackout_days > 0:
        codes.append(rc.SKIP_EARNINGS)
    if sig.halted:
        codes.append(rc.SKIP_HALTED)
    # Structural liquidity, not the current bar: a 15-minute bar's dollar volume
    # is ~1/26 of a day, so gating on it blocks every name in the morning.
    liquidity = sig.avg_dollar_volume or sig.dollar_volume
    if sig.price < cfg.universe.min_price or liquidity < cfg.universe.min_avg_dollar_volume / 2:
        codes.append(rc.SKIP_ILLIQUID)
    max_age = max(2 * cfg.data.data_delay_minutes, 30) * 60
    if sig.quote_age_s is not None and sig.quote_age_s > max_age:
        codes.append(rc.SKIP_STALE_QUOTE)
    if sig.spread_bps is not None and sig.spread_bps > s.max_spread_bps:
        codes.append(rc.SKIP_SPREAD)
    # --- signal-strength gate -------------------------------------------
    # p_win is not stationary: its cross-sectional median moves by ±0.06 within a
    # single session, so a single global cut either blocks everything or nothing.
    # The operating point is therefore the *percentile inside this run's
    # cross-section* (the strategy is a relative-value selector anyway), with the
    # absolute threshold kept as a fallback when the cross-section is too small.
    th = thresholds or {}
    if apply_pwin:
        min_pct = float(th.get("min_bar_pct") or 0.0)
        if th.get("mode") == "cross_sectional" and min_pct > 0 and sig.p_win_pct is not None:
            if sig.p_win_pct < min_pct:
                codes.append(rc.SKIP_LOW_PWIN)
        else:
            buy_th = th.get("buy", s.buy_threshold_p_win)
            if sig.p_win < buy_th:
                codes.append(rc.SKIP_LOW_PWIN)
    # expected_r is already net of trading costs (labels are cost-adjusted).
    # LIVE always enforces the edge gate; PAPER may run in "warn" mode so the
    # system can collect forward data with an unproven model.
    edge_mode = "strict" if st.mode == "live" else cfg.strategy.edge_gate_mode
    if edge_mode == "strict" and sig.expected_r <= 0:
        codes.append(rc.SKIP_NEG_EDGE)
    if s.entry_timing_gate:
        entry_pct = float(th.get("min_entry_pct") or 0.0)
        if th.get("mode") == "cross_sectional" and entry_pct > 0 and sig.p_entry_pct is not None:
            if sig.p_entry_pct < entry_pct:
                codes.append(rc.SKIP_POOR_ENTRY)
        else:
            min_pe = th.get("min_p_entry", s.min_p_entry)
            if sig.p_entry < min_pe:
                codes.append(rc.SKIP_POOR_ENTRY)
    if sig.disagreement > s.max_model_disagreement:
        codes.append(rc.SKIP_DISAGREEMENT)
    if s.sentiment_gate_enabled:
        if sig.sent_ewm_240 < s.min_ewm_240m:
            codes.append(rc.SKIP_NEG_SENTIMENT)
        elif s.block_if_neg_shock and sig.news_neg_240 >= 2 and sig.sent_max_neg_240 <= s.neg_shock_threshold:
            codes.append(rc.SKIP_NEG_SENTIMENT)
    # "no data" means no usable price/volatility - not "this bar had no prints".
    # A 15-minute bar can legitimately have zero volume; the structural
    # liquidity check above already covers genuinely untradeable names.
    if sig.price <= 0 or sig.atr <= 0:
        codes.append(rc.SKIP_NO_DATA)
    del r
    return codes


def _sector_counts(held: dict[str, PositionState]) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in held.values():
        out[p.sector] = out.get(p.sector, 0) + 1
    return out


def _hold_score(sig: TickerSignal | None, pos: PositionState, cfg) -> float:
    if sig is None:
        return 0.0
    w = cfg.strategy.rank_weights
    base = (
        w.get("p_win", 0) * (sig.p_win - 0.5) * 4
        + w.get("exp_r_per_vol", 0) * np.tanh(sig.exp_r_per_vol)
        + w.get("sentiment", 0) * sig.sent_ewm_240 * sig.confidence
        + w.get("rel_volume", 0) * np.tanh(sig.rel_volume - 1)
    )
    return float(base + 0.15 * np.tanh(pos.unrealized_r))


def _exit_reasons(
    pos: PositionState, sig: TickerSignal | None, state: PortfolioState, cfg,
    thresholds: dict | None = None,
) -> list[str]:
    codes: list[str] = []
    if state.breakers is not None and state.breakers.flatten:
        codes.append(rc.SELL_BREAKER)
        return codes
    if state.clock is not None and cfg.pipeline.flatten_eod and state.clock.should_flatten(10):
        codes.append(rc.SELL_EOD)
    if pos.bars_held >= cfg.risk.time_stop_bars:
        codes.append(rc.SELL_TIME)
    if sig is not None:
        th = thresholds or {}
        exit_pct = float(th.get("exit_bar_pct") or 0.0)
        if th.get("mode") == "cross_sectional" and exit_pct > 0 and sig.p_win_pct is not None:
            decayed = sig.p_win_pct <= exit_pct
        else:
            decayed = sig.p_win < th.get("exit", cfg.strategy.exit_threshold_p_win)
        if decayed:
            count = state.decay_counts.get(pos.ticker, 0) + 1
            if count >= 2:
                codes.append(rc.SELL_SIGNAL_DECAY)
        if (
            cfg.strategy.sentiment_gate_enabled
            and sig.news_neg_240 >= 2
            and sig.sent_max_neg_240 <= cfg.strategy.neg_shock_threshold
        ):
            codes.append(rc.SELL_SENT_SHOCK)
    return list(dict.fromkeys(codes))


def decide(
    state: PortfolioState,
    signals: dict[str, TickerSignal],
    cfg,
    universe_size: int = 0,
    thresholds: dict | None = None,
) -> DecisionBundle:
    bundle = DecisionBundle()
    held = state.held
    # tickers that already have a working queued BUY: they occupy a slot and are
    # maintained separately below rather than re-selected as new candidates
    queued_set = {t for t in (state.queued_tickers or set()) if t not in held}
    decisions: list[Decision] = []

    # ------------------------------------------------------------- exits
    for ticker, pos in held.items():
        sig = signals.get(ticker)
        reasons = _exit_reasons(pos, sig, state, cfg, thresholds)
        if reasons:
            decisions.append(
                Decision(
                    ticker=ticker, action="SELL", reason_codes=reasons,
                    planned_qty=int(pos.qty),
                    detail={"exit_reason": reasons[0], "bars_held": pos.bars_held},
                )
            )
            bundle.intents.append(
                OrderIntent(
                    run_id=state.run_id, ticker=ticker, side="SELL", qty=int(pos.qty),
                    order_type="MKT", reason=reasons[0],
                    order_ref=f"{state.run_id or 0}:{ticker}:SELL:{reasons[0]}",
                    source="auto",
                )
            )
        else:
            decisions.append(Decision(ticker=ticker, action="HOLD", reason_codes=[rc.OK_HOLD]))

    # ------------------------------------------------------------ entries
    exiting = {d.ticker for d in decisions if d.action == "SELL"}
    candidates: list[tuple[TickerSignal, list[str]]] = []
    rejected: list[Decision] = []
    for ticker, sig in signals.items():
        if ticker in held or ticker in queued_set:
            continue
        codes = _entry_gates(sig, state, cfg, held, thresholds, apply_pwin=False)
        if codes:
            rejected.append(Decision(ticker=ticker, action="SKIP", reason_codes=codes))
        else:
            candidates.append((sig, []))

    sectors = _sector_counts({k: v for k, v in held.items() if k not in exiting})
    scored: list[tuple[TickerSignal, float]] = []
    if candidates:
        scores = _rank_scores([c[0] for c in candidates], cfg)
        for (sig, _codes), sc in zip(candidates, scores):
            if sectors.get(sig.sector, 0) >= cfg.strategy.max_per_sector:
                rejected.append(
                    Decision(ticker=sig.ticker, action="SKIP", reason_codes=[rc.SKIP_SECTOR_CAP])
                )
                continue
            scored.append((sig, float(sc)))

    scored.sort(key=lambda x: x[1], reverse=True)
    free_slots = max(
        0, cfg.strategy.max_positions - len(held) - len(queued_set) + len(exiting)
    )

    # ------------------------------------------------- p_win quality tier
    # The percentile is a *quality ladder*, not a wall. A fixed "top 10%" only
    # ever admits ~10% of the universe, and once the other gates filter those
    # the book is left half empty. Instead: take the strictest tier that still
    # fills the free slots, and never dip below `quality_floor_pct`.
    th = thresholds or {}
    base_tier = float(th.get("min_bar_pct") or 0.0)
    tier = base_tier
    buy_abs = float(th.get("buy", cfg.strategy.buy_threshold_p_win) or 0.0)

    def _above(sig: TickerSignal, t: float) -> bool:
        """Percentile when the cross-section is big enough, absolute otherwise."""
        if sig.p_win_pct is not None:
            return sig.p_win_pct >= t
        return sig.p_win >= buy_abs if buy_abs > 0 else True

    if th.get("mode") == "cross_sectional" and scored and free_slots > 0:
        floor = float(getattr(cfg.strategy, "quality_floor_pct", 0.0) or 0.0)
        ladder = sorted({base_tier, 0.85, 0.75, 0.65, floor}, reverse=True)
        ladder = [x for x in ladder if x >= floor - 1e-9] or [0.0]
        pool: list[tuple[TickerSignal, float]] = []
        for t in ladder:
            cand = [(sig, sc) for sig, sc in scored if _above(sig, t)]
            if len(cand) >= free_slots or t == ladder[-1]:
                pool, tier = cand, t
                break
        kept = {sig.ticker for sig, _ in pool}
        for sig, sc in scored:
            if sig.ticker not in kept:
                rejected.append(
                    Decision(ticker=sig.ticker, action="SKIP",
                             reason_codes=[rc.SKIP_LOW_PWIN], rank_score=round(sc, 3))
                )
        scored = pool
    elif scored:
        # absolute operating point (no model thresholds / tiny cross-section)
        keep = [(sig, sc) for sig, sc in scored if sig.p_win >= buy_abs]
        for sig, sc in scored:
            if sig.p_win < buy_abs:
                rejected.append(
                    Decision(ticker=sig.ticker, action="SKIP",
                             reason_codes=[rc.SKIP_LOW_PWIN], rank_score=round(sc, 3))
                )
        scored = keep

    # `capacity` is the number of slots we had at the start; `remaining` counts
    # down as names are picked. Comparing len(selected) against a *decrementing*
    # counter (the old behaviour) stops at roughly half the capacity.
    capacity = free_slots
    remaining = free_slots
    rotated = False
    selected: list[tuple[TickerSignal, float]] = []
    for sig, sc in scored:
        # Enforce the per-sector cap across *new* selections too, not just
        # against existing holdings - otherwise one hot sector can fill the book.
        if sectors.get(sig.sector, 0) >= cfg.strategy.max_per_sector:
            rejected.append(
                Decision(ticker=sig.ticker, action="SKIP",
                         reason_codes=[rc.SKIP_SECTOR_CAP], rank_score=round(sc, 3))
            )
            continue
        if remaining <= 0:
            # rotation: displace the weakest holding when clearly better
            if (
                capacity == 0 and not rotated and cfg.strategy.rotation_enabled and held
            ):
                weakest = min(
                    held.values(), key=lambda p: _hold_score(signals.get(p.ticker), p, cfg)
                )
                weakest_score = _hold_score(signals.get(weakest.ticker), weakest, cfg)
                if (
                    weakest.bars_held >= cfg.strategy.min_hold_bars
                    and sc - weakest_score >= cfg.strategy.min_score_gap_z
                    and weakest.ticker not in exiting
                ):
                    exiting.add(weakest.ticker)
                    rotated = True
                    bundle.intents.append(
                        OrderIntent(
                            run_id=state.run_id, ticker=weakest.ticker, side="SELL",
                            qty=int(weakest.qty), order_type="MKT", reason=rc.SELL_ROTATION,
                            order_ref=(f"{state.run_id or 0}:{weakest.ticker}"
                                       f":SELL:{rc.SELL_ROTATION}"),
                            source="auto",
                        )
                    )
                    decisions.append(
                        Decision(ticker=weakest.ticker, action="SELL",
                                 reason_codes=[rc.SELL_ROTATION], planned_qty=int(weakest.qty))
                    )
                    remaining += 1
            if remaining <= 0:
                rejected.append(
                    Decision(ticker=sig.ticker, action="SKIP",
                             reason_codes=[rc.SKIP_NO_SLOT], rank_score=round(sc, 3))
                )
                continue
        selected.append((sig, sc))
        sectors[sig.sector] = sectors.get(sig.sector, 0) + 1
        remaining -= 1

    equity_usd = state.account.net_liq_usd
    # IBKR's available funds already nets out the margin held by working orders,
    # so subtracting the queue's notional on top would double count. The binding
    # constraint is whichever is smaller: available funds, or cash that the queue
    # has not already committed.
    reserved = max(state.reserved_usd, 0.0)
    notional_headroom = max(state.account.cash_usd - reserved, 0.0)
    available_usd = min(max(state.account.available_usd, 0.0), notional_headroom)
    size_mult = state.breakers.size_multiplier if state.breakers else 1.0
    # notional already deployed in open positions (reserve it before sizing new ones)
    committed = sum((p.last or p.avg_cost) * abs(p.qty) for p in held.values())
    exploratory = bool(state.mode != "live" and cfg.strategy.edge_gate_mode == "warn")

    for rank, (sig, sc) in enumerate(selected, start=1):
        sizing = size_position(
            sig, equity_usd, max(available_usd - committed, 0), cfg,
            allow_exploratory=exploratory,
        )
        if sizing.qty > 0:
            sizing.qty = max(1, int(sizing.qty * size_mult))
        if not sizing.ok:
            rejected.append(
                Decision(ticker=sig.ticker, action="SKIP", reason_codes=[rc.SKIP_RISK_TOO_SMALL],
                         rank=rank, rank_score=round(sc, 3), detail={"why": sizing.reason})
            )
            continue
        committed += sizing.qty * sig.price
        is_queued = bool(state.clock is not None and not state.clock.is_open)
        buy_codes = [rc.OK_BUY]
        if cfg.strategy.edge_gate_mode == "warn" and sig.expected_r <= 0:
            buy_codes.append(rc.WARN_NEG_EDGE)
        if is_queued:
            buy_codes.append(rc.WARN_QUEUED_OPEN)
        decisions.append(
            Decision(
                ticker=sig.ticker, action="BUY", reason_codes=buy_codes, rank=rank,
                rank_score=round(sc, 3), planned_qty=sizing.qty, planned_sl=sizing.sl,
                planned_tp=sizing.tp,
                detail={
                    "p_win": round(sig.p_win, 4),
                    "expected_r": round(sig.expected_r, 3),
                    "p_entry": round(sig.p_entry, 4),
                    "disagreement": round(sig.disagreement, 4),
                    "sent_ewm_240": round(sig.sent_ewm_240, 4),
                    "spread_bps": None if sig.spread_bps is None else round(sig.spread_bps, 2),
                    "sizing": sizing.detail,
                },
            )
        )
        bundle.intents.append(
            OrderIntent(
                run_id=state.run_id, ticker=sig.ticker, side="BUY", qty=sizing.qty,
                order_type=cfg.execution.entry_order_type, reason=rc.OK_BUY,
                order_ref=f"{state.run_id or 0}:{sig.ticker}:BUY",
                source="auto", planned_sl=sizing.sl, planned_tp=sizing.tp,
                meta={
                    "sizing": sizing.detail or {},
                    # distances keep protection anchored to the *actual* fill price
                    "sl_dist": round(sizing.sl_distance, 4),
                    "tp_dist": round(sizing.tp_distance, 4),
                    "queued": is_queued,
                },
            )
        )

    # ------------------------------------------------ maintain the queue
    # A queued entry occupies a slot and must track the *current* decision. It is
    # re-emitted here with fresh size/price; if the gates no longer accept it, it
    # is simply not re-emitted and the queue sync cancels the working bracket.
    if queued_set:
        for ticker in sorted(queued_set):
            sig = signals.get(ticker)
            if sig is None:
                continue
            if _entry_gates(sig, state, cfg, held, thresholds, apply_pwin=False):
                continue
            if not _above(sig, tier):
                continue
            sizing = size_position(
                sig, equity_usd, max(available_usd - committed, 0), cfg,
                allow_exploratory=exploratory,
            )
            if sizing.qty > 0:
                sizing.qty = max(1, int(sizing.qty * size_mult))
            if not sizing.ok:
                continue
            committed += sizing.qty * sig.price
            bundle.intents.append(
                OrderIntent(
                    run_id=state.run_id, ticker=ticker, side="BUY", qty=sizing.qty,
                    order_type=cfg.execution.entry_order_type, reason=rc.OK_BUY,
                    order_ref=f"{state.run_id or 0}:{ticker}:BUY",
                    source="auto", planned_sl=sizing.sl, planned_tp=sizing.tp,
                    meta={
                        "sizing": sizing.detail or {},
                        "sl_dist": round(sizing.sl_distance, 4),
                        "tp_dist": round(sizing.tp_distance, 4),
                        "queued": True,
                        "maintain": True,
                    },
                )
            )

    for d in rejected:
        decisions.append(d)

    bundle.decisions = decisions
    bundle.funnel = {
        "universe": universe_size or len(signals),
        "with_signals": len(signals),
        "with_news": sum(1 for s in signals.values() if s.news_count_240 > 0),
        "passed_gates": len(candidates) - sum(
            1 for d in rejected if rc.SKIP_SECTOR_CAP in d.reason_codes
        ),
        "ranked": len(scored),
        "free_slots": free_slots,
        "quality_tier": round(float(tier), 3),
        "tier_relaxed": bool(tier < base_tier - 1e-9),
        "selected": len(selected),
        "queued_maintained": sum(
            1 for i in bundle.intents if (i.meta or {}).get("maintain")
        ),
        "orders": len(bundle.intents),
        "exits": len([d for d in decisions if d.action == "SELL"]),
    }
    return bundle


def preflight_codes(clock: MarketClock | None, cfg) -> list[str]:
    codes: list[str] = []
    if clock is not None and not clock.is_open and cfg.pipeline.market_hours_only:
        codes.append(rc.SKIP_MARKET_CLOSED)
    return codes


def now_ts() -> datetime:
    from tradebot.core.clock import now_utc

    return now_utc()
