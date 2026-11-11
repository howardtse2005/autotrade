"""Strategy and risk invariants (the plan's property tests)."""

from __future__ import annotations

import random
from datetime import timedelta

import pandas as pd
import pytest

from tradebot.core.clock import market_clock, now_utc
from tradebot.core.types import AccountState, PositionState, TickerSignal
from tradebot.risk.breakers import BreakerEvaluation, CooldownTracker
from tradebot.risk.sizing import compute_stops, new_trailing_stop, size_position
from tradebot.settings import load_config
from tradebot.strategy.decide import PortfolioState, decide


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _signal(ticker="AAA", p_win=0.55, price=50.0, atr=1.0, **kw) -> TickerSignal:
    base = dict(
        ticker=ticker, bar_ts=now_utc(), price=price, atr=atr, atr_pct=atr / price,
        p_win=p_win, expected_r=p_win * 1.5 - (1 - p_win), vol_hat=0.01, p_entry=0.6,
        disagreement=0.05, sent_ewm_240=0.2, sent_max_neg_240=-0.1, news_count_240=3,
        rel_volume=1.5, dollar_volume=5e8, spread_bps=3.0, quote_age_s=30.0,
        sector="Technology",
    )
    base.update(kw)
    return TickerSignal(**base)


def _account(net_liq=100_000, cash=100_000) -> AccountState:
    return AccountState(currency="USD", net_liq=net_liq, cash=cash, available_funds=cash,
                        buying_power=cash * 2, excess_liquidity=net_liq * 0.5, fx_to_usd=1.0)


def _open_clock():
    return market_clock(pd.Timestamp("2026-11-06 16:00:00+00:00").to_pydatetime())


def _state(cfg, positions=None, **kw) -> PortfolioState:
    return PortfolioState(
        positions=positions or [], account=_account(), clock=_open_clock(),
        breakers=kw.pop("breakers", BreakerEvaluation(state=__import__(
            "tradebot.core.types", fromlist=["BreakerState"]).BreakerState())),
        cooldowns=CooldownTracker(), mode="paper", **kw,
    )


def test_stops_bracket_price(cfg):
    for _ in range(200):
        price = random.uniform(5, 500)
        atr = price * random.uniform(0.002, 0.05)
        sig = _signal(price=price, atr=atr)
        sl, tp, sl_d, tp_d = compute_stops(sig, cfg)
        assert sl < price < tp
        assert sl_d > 0 and tp_d > sl_d
        assert sl_d >= cfg.risk.min_sl_pct / 100 * price - 1e-9
        assert sl_d <= cfg.risk.max_sl_pct / 100 * price + 1e-9


def test_sizing_respects_caps(cfg):
    sig = _signal(price=50, atr=1.0)
    s = size_position(sig, 100_000, 100_000, cfg)
    assert s.qty > 0
    assert s.notional <= cfg.risk.max_order_notional + 1e-6
    assert s.notional <= cfg.risk.max_position_pct / 100 * 100_000 + 1e-6
    assert s.risk_usd <= 100_000 * cfg.risk.risk_per_trade_pct / 100 * 1.31 + 1e-6


def test_sizing_zero_when_no_cash(cfg):
    sig = _signal()
    s = size_position(sig, 100_000, 0.0, cfg)
    assert s.qty == 0


def test_decide_never_exceeds_max_positions(cfg):
    signals = {f"T{i}": _signal(f"T{i}", p_win=0.60 + i * 0.005) for i in range(20)}
    bundle = decide(_state(cfg), signals, cfg, universe_size=20)
    buys = [d for d in bundle.decisions if d.action == "BUY"]
    assert len(buys) <= cfg.strategy.max_positions
    assert len([i for i in bundle.intents if i.side == "BUY"]) == len(buys)


def test_decide_skips_low_probability(cfg):
    signals = {"AAA": _signal(p_win=0.05)}
    bundle = decide(_state(cfg), signals, cfg)
    d = bundle.decisions[0]
    assert d.action == "SKIP"
    assert "SKIP_LOW_PWIN" in d.reason_codes


def test_decide_blocks_when_breaker_tripped(cfg):
    from tradebot.core.types import BreakerState

    br = BreakerEvaluation(state=BreakerState(tripped=True, entries_blocked=True, detail="test"),
                           entries_blocked=True)
    bundle = decide(_state(cfg, breakers=br), {"AAA": _signal(p_win=0.9)}, cfg)
    assert all(d.action != "BUY" for d in bundle.decisions)
    assert "SKIP_BREAKER" in bundle.decisions[0].reason_codes


def test_held_ticker_is_not_rebought(cfg):
    pos = PositionState(ticker="AAA", qty=100, avg_cost=50, sl=48, tp=53, sector="Technology")
    bundle = decide(_state(cfg, positions=[pos]), {"AAA": _signal(p_win=0.9)}, cfg)
    assert all(d.action != "BUY" for d in bundle.decisions)


def test_time_stop_triggers_sell(cfg):
    pos = PositionState(ticker="AAA", qty=100, avg_cost=50, sl=48, tp=53,
                        bars_held=cfg.risk.time_stop_bars, sector="Technology")
    bundle = decide(_state(cfg, positions=[pos]), {"AAA": _signal(p_win=0.9)}, cfg)
    sells = [d for d in bundle.decisions if d.action == "SELL"]
    assert sells and "SELL_TIME" in sells[0].reason_codes


def test_trailing_stop_only_ratchets_up(cfg):
    pos = PositionState(ticker="AAA", qty=100, avg_cost=50, sl=48, tp=60, last=53.5,
                        highest_close=53.5)
    first = new_trailing_stop(pos, atr=1.0, cfg=cfg)
    assert first is not None and first > pos.sl
    pos.sl = first
    pos.last = 53.2
    second = new_trailing_stop(pos, atr=1.0, cfg=cfg)
    assert second is None or second >= first


def test_sector_cap(cfg):
    positions = [
        PositionState(ticker=f"S{i}", qty=10, avg_cost=50, sl=48, tp=53, sector="Tech")
        for i in range(cfg.strategy.max_per_sector)
    ]
    signals = {f"S{i}": _signal(f"S{i}", p_win=0.7) for i in range(cfg.strategy.max_per_sector)}
    signals["NEW"] = _signal("NEW", p_win=0.9, sector="Tech")
    bundle = decide(_state(cfg, positions=positions), signals, cfg)
    new = [d for d in bundle.decisions if d.ticker == "NEW"]
    assert new and "SKIP_SECTOR_CAP" in new[0].reason_codes


def test_cooldown_blocks_entry(cfg):
    state = _state(cfg)
    state.cooldowns.add("AAA", 60)
    bundle = decide(state, {"AAA": _signal(p_win=0.9)}, cfg)
    assert "SKIP_COOLDOWN" in bundle.decisions[0].reason_codes


def test_market_closed_blocks_entries(cfg):
    cfg2 = cfg.model_copy(deep=True)
    cfg2.pipeline.market_hours_only = True
    closed = market_clock(pd.Timestamp("2026-11-07 03:00:00+00:00").to_pydatetime())
    state = _state(cfg2)
    state.clock = closed
    bundle = decide(state, {"AAA": _signal(p_win=0.9)}, cfg2)
    assert "SKIP_MARKET_CLOSED" in bundle.decisions[0].reason_codes


# --------------------------------------------------------------- slot filling
_SECTORS = ["Technology", "Finance", "Healthcare", "Energy", "Industrials",
            "Utilities", "Materials", "Consumer"]


def _cross_sectional(pcts: list[float], base=0.90, floor=0.60) -> dict:
    """Signals with the given p_win percentiles (all other gates passing).

    Sectors rotate so the per-sector cap does not mask the tier behaviour.
    """
    return {
        f"S{i:02d}": _signal(f"S{i:02d}", p_win=0.20 + pct * 0.1, p_win_pct=pct,
                             p_entry_pct=0.8, avg_dollar_volume=5e8,
                             sector=_SECTORS[i % len(_SECTORS)])
        for i, pct in enumerate(pcts)
    }


def _thresholds(base=0.90, floor=0.60) -> dict:
    return {"mode": "cross_sectional", "min_bar_pct": base, "exit_bar_pct": 0.35,
            "min_entry_pct": 0.30, "buy": 0.3563, "exit": 0.2858, "min_p_entry": 0.28,
            "source": "model"}


def test_tier_keeps_the_strict_cut_when_it_fills_the_book(cfg):
    """Six names in the top decile -> the strict tier is enough for 5 slots."""
    pcts = [0.99, 0.97, 0.95, 0.93, 0.91, 0.89, 0.80, 0.75, 0.70, 0.65]
    bundle = decide(_state(cfg), _cross_sectional(pcts), cfg, universe_size=10,
                    thresholds=_thresholds())
    assert bundle.funnel["quality_tier"] == pytest.approx(0.90)
    assert bundle.funnel["tier_relaxed"] is False
    assert bundle.funnel["selected"] == cfg.strategy.max_positions


def test_tier_relaxes_to_fill_slots(cfg):
    """Only two names clear the top decile -> relax until 5 slots are filled."""
    pcts = [0.99, 0.95, 0.88, 0.84, 0.78, 0.72, 0.68, 0.62, 0.40, 0.20]
    bundle = decide(_state(cfg), _cross_sectional(pcts), cfg, universe_size=10,
                    thresholds=_thresholds())
    assert bundle.funnel["selected"] == cfg.strategy.max_positions
    assert bundle.funnel["quality_tier"] < 0.90
    assert bundle.funnel["tier_relaxed"] is True
    # the bottom of the cross-section is still refused
    skips = [d for d in bundle.decisions if d.action == "SKIP"]
    assert any("SKIP_LOW_PWIN" in d.reason_codes for d in skips)


def test_tier_never_goes_below_the_quality_floor(cfg):
    """Nothing above the floor -> take what exists, never the junk below it."""
    pcts = [0.85, 0.72, 0.65, 0.55, 0.40, 0.30, 0.20, 0.10]
    bundle = decide(_state(cfg), _cross_sectional(pcts), cfg, universe_size=8,
                    thresholds=_thresholds(floor=0.60))
    assert bundle.funnel["quality_tier"] == pytest.approx(0.60)
    assert bundle.funnel["selected"] == 3  # 0.85, 0.72, 0.65
    bought = {d.ticker for d in bundle.decisions if d.action == "BUY"}
    assert len(bought) == 3


def test_tier_respects_existing_positions(cfg):
    """Free slots are max_positions minus what is already held."""
    held = [
        PositionState(ticker=f"H{i}", qty=10, avg_cost=50, sl=48, tp=53, sector="Finance")
        for i in range(3)
    ]
    pcts = [0.99, 0.97, 0.95, 0.93, 0.91, 0.89, 0.80, 0.70]
    bundle = decide(_state(cfg, positions=held), _cross_sectional(pcts), cfg,
                    universe_size=8, thresholds=_thresholds())
    assert bundle.funnel["free_slots"] == cfg.strategy.max_positions - 3
    assert bundle.funnel["selected"] == cfg.strategy.max_positions - 3
