"""Position sizing, stop/target distances and trailing logic (pure functions)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tradebot.core.types import PositionState, TickerSignal


@dataclass
class Sizing:
    qty: int = 0
    sl: float = 0.0
    tp: float = 0.0
    sl_distance: float = 0.0
    tp_distance: float = 0.0
    risk_pct: float = 0.0
    risk_usd: float = 0.0
    notional: float = 0.0
    reason: str = ""
    detail: dict | None = None

    @property
    def ok(self) -> bool:
        return self.qty > 0 and not self.reason


def effective_atr(signal: TickerSignal, horizon_bars: int) -> float:
    """Blend the bar ATR with the volatility head (per-bar equivalent).

    `vol_hat` is the forecast volatility *over the horizon* (e.g. 0.062 = 6.2 %),
    so it is divided by sqrt(horizon) to get a per-bar figure. It is clamped to a
    plausible range so a bad model output cannot blow up the stop distance.
    """
    atr = max(signal.atr, 1e-6)
    price = max(signal.price, 1e-6)
    vol_h = float(signal.vol_hat)
    if not math.isfinite(vol_h) or vol_h <= 0:
        return atr
    vol_h = min(max(vol_h, 0.0005), 0.25)
    atr_from_vol = price * vol_h / math.sqrt(max(horizon_bars, 1))
    if not math.isfinite(atr_from_vol) or atr_from_vol <= 0:
        return atr
    return 0.7 * atr + 0.3 * atr_from_vol


def compute_stops(signal: TickerSignal, cfg) -> tuple[float, float, float, float]:
    """Return (sl_price, tp_price, sl_distance, tp_distance)."""
    r = cfg.risk
    price = signal.price
    atr_eff = effective_atr(signal, cfg.ml.horizon_bars)
    neg = min(max(-signal.sent_ewm_240, 0.0), 1.0)
    pos = min(max(signal.sent_ewm_240 * signal.confidence, 0.0), 1.0)

    sl_dist = r.sl_atr * (1 - r.sl_tighten_max * neg) * atr_eff
    sl_dist = min(max(sl_dist, r.min_sl_pct / 100 * price), r.max_sl_pct / 100 * price)
    tp_dist = r.tp_atr * (1 + r.tp_widen_max * pos) * atr_eff
    # keep the target sane relative to the stop and to the price
    # The target must clear the same ratio the sizer requires, otherwise a name
    # is selected and then discarded by its own risk check.
    floor_ratio = float(getattr(r, "min_tp_sl_ratio", 1.2))
    tp_dist = max(tp_dist, sl_dist * floor_ratio)
    tp_dist = min(tp_dist, max(sl_dist * floor_ratio, r.max_sl_pct / 100 * price * 2.5))
    return price - sl_dist, price + tp_dist, sl_dist, tp_dist


def size_position(
    signal: TickerSignal,
    account_net_liq_usd: float,
    available_usd: float,
    cfg,
    allow_exploratory: bool = False,
) -> Sizing:
    r, s = cfg.risk, cfg.strategy
    price = signal.price
    if price <= 0 or account_net_liq_usd <= 0:
        return Sizing(reason="invalid price/equity")

    sl, tp, sl_dist, tp_dist = compute_stops(signal, cfg)
    if sl_dist <= 0:
        return Sizing(reason="zero stop distance")
    ratio = tp_dist / sl_dist
    if ratio < float(getattr(r, "min_tp_sl_ratio", 1.2)) - 1e-9:
        return Sizing(reason="TP/SL ratio below 1.2", sl=sl, tp=tp,
                      sl_distance=sl_dist, tp_distance=tp_dist)

    # Edge multiplier: how far above the entry gate this name sits. With a
    # cross-sectional gate the meaningful scale is the percentile, not an
    # absolute probability (absolute p_win is regime dependent).
    if signal.p_win_pct is not None:
        gate = float(getattr(cfg.strategy, "min_bar_pct", 0.0) or 0.0)
        span = max(1.0 - gate, 1e-6)
        edge_mult = min(max((signal.p_win_pct - gate) / span, 0.0), 1.0)
    else:
        theta = s.buy_threshold_p_win
        edge_mult = min(max((signal.p_win - theta) / max(0.80 - theta, 1e-6), 0.0), 1.0)
    conf_mult = (0.6 + 0.8 * edge_mult) * signal.confidence
    sent_mult = min(max(1 + 0.5 * signal.sent_ewm_240, r.sentiment_size_mult[0]),
                    r.sentiment_size_mult[1])
    f_kelly = signal.p_win - (1 - signal.p_win) / ratio
    risk_pct = min(
        r.risk_per_trade_pct * conf_mult * sent_mult,
        r.kelly_fraction_cap * max(f_kelly, 0.0) * 100,
    )
    risk_pct = max(risk_pct, 0.0)
    sizing_mode = "kelly"
    if risk_pct <= 0:
        # Negative Kelly means the *absolute* probability does not clear the
        # barrier ratio. That is a real veto for LIVE trading. In PAPER with
        # edge_gate_mode=warn we still want forward data, so use a reduced
        # risk-parity budget instead of zeroing the trade - clearly labelled so
        # it is never mistaken for a validated edge.
        if allow_exploratory:
            risk_pct = r.risk_per_trade_pct * 0.5 * conf_mult * sent_mult
            sizing_mode = "exploratory"
        if risk_pct <= 0:
            return Sizing(reason="kelly veto", sl=sl, tp=tp, sl_distance=sl_dist,
                          tp_distance=tp_dist)

    risk_usd = account_net_liq_usd * risk_pct / 100
    qty = int(risk_usd / sl_dist) if sl_dist > 0 else 0
    cash_buffer = available_usd * r.cash_buffer_pct / 100
    caps = {
        "risk": qty,
        "max_position": int(r.max_position_pct / 100 * account_net_liq_usd / price),
        "cash": int(max(available_usd - cash_buffer, 0) / price),
        # Participation is a share of a normal *bar*, so use the average bar's
        # dollar volume; the current bar is ~1/26 of a day in the morning and
        # would zero out every order.
        "participation": int(
            r.participation_cap_pct / 100
            * max(signal.dollar_volume or 0.0,
                  (signal.avg_dollar_volume or 0.0) / 26.0, 0.0)
            / price
        ),
        "order_notional": int(r.max_order_notional / price),
    }
    qty = min(caps.values())
    binding = min(caps, key=lambda k: caps[k])
    if qty <= 0 or qty * price < r.min_notional:
        return Sizing(reason=f"size below minimum ({binding})", sl=sl, tp=tp,
                      sl_distance=sl_dist, tp_distance=tp_dist)

    return Sizing(
        qty=qty, sl=round(sl, 2), tp=round(tp, 2), sl_distance=sl_dist, tp_distance=tp_dist,
        risk_pct=risk_pct, risk_usd=qty * sl_dist, notional=qty * price,
        detail={
            "edge_mult": round(edge_mult, 3),
            "conf_mult": round(conf_mult, 3),
            "sent_mult": round(sent_mult, 3),
            "f_kelly": round(f_kelly, 3),
            "risk_pct": round(risk_pct, 3),
            "binding_cap": binding,
            "atr": round(signal.atr, 4),
            "atr_eff": round(effective_atr(signal, cfg.ml.horizon_bars), 4),
            "tp_sl_ratio": round(ratio, 2),
            "sizing_mode": sizing_mode,
        },
    )


def new_trailing_stop(position: PositionState, atr: float, cfg) -> float | None:
    """Ratchet-only trailing stop once the trade is +trigger_R in profit."""
    if position.last is None or position.sl is None or atr <= 0:
        return None
    risk = position.avg_cost - position.sl
    if risk <= 0:
        return None
    r_mult = (position.last - position.avg_cost) / risk
    if r_mult < cfg.risk.trail_trigger_r:
        return None
    breakeven = position.avg_cost * (1 + 0.0005)
    trail = (position.highest_close or position.last) - cfg.risk.trail_atr_mult * atr
    candidate = max(breakeven, trail, position.sl)
    if candidate <= position.sl + 1e-6:
        return None
    return round(candidate, 2)


def time_stop_hit(position: PositionState, cfg) -> bool:
    return position.bars_held >= cfg.risk.time_stop_bars
