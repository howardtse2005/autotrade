"""Portfolio-level risk breakers (daily loss, drawdown, trade count, data health)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from tradebot.core.clock import now_utc
from tradebot.core.types import AccountState, BreakerState


@dataclass
class BreakerInputs:
    account: AccountState
    day_pnl_pct: float
    peak_equity: float
    trades_today: int
    consecutive_losses: int = 0
    data_ok: bool = True
    data_detail: str = ""
    model_ok: bool = True
    model_detail: str = ""
    manual_halt: bool = False
    drawdown_halt: bool = False


@dataclass
class BreakerEvaluation:
    state: BreakerState
    entries_blocked: bool = False
    flatten: bool = False
    reasons: list[str] = field(default_factory=list)
    size_multiplier: float = 1.0

    @property
    def ok(self) -> bool:
        return not self.state.tripped


def evaluate(inputs: BreakerInputs, cfg) -> BreakerEvaluation:
    r = cfg.risk
    reasons: list[str] = []
    tripped = False
    flatten = False

    if inputs.manual_halt:
        reasons.append("manual halt / kill switch")
        tripped = True
        flatten = True

    if inputs.day_pnl_pct <= -abs(r.daily_loss_limit_pct):
        reasons.append(f"daily loss {inputs.day_pnl_pct:.2f}% <= -{r.daily_loss_limit_pct}%")
        tripped = True
        flatten = True

    dd_pct = 0.0
    if inputs.peak_equity > 0:
        dd_pct = (inputs.account.net_liq - inputs.peak_equity) / inputs.peak_equity * 100
        if inputs.drawdown_halt or dd_pct <= -abs(r.max_drawdown_halt_pct):
            reasons.append(f"drawdown {dd_pct:.2f}% <= -{r.max_drawdown_halt_pct}%")
            tripped = True

    if inputs.trades_today >= r.max_trades_per_day:
        reasons.append(f"max trades/day reached ({inputs.trades_today})")
        tripped = True

    if not inputs.data_ok:
        reasons.append(f"data health: {inputs.data_detail or 'stale'}")
        tripped = True

    if not inputs.model_ok:
        reasons.append(f"model health: {inputs.model_detail or 'unhealthy'}")
        tripped = True

    if r.intraday_margin_guard and inputs.account.net_liq > 0:
        if inputs.account.excess_liq_pct < r.min_excess_liquidity_pct:
            reasons.append(
                f"excess liquidity {inputs.account.excess_liq_pct:.1f}% < {r.min_excess_liquidity_pct}%"
            )
            tripped = True

    size_mult = 1.0
    if inputs.consecutive_losses >= 4:
        size_mult = 0.5
        reasons.append("halved size after 4 consecutive losses")

    state = BreakerState(
        tripped=tripped,
        kind=", ".join(reasons[:2]) if reasons else "",
        detail="; ".join(reasons),
        since=now_utc() if tripped else None,
        entries_blocked=tripped,
    )
    return BreakerEvaluation(
        state=state,
        entries_blocked=tripped,
        flatten=flatten,
        reasons=reasons,
        size_multiplier=size_mult,
    )


@dataclass
class CooldownTracker:
    """Ticker-level cooldown after a stop-out."""

    until: dict[str, datetime] = field(default_factory=dict)

    def add(self, ticker: str, minutes: int) -> None:
        self.until[ticker] = now_utc() + timedelta(minutes=minutes)

    def blocked(self, ticker: str, now: datetime | None = None) -> bool:
        now = now or now_utc()
        ts = self.until.get(ticker)
        return ts is not None and ts > now

    def prune(self) -> None:
        now = now_utc()
        self.until = {k: v for k, v in self.until.items() if v > now}
