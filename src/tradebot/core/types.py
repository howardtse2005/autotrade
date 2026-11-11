"""Domain types shared by every layer (pure data, no I/O)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

Side = Literal["BUY", "SELL"]
Action = Literal["BUY", "SELL", "HOLD", "SKIP"]


# --------------------------------------------------------------------- market
@dataclass(slots=True)
class Quote:
    ticker: str
    ts: datetime
    last: float | None = None
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    source: str = "unknown"
    market_data_type: int = 0
    prev_close: float | None = None

    @property
    def mid(self) -> float | None:
        if self.bid and self.ask and self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def price(self) -> float | None:
        return self.last or self.mid

    @property
    def spread_bps(self) -> float | None:
        if not (self.bid and self.ask and self.bid > 0 and self.ask > 0):
            return None
        mid = (self.bid + self.ask) / 2
        if mid <= 0:
            return None
        return (self.ask - self.bid) / mid * 10_000

    @property
    def size_imbalance(self) -> float | None:
        if not (self.bid_size and self.ask_size):
            return None
        tot = (self.bid_size or 0) + (self.ask_size or 0)
        if tot <= 0:
            return None
        return ((self.bid_size or 0) - (self.ask_size or 0)) / tot

    def age_seconds(self, now: datetime) -> float:
        return max(0.0, (now - self.ts).total_seconds())


# ----------------------------------------------------------------------- news
@dataclass(slots=True)
class RawNewsItem:
    ticker: str | None
    title: str
    snippet: str
    url: str
    source: str
    source_type: str
    published_at: datetime
    first_seen_at: datetime
    relevance: float = 0.5
    raw: dict[str, Any] = field(default_factory=dict)
    cluster_id: str | None = None
    cluster_size: int = 1
    item_id: str | None = None

    @property
    def available_at(self) -> datetime:
        return max(self.published_at, self.first_seen_at)

    @property
    def text(self) -> str:
        t = (self.title or "").strip()
        s = (self.snippet or "").strip()[:200]
        return f"{t} — {s}" if s else t


@dataclass(slots=True)
class ScoredNews:
    item: RawNewsItem
    p_pos: float
    p_neg: float
    p_neu: float

    @property
    def label(self) -> str:
        m = max(self.p_pos, self.p_neg, self.p_neu)
        if m == self.p_pos:
            return "positive"
        if m == self.p_neg:
            return "negative"
        return "neutral"

    @property
    def score(self) -> float:
        return self.p_pos - self.p_neg


# ------------------------------------------------------------------- universe
@dataclass(slots=True)
class UniverseRow:
    ticker: str
    exchange: str = ""
    name: str = ""
    description: str = ""
    price: float = 0.0
    volume: float = 0.0
    change_pct: float = 0.0
    rel_volume: float = 0.0
    market_cap: float = 0.0
    avg_volume: float = 0.0
    dollar_volume: float = 0.0
    # structural liquidity (price x 10-day average daily volume); the per-bar
    # dollar volume is meaningless early in the session
    avg_dollar_volume: float = 0.0
    atr: float = 0.0
    atr_pct: float = 0.0
    rsi: float = 0.0
    sector: str = "Unknown"
    industry: str = ""
    screen_score: float = 0.0
    rank: int = 0
    is_held: bool = False
    earnings_date: str | None = None


# --------------------------------------------------------------------- signals
@dataclass(slots=True)
class TickerSignal:
    ticker: str
    bar_ts: datetime
    price: float
    atr: float
    atr_pct: float
    p_win: float = 0.5
    # percentile of p_win inside the current run's cross-section (None = too few names)
    p_win_pct: float | None = None
    expected_r: float = 0.0
    vol_hat: float = 0.01
    p_entry: float = 0.5
    # percentile of p_entry inside the current run's cross-section
    p_entry_pct: float | None = None
    disagreement: float = 0.0
    ml_score: float = 50.0
    p_up_trees: float = 0.5
    p_down_trees: float = 0.5
    p_up_lin: float = 0.5
    exp_ret: float = 0.0
    sent_ewm_240: float = 0.0
    sent_mean_60: float = 0.0
    sent_max_neg_240: float = 0.0
    news_count_240: int = 0
    news_neg_240: int = 0
    news_pending_frac: float = 0.0
    rel_volume: float = 1.0
    dollar_volume: float = 0.0
    # structural liquidity (price x 10-day average daily volume); the per-bar
    # dollar volume is meaningless early in the session
    avg_dollar_volume: float = 0.0
    spread_bps: float | None = None
    quote_age_s: float | None = None
    sector: str = "Unknown"
    earnings_flag: bool = False
    halted: bool = False
    bootstrap: bool = False
    shap_top3: list[tuple[str, float]] = field(default_factory=list)

    @property
    def exp_r_per_vol(self) -> float:
        return self.expected_r / max(self.vol_hat, 1e-6)

    @property
    def confidence(self) -> float:
        return max(0.0, 1.0 - self.disagreement)


# ------------------------------------------------------------------ portfolio
@dataclass(slots=True)
class PositionState:
    ticker: str
    qty: float
    avg_cost: float
    last: float | None = None
    sl: float | None = None
    tp: float | None = None
    entry_ts: datetime | None = None
    entry_run_id: int | None = None
    bars_held: int = 0
    trail_active: bool = False
    highest_close: float | None = None
    ib_stop_id: int | None = None
    ib_tp_id: int | None = None
    status: str = "open"
    external: bool = False
    sector: str = "Unknown"
    entry_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def market_value(self) -> float:
        return (self.last or self.avg_cost) * self.qty

    @property
    def unrealized_pnl(self) -> float:
        if self.last is None:
            return 0.0
        return (self.last - self.avg_cost) * self.qty

    @property
    def unrealized_pct(self) -> float:
        """Signed return on the position, direction-aware (shorts invert)."""
        if not self.avg_cost:
            return 0.0
        last = self.last or self.avg_cost
        pct = (last - self.avg_cost) / self.avg_cost * 100
        return pct if self.qty >= 0 else -pct

    @property
    def unrealized_r(self) -> float:
        if self.last is None or self.sl is None:
            return 0.0
        risk = self.avg_cost - self.sl
        if risk <= 0:
            return 0.0
        return (self.last - self.avg_cost) / risk

    @property
    def is_long(self) -> bool:
        return self.qty > 0


@dataclass(slots=True)
class AccountState:
    currency: str = "USD"
    net_liq: float = 0.0
    cash: float = 0.0
    available_funds: float = 0.0
    buying_power: float = 0.0
    excess_liquidity: float = 0.0
    gross_position_value: float = 0.0
    init_margin: float = 0.0
    maint_margin: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    fx_to_usd: float = 1.0

    @property
    def net_liq_usd(self) -> float:
        return self.net_liq * self.fx_to_usd

    @property
    def cash_usd(self) -> float:
        return self.cash * self.fx_to_usd

    @property
    def available_usd(self) -> float:
        return self.available_funds * self.fx_to_usd

    @property
    def buying_power_usd(self) -> float:
        return self.buying_power * self.fx_to_usd

    @property
    def excess_liq_pct(self) -> float:
        if self.net_liq <= 0:
            return 0.0
        return self.excess_liquidity / self.net_liq * 100

    @property
    def daily_pnl_pct(self) -> float:
        base = self.net_liq - self.daily_pnl
        if base <= 0:
            return 0.0
        return self.daily_pnl / base * 100


# ------------------------------------------------------------------ execution
@dataclass(slots=True)
class OrderIntent:
    run_id: int | None
    ticker: str
    side: Side
    qty: int
    order_type: str = "MKT"
    limit_price: float | None = None
    reason: str = ""
    order_ref: str = ""
    source: str = "auto"
    planned_sl: float | None = None
    planned_tp: float | None = None
    status: str = "pending"
    id: int | None = None
    con_id: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Decision:
    ticker: str
    action: Action
    reason_codes: list[str] = field(default_factory=list)
    rank: int | None = None
    rank_score: float = 0.0
    planned_qty: int = 0
    planned_sl: float | None = None
    planned_tp: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DecisionBundle:
    decisions: list[Decision] = field(default_factory=list)
    intents: list[OrderIntent] = field(default_factory=list)
    funnel: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BreakerState:
    tripped: bool = False
    kind: str = ""
    detail: str = ""
    since: datetime | None = None
    entries_blocked: bool = False


@dataclass(slots=True)
class StageResult:
    name: str
    status: str = "ok"
    started_at: datetime | None = None
    duration_ms: int = 0
    detail: str = ""
    counters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
