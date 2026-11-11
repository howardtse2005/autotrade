"""SQLAlchemy models. Timestamps are stored as UTC via UtcDateTime."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from tradebot.core.clock import now_utc


class UtcDateTime(TypeDecorator):
    """Always hands back tz-aware UTC datetimes, regardless of backend."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect):  # noqa: ANN001
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect):  # noqa: ANN001
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc)


class Base(DeclarativeBase):
    pass


def _ts() -> Mapped[datetime]:
    return mapped_column(UtcDateTime, default=now_utc, nullable=False)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = _ts()
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    status: Mapped[str] = mapped_column(String(24), default="running", index=True)
    trigger: Mapped[str] = mapped_column(String(32), default="scheduler")
    mode: Mapped[str] = mapped_column(String(16), default="dry_run")
    interval_min: Mapped[int] = mapped_column(Integer, default=15)
    config_hash: Mapped[str] = mapped_column(String(32), default="")
    model_id: Mapped[str] = mapped_column(String(64), default="")
    stage_timings: Mapped[dict | None] = mapped_column(JSON)
    funnel: Mapped[dict | None] = mapped_column(JSON)
    summary: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)


class RunEvent(Base):
    __tablename__ = "run_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    ts: Mapped[datetime] = _ts()
    stage: Mapped[str] = mapped_column(String(32), default="")
    level: Mapped[str] = mapped_column(String(16), default="info")
    message: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict | None] = mapped_column(JSON)


class UniverseSnapshot(Base):
    __tablename__ = "universe_snapshots"
    __table_args__ = (Index("ix_universe_run_ticker", "run_id", "ticker"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    exchange: Mapped[str] = mapped_column(String(16), default="")
    price: Mapped[float] = mapped_column(Float, default=0.0)
    volume: Mapped[float] = mapped_column(Float, default=0.0)
    change_pct: Mapped[float] = mapped_column(Float, default=0.0)
    rel_volume: Mapped[float] = mapped_column(Float, default=0.0)
    market_cap: Mapped[float] = mapped_column(Float, default=0.0)
    dollar_volume: Mapped[float] = mapped_column(Float, default=0.0)
    atr_pct: Mapped[float] = mapped_column(Float, default=0.0)
    sector: Mapped[str] = mapped_column(String(64), default="")
    industry: Mapped[str] = mapped_column(String(96), default="")
    screen_score: Mapped[float] = mapped_column(Float, default=0.0)
    rank: Mapped[int] = mapped_column(Integer, default=0)
    is_held: Mapped[bool] = mapped_column(Boolean, default=False)
    stale: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str] = mapped_column(String(32), default="tradingview")


class NewsItem(Base):
    __tablename__ = "news_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_hash: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    run_id: Mapped[int | None] = mapped_column(Integer, index=True)
    ticker: Mapped[str | None] = mapped_column(String(16), index=True)
    title: Mapped[str] = mapped_column(Text, default="")
    snippet: Mapped[str] = mapped_column(Text, default="")
    url: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(64), default="")
    source_type: Mapped[str] = mapped_column(String(32), default="", index=True)
    published_at: Mapped[datetime] = mapped_column(UtcDateTime, index=True)
    first_seen_at: Mapped[datetime] = _ts()
    relevance: Mapped[float] = mapped_column(Float, default=0.5)
    cluster_id: Mapped[str] = mapped_column(String(40), default="", index=True)
    cluster_size: Mapped[int] = mapped_column(Integer, default=1)
    p_pos: Mapped[float | None] = mapped_column(Float)
    p_neg: Mapped[float | None] = mapped_column(Float)
    p_neu: Mapped[float | None] = mapped_column(Float)
    label: Mapped[str | None] = mapped_column(String(16), index=True)
    raw: Mapped[dict | None] = mapped_column(JSON)


class SentimentScore(Base):
    __tablename__ = "sentiment_scores"

    text_hash: Mapped[str] = mapped_column(String(40), primary_key=True)
    model_version: Mapped[str] = mapped_column(String(64), primary_key=True)
    p_pos: Mapped[float] = mapped_column(Float)
    p_neg: Mapped[float] = mapped_column(Float)
    p_neu: Mapped[float] = mapped_column(Float)
    label: Mapped[str] = mapped_column(String(16))
    scored_at: Mapped[datetime] = _ts()


class QuoteRow(Base):
    __tablename__ = "quotes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(Integer, index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    ts: Mapped[datetime] = _ts()
    bid: Mapped[float | None] = mapped_column(Float)
    ask: Mapped[float | None] = mapped_column(Float)
    last: Mapped[float | None] = mapped_column(Float)
    bid_size: Mapped[float | None] = mapped_column(Float)
    ask_size: Mapped[float | None] = mapped_column(Float)
    spread_bps: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), default="")
    market_data_type: Mapped[int] = mapped_column(Integer, default=0)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (Index("ix_pred_run_ticker", "run_id", "ticker"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(Integer, index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    bar_ts: Mapped[datetime] = mapped_column(UtcDateTime)
    model_id: Mapped[str] = mapped_column(String(64), default="")
    p_win: Mapped[float] = mapped_column(Float, default=0.5)
    expected_r: Mapped[float] = mapped_column(Float, default=0.0)
    vol_hat: Mapped[float] = mapped_column(Float, default=0.0)
    p_entry: Mapped[float] = mapped_column(Float, default=0.5)
    disagreement: Mapped[float] = mapped_column(Float, default=0.0)
    ml_score: Mapped[float] = mapped_column(Float, default=50.0)
    p_up_trees: Mapped[float] = mapped_column(Float, default=0.5)
    p_down_trees: Mapped[float] = mapped_column(Float, default=0.5)
    p_up_lin: Mapped[float] = mapped_column(Float, default=0.5)
    exp_ret: Mapped[float] = mapped_column(Float, default=0.0)
    shap_top3: Mapped[list | None] = mapped_column(JSON)
    features: Mapped[dict | None] = mapped_column(JSON)


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(Integer, index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    action: Mapped[str] = mapped_column(String(8), default="SKIP")
    reason_codes: Mapped[list | None] = mapped_column(JSON)
    rank: Mapped[int | None] = mapped_column(Integer)
    rank_score: Mapped[float] = mapped_column(Float, default=0.0)
    planned_qty: Mapped[int] = mapped_column(Integer, default=0)
    planned_sl: Mapped[float | None] = mapped_column(Float)
    planned_tp: Mapped[float | None] = mapped_column(Float)
    detail: Mapped[dict | None] = mapped_column(JSON)
    ts: Mapped[datetime] = _ts()


class OrderIntentRow(Base):
    __tablename__ = "order_intents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(Integer, index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(4))
    qty: Mapped[int] = mapped_column(Integer)
    order_type: Mapped[str] = mapped_column(String(16), default="MKT")
    limit_price: Mapped[float | None] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(String(48), default="")
    order_ref: Mapped[str] = mapped_column(String(96), unique=True, index=True)
    source: Mapped[str] = mapped_column(String(16), default="auto")
    planned_sl: Mapped[float | None] = mapped_column(Float)
    planned_tp: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    meta: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = _ts()
    updated_at: Mapped[datetime | None] = mapped_column(UtcDateTime)


class OrderRow(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    intent_id: Mapped[int | None] = mapped_column(Integer, index=True)
    run_id: Mapped[int | None] = mapped_column(Integer, index=True)
    ib_order_id: Mapped[int | None] = mapped_column(Integer, index=True)
    perm_id: Mapped[int | None] = mapped_column(Integer)
    parent_id: Mapped[int | None] = mapped_column(Integer)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    action: Mapped[str] = mapped_column(String(4))
    qty: Mapped[float] = mapped_column(Float, default=0)
    order_type: Mapped[str] = mapped_column(String(16), default="MKT")
    limit_price: Mapped[float | None] = mapped_column(Float)
    aux_price: Mapped[float | None] = mapped_column(Float)
    tif: Mapped[str] = mapped_column(String(8), default="DAY")
    oca_group: Mapped[str | None] = mapped_column(String(64))
    order_ref: Mapped[str] = mapped_column(String(96), default="", index=True)
    status: Mapped[str] = mapped_column(String(24), default="PendingSubmit", index=True)
    filled: Mapped[float] = mapped_column(Float, default=0)
    avg_fill_price: Mapped[float] = mapped_column(Float, default=0)
    role: Mapped[str] = mapped_column(String(16), default="parent")
    created_at: Mapped[datetime] = _ts()
    updated_at: Mapped[datetime | None] = mapped_column(UtcDateTime)


class FillRow(Base):
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_ref: Mapped[str] = mapped_column(String(96), index=True, default="")
    exec_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(4))
    qty: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    commission: Mapped[float] = mapped_column(Float, default=0)
    ts: Mapped[datetime] = _ts()


class PositionRow(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    qty: Mapped[float] = mapped_column(Float, default=0)
    avg_cost: Mapped[float] = mapped_column(Float, default=0)
    entry_ts: Mapped[datetime | None] = mapped_column(UtcDateTime)
    entry_run_id: Mapped[int | None] = mapped_column(Integer)
    sl: Mapped[float | None] = mapped_column(Float)
    tp: Mapped[float | None] = mapped_column(Float)
    planned_r: Mapped[float | None] = mapped_column(Float)
    trail_active: Mapped[bool] = mapped_column(Boolean, default=False)
    highest_close: Mapped[float | None] = mapped_column(Float)
    bars_held: Mapped[int] = mapped_column(Integer, default=0)
    ib_stop_id: Mapped[int | None] = mapped_column(Integer)
    ib_tp_id: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)
    external: Mapped[bool] = mapped_column(Boolean, default=False)
    sector: Mapped[str] = mapped_column(String(64), default="")
    entry_snapshot: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = _ts()
    closed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    position_id: Mapped[int | None] = mapped_column(Integer, index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    qty: Mapped[float] = mapped_column(Float, default=0)
    entry_ts: Mapped[datetime | None] = mapped_column(UtcDateTime)
    entry_price: Mapped[float] = mapped_column(Float, default=0)
    exit_ts: Mapped[datetime | None] = mapped_column(UtcDateTime)
    exit_price: Mapped[float] = mapped_column(Float, default=0)
    pnl: Mapped[float] = mapped_column(Float, default=0)
    pnl_pct: Mapped[float] = mapped_column(Float, default=0)
    r_multiple: Mapped[float] = mapped_column(Float, default=0)
    exit_reason: Mapped[str] = mapped_column(String(32), default="")
    hold_bars: Mapped[int] = mapped_column(Integer, default=0)
    commission: Mapped[float] = mapped_column(Float, default=0)
    entry_snapshot: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = _ts()


class AccountSnapshot(Base):
    __tablename__ = "account_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = _ts()
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    net_liq: Mapped[float] = mapped_column(Float, default=0)
    cash: Mapped[float] = mapped_column(Float, default=0)
    available_funds: Mapped[float] = mapped_column(Float, default=0)
    buying_power: Mapped[float] = mapped_column(Float, default=0)
    excess_liquidity: Mapped[float] = mapped_column(Float, default=0)
    gross_position_value: Mapped[float] = mapped_column(Float, default=0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0)
    daily_pnl: Mapped[float] = mapped_column(Float, default=0)
    fx_to_usd: Mapped[float] = mapped_column(Float, default=1.0)


class ModelRecord(Base):
    __tablename__ = "models"

    model_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = _ts()
    train_start: Mapped[datetime | None] = mapped_column(UtcDateTime)
    train_end: Mapped[datetime | None] = mapped_column(UtcDateTime)
    rows: Mapped[int] = mapped_column(Integer, default=0)
    features: Mapped[int] = mapped_column(Integer, default=0)
    metrics: Mapped[dict | None] = mapped_column(JSON)
    config: Mapped[dict | None] = mapped_column(JSON)
    feature_schema_hash: Mapped[str] = mapped_column(String(32), default="")
    status: Mapped[str] = mapped_column(String(16), default="challenger", index=True)
    notes: Mapped[str] = mapped_column(Text, default="")


class Backtest(Base):
    __tablename__ = "backtests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = _ts()
    status: Mapped[str] = mapped_column(String(16), default="running", index=True)
    config: Mapped[dict | None] = mapped_column(JSON)
    metrics: Mapped[dict | None] = mapped_column(JSON)
    equity_curve: Mapped[list | None] = mapped_column(JSON)
    trades: Mapped[list | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)


class SettingOverride(Base):
    __tablename__ = "settings_overrides"

    key: Mapped[str] = mapped_column(String(96), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON)
    updated_at: Mapped[datetime] = _ts()


class SettingsAudit(Base):
    __tablename__ = "settings_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = _ts()
    key: Mapped[str] = mapped_column(String(96), index=True)
    old_value: Mapped[Any] = mapped_column(JSON)
    new_value: Mapped[Any] = mapped_column(JSON)
    source: Mapped[str] = mapped_column(String(32), default="ui")


class ConnectorHealth(Base):
    __tablename__ = "connector_health"

    name: Mapped[str] = mapped_column(String(48), primary_key=True)
    last_success: Mapped[datetime | None] = mapped_column(UtcDateTime)
    last_error: Mapped[str | None] = mapped_column(Text)
    last_error_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    items_last_run: Mapped[int] = mapped_column(Integer, default=0)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    breaker_open_until: Mapped[datetime | None] = mapped_column(UtcDateTime)
    total_runs: Mapped[int] = mapped_column(Integer, default=0)
    total_items: Mapped[int] = mapped_column(Integer, default=0)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = _ts()
    level: Mapped[str] = mapped_column(String(16), default="info", index=True)
    kind: Mapped[str] = mapped_column(String(48), default="")
    message: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict | None] = mapped_column(JSON)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)


class AppState(Base):
    """Small key/value store for runtime state (breakers, kill switch, mode)."""

    __tablename__ = "app_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON)
    updated_at: Mapped[datetime] = _ts()


class TickerMap(Base):
    __tablename__ = "ticker_map"
    __table_args__ = (UniqueConstraint("tv_symbol", name="uq_tv_symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    tv_symbol: Mapped[str] = mapped_column(String(32), default="")
    ib_symbol: Mapped[str] = mapped_column(String(32), default="")
    con_id: Mapped[int | None] = mapped_column(Integer)
    min_tick: Mapped[float | None] = mapped_column(Float)
    primary_exchange: Mapped[str] = mapped_column(String(24), default="")
    company_name: Mapped[str] = mapped_column(Text, default="")
    aliases: Mapped[list | None] = mapped_column(JSON)
    updated_at: Mapped[datetime] = _ts()
