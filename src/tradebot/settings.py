"""Typed configuration: YAML defaults + environment secrets + DB overrides."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "default.yaml"
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

Mode = Literal["off", "dry_run", "paper", "live"]


# --------------------------------------------------------------------------- env
class Env(BaseSettings):
    """Secrets and machine-level settings. Never persisted to the DB or logs."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    IB_HOST: str = "127.0.0.1"
    IB_PORT: int = 4002
    IB_CLIENT_ID: int = 17
    IB_ACCOUNT: str = ""
    IB_MARKET_DATA_TYPE: int = 3
    TRADING_MODE: str = "paper"
    ALLOW_LIVE: int = 0

    FINNHUB_API_KEY: str = ""
    MARKETAUX_API_KEY: str = ""
    SEC_USER_AGENT: str = ""

    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = 8000
    APP_API_TOKEN: str = ""

    def has(self, key: str) -> bool:
        return bool(getattr(self, key, ""))


# ---------------------------------------------------------------------- sections
class PipelineCfg(BaseModel):
    interval_minutes: int = Field(15, ge=3, le=240, description="Pipeline interval in minutes")
    market_hours_only: bool = True
    skip_open_minutes: int = Field(10, ge=0, le=120, description="No entries in first N minutes")
    no_new_entries_last_minutes: int = Field(60, ge=0, le=240)
    flatten_eod: bool = True
    run_timeout_seconds: int = Field(600, ge=60, le=3600)
    position_monitor_seconds: int = Field(60, ge=15, le=600)
    research_when_closed: bool = True
    queue_entries_outside_rth: bool = Field(
        False, description="Allow entries outside RTH: the order waits at IBKR and fills at the open"
    )


class UniverseCfg(BaseModel):
    size: int = Field(50, ge=10, le=90)
    # The run evaluates `size` names first, then keeps pulling further tranches
    # from the screened pool until the book is full or the pool is exhausted.
    names_per_slot: int = Field(
        20, ge=5, le=60,
        description="Names evaluated per open slot in the first pass (5 slots -> 100 names)",
    )
    max_candidates: int = Field(200, ge=10, le=500)
    expand_step: int = Field(50, ge=10, le=200)
    max_expand_rounds: int = Field(2, ge=0, le=5)
    candidate_pool: int = Field(250, ge=50, le=500)
    exchanges: list[str] = Field(default_factory=lambda: ["NASDAQ", "NYSE"])
    min_price: float = Field(5.0, ge=0)
    max_price: float = Field(1000.0, ge=1)
    min_market_cap: float = Field(2.0e9, ge=0)
    min_avg_dollar_volume: float = Field(25.0e6, ge=0)
    min_relative_volume: float = Field(1.0, ge=0)
    always_include_positions: bool = True
    benchmarks: list[str] = Field(default_factory=lambda: ["SPY", "QQQ"])
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "rel_volume": 0.35,
            "abs_change": 0.25,
            "dollar_volume": 0.25,
            "atr_pct": 0.15,
        }
    )
    static_watchlist: list[str] = Field(default_factory=list)


class DataCfg(BaseModel):
    bar_minutes: int = Field(15, ge=1, le=60)
    data_delay_minutes: int = Field(15, ge=0, le=60)
    bar_provider_order: list[str] = Field(default_factory=lambda: ["yfinance", "ibkr"])
    intraday_history_days: int = Field(55, ge=5, le=59)
    daily_lookback_days: int = Field(300, ge=60, le=1000)
    l2_enabled: bool = False
    quote_provider_order: list[str] = Field(default_factory=lambda: ["ibkr", "yfinance"])
    bars_refresh_minutes: int = Field(15, ge=1, le=240)

    @property
    def execution_lag_bars(self) -> int:
        return max(1, -(-self.data_delay_minutes // self.bar_minutes))


class NewsConnectorCfg(BaseModel):
    enabled: bool | Literal["auto"] = True
    weight: float = Field(0.8, ge=0, le=2)
    rpm: int = Field(30, ge=1, le=600)


class NewsCfg(BaseModel):
    lookback_hours: int = Field(48, ge=1, le=240)
    max_items_per_ticker: int = Field(30, ge=1, le=200)
    connectors: dict[str, Any] = Field(
        default_factory=lambda: {
            "google_news_rss": {"enabled": True, "weight": 0.8, "rpm": 30},
            "yahoo_finance": {"enabled": True, "weight": 0.8, "rpm": 30},
            "finnhub": {"enabled": "auto", "weight": 1.0, "rpm": 50},
            "marketaux": {"enabled": "auto", "weight": 0.8, "rpm": 5},
            "generic_rss": {"enabled": True, "weight": 0.7, "rpm": 20},
        }
    )
    generic_feeds: dict[str, str] = Field(
        default_factory=lambda: {
            "cnbc": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "marketwatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
            "seekingalpha": "https://seekingalpha.com/market_currents.xml",
        }
    )
    blocked_domains: list[str] = Field(default_factory=list)
    dedupe_similarity: int = Field(90, ge=70, le=100)


class SentimentCfg(BaseModel):
    model: str = "ProsusAI/finbert"
    max_tokens: int = Field(128, ge=32, le=512)
    batch_size: int = Field(16, ge=1, le=128)
    threads: int = Field(0, ge=0, le=64, description="0 = auto (physical cores - 1)")
    quantize_int8: bool = False
    half_life_minutes: float = Field(120, ge=10)
    windows_minutes: list[int] = Field(default_factory=lambda: [60, 240, 1440, 4320])
    score_budget_seconds: int = Field(90, ge=10, le=600)
    assumed_backfill_lag_minutes: int = Field(5, ge=0, le=120)
    enabled: bool = True


class MLCfg(BaseModel):
    horizon_bars: int = Field(8, ge=2, le=40)
    label_tp_atr: float = Field(1.5, ge=0.2, le=10)
    label_sl_atr: float = Field(1.0, ge=0.2, le=10)
    min_train_rows: int = Field(8000, ge=500)
    max_train_rows: int = Field(15000, ge=2000, le=500000)
    train_window_days: int = Field(45, ge=10, le=400)
    cv_splits: int = Field(3, ge=2, le=10)
    purge_bars: int = Field(8, ge=0, le=50)
    embargo_bars: int = Field(4, ge=0, le=50)
    feature_groups: dict[str, bool] = Field(
        default_factory=lambda: {
            "ta": True,
            "cross_section": True,
            "sentiment": True,
            "microstructure": False,
            "regime": True,
        }
    )
    svm_top_k_features: int = Field(24, ge=4, le=60)
    svm_max_train_rows: int = Field(5000, ge=1000)
    retrain_weekday: int = Field(6, ge=0, le=6, description="0=Mon .. 6=Sun")
    promote_if_oos_ic_ge: float = 0.0
    max_auc_drop: float = 0.01


class StrategyCfg(BaseModel):
    max_positions: int = Field(5, ge=1, le=5)
    auto_thresholds: bool = True
    quality_floor_pct: float = Field(
        0.60, ge=0.0, le=0.99,
        description="Lowest p_win percentile accepted when filling the remaining slots",
    )
    min_cross_section_names: int = Field(
        10, ge=3, le=200,
        description="Below this many names the p_win percentile is meaningless -> use absolute thresholds",
    )
    edge_gate_mode: Literal["strict", "warn", "off"] = "warn"
    buy_threshold_p_win: float = Field(0.58, ge=0.01, le=0.95)
    exit_threshold_p_win: float = Field(0.45, ge=0.01, le=0.7)
    rank_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "p_win": 0.50,
            "exp_r_per_vol": 0.20,
            "sentiment": 0.20,
            "rel_volume": 0.10,
        }
    )
    max_per_sector: int = Field(2, ge=1, le=10)
    max_spread_bps: float = Field(25, ge=1, le=500)
    sentiment_gate_enabled: bool = True
    min_ewm_240m: float = Field(-0.10, ge=-1, le=1)
    block_if_neg_shock: bool = True
    neg_shock_threshold: float = Field(-0.6, ge=-1, le=0)
    earnings_blackout_days: int = Field(1, ge=0, le=10)
    rotation_enabled: bool = True
    min_score_gap_z: float = Field(0.5, ge=0, le=5)
    min_hold_bars: int = Field(4, ge=0, le=100)
    max_model_disagreement: float = Field(0.25, ge=0, le=1)
    entry_timing_gate: bool = True
    min_p_entry: float = Field(0.45, ge=0, le=1)


class RiskCfg(BaseModel):
    risk_per_trade_pct: float = Field(0.75, ge=0.05, le=5)
    max_position_pct: float = Field(20, ge=1, le=100)
    cash_buffer_pct: float = Field(5, ge=0, le=50)
    sl_atr: float = Field(1.0, ge=0.2, le=10)
    tp_atr: float = Field(1.5, ge=0.2, le=20)
    min_sl_pct: float = Field(0.6, ge=0.05, le=20)
    max_sl_pct: float = Field(3.0, ge=0.1, le=50)
    trail_trigger_r: float = Field(1.0, ge=0.1, le=10)
    trail_atr_mult: float = Field(1.0, ge=0.1, le=10)
    time_stop_bars: int = Field(8, ge=1, le=100)
    sentiment_size_mult: list[float] = Field(default_factory=lambda: [0.7, 1.3])
    sl_tighten_max: float = Field(0.25, ge=0, le=0.9)
    tp_widen_max: float = Field(0.25, ge=0, le=0.9)
    min_tp_sl_ratio: float = Field(
        1.2, ge=1.0, le=3.0,
        description="Minimum reward:risk; compute_stops widens the target to honour it",
    )
    kelly_fraction_cap: float = Field(0.25, ge=0, le=1)
    daily_loss_limit_pct: float = Field(2.0, ge=0.1, le=50)
    max_drawdown_halt_pct: float = Field(8.0, ge=0.5, le=90)
    max_trades_per_day: int = Field(30, ge=1, le=200)
    cooldown_after_stop_minutes: int = Field(120, ge=0, le=1440)
    intraday_margin_guard: bool = True
    min_excess_liquidity_pct: float = Field(15, ge=0, le=100)
    min_notional: float = Field(500, ge=0)
    participation_cap_pct: float = Field(0.5, ge=0.01, le=5)
    max_order_notional: float = Field(25000, ge=100)


class ExecutionCfg(BaseModel):
    entry_order_type: Literal["MKT", "LMT_MARKETABLE"] = "MKT"
    protective_orders: Literal["oca_stop_tp", "none"] = "oca_stop_tp"
    slippage_buffer_bps: int = Field(25, ge=0, le=200)
    queued_reprice_band_pct: float = Field(
        0.5, ge=0.05, le=10.0,
        description="Re-price a working queued entry once the decision price moves this far",
    )
    approval_max_drift_pct: float = Field(
        1.5, ge=0.1, le=20.0,
        description="Reject an approval whose limit has drifted this far from the live price",
    )
    queued_limit_buffer_bps: int = Field(
        50, ge=0, le=500,
        description="Limit buffer for orders queued outside RTH (they must fill at the open)",
    )
    outside_rth: bool = False
    fill_timeout_seconds: int = Field(30, ge=5, le=300)
    protect_retry_attempts: int = Field(2, ge=1, le=5)


class CostsCfg(BaseModel):
    commission_per_share: float = Field(0.005, ge=0)
    commission_min: float = Field(1.00, ge=0)
    commission_max_pct: float = Field(1.0, ge=0)
    slippage_bps: float = Field(5, ge=0, le=100)
    half_spread_from_data: bool = True


class BacktestCfg(BaseModel):
    initial_cash: float = Field(100000, ge=1000)
    intrabar_conflict: Literal["stop_first", "tp_first"] = "stop_first"
    days: int = Field(20, ge=1, le=120)


class AppConfig(BaseModel):
    """Runtime configuration (no secrets)."""

    mode: Mode = "dry_run"
    approval_mode: Literal["auto", "manual_approve"] = "auto"
    approval_ttl_minutes: int = Field(
        30, ge=1, le=1440,
        description="Pending approvals older than this expire instead of executing",
    )
    base_currency: str = "HKD"
    pipeline: PipelineCfg = Field(default_factory=PipelineCfg)
    universe: UniverseCfg = Field(default_factory=UniverseCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    news: NewsCfg = Field(default_factory=NewsCfg)
    sentiment: SentimentCfg = Field(default_factory=SentimentCfg)
    ml: MLCfg = Field(default_factory=MLCfg)
    strategy: StrategyCfg = Field(default_factory=StrategyCfg)
    risk: RiskCfg = Field(default_factory=RiskCfg)
    execution: ExecutionCfg = Field(default_factory=ExecutionCfg)
    costs: CostsCfg = Field(default_factory=CostsCfg)
    backtest: BacktestCfg = Field(default_factory=BacktestCfg)

    @model_validator(mode="after")
    def _couple(self) -> AppConfig:
        # Labels and live risk are coupled by design; divergence is surfaced as a
        # warning instead of being silently rewritten, so the UI stays honest.
        return self

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        return v.lower()

    def hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def coupled_warnings(self) -> list[str]:
        out = []
        if self.risk.tp_atr / max(self.risk.sl_atr, 1e-9) < 1.2:
            out.append("TP/SL ratio below 1.2 — trades will be skipped")
        if self.risk.time_stop_bars != self.ml.horizon_bars:
            out.append("time_stop_bars differs from ml.horizon_bars — retrain recommended")
        if abs(self.risk.tp_atr - self.ml.label_tp_atr) > 1e-9:
            out.append("risk.tp_atr differs from label_tp_atr — retrain recommended")
        if abs(self.risk.sl_atr - self.ml.label_sl_atr) > 1e-9:
            out.append("risk.sl_atr differs from label_sl_atr — retrain recommended")
        return out


# ------------------------------------------------------------------- loading
def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(overrides: dict | None = None, path: Path = CONFIG_PATH) -> AppConfig:
    data = load_yaml(path)
    if overrides:
        data = _deep_merge(data, overrides)
    cfg = AppConfig(**data)
    # `.env` is authoritative for the trading mode (it is the safety switch).
    env = get_env()
    m = (env.TRADING_MODE or "").strip().lower()
    if m in ("dry_run", "paper", "live", "off"):
        cfg.mode = m  # type: ignore[assignment]
    return cfg


_env_cache: Env | None = None


def get_env() -> Env:
    global _env_cache
    if _env_cache is None:
        _env_cache = Env()
    return _env_cache


def reset_env_cache() -> None:
    global _env_cache
    _env_cache = None


def ensure_dirs() -> None:
    for p in (DATA_DIR, MODELS_DIR, DATA_DIR / "bars", DATA_DIR / "logs", DATA_DIR / "backtests"):
        p.mkdir(parents=True, exist_ok=True)


def flatten_schema(model: type[BaseModel], prefix: str = "") -> list[dict[str, Any]]:
    """Flatten a pydantic model into editable leaf fields for the settings UI."""
    out: list[dict[str, Any]] = []
    schema = model.model_json_schema(ref_template="#/$defs/{model}")
    defs = schema.get("$defs", {})

    def resolve(node: dict) -> dict:
        if "$ref" in node:
            name = node["$ref"].split("/")[-1]
            return defs.get(name, {})
        return node

    props = schema.get("properties", {})
    for name, raw in props.items():
        node = resolve(raw)
        if node.get("type") == "object" and "properties" in node:
            out.extend(flatten_schema_node(node, f"{prefix}{name}."))
            continue
        out.append(_leaf(name, node, prefix))
    return out


def flatten_schema_node(node: dict, prefix: str) -> list[dict[str, Any]]:
    out = []
    for name, raw in node.get("properties", {}).items():
        sub = raw
        if sub.get("type") == "object" and "properties" in sub:
            out.extend(flatten_schema_node(sub, f"{prefix}{name}."))
        else:
            out.append(_leaf(name, sub, prefix))
    return out


def _leaf(name: str, node: dict, prefix: str) -> dict[str, Any]:
    typ = node.get("type")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), "string")
    return {
        "key": f"{prefix}{name}",
        "name": name,
        "type": typ or "string",
        "title": node.get("title", name),
        "description": node.get("description", ""),
        "default": node.get("default"),
        "minimum": node.get("minimum"),
        "maximum": node.get("maximum"),
        "enum": node.get("enum"),
    }


def get_path(cfg: AppConfig | dict, dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if isinstance(cur, BaseModel):
            cur = getattr(cur, part)
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def set_path(data: dict, dotted: str, value: Any) -> dict:
    parts = dotted.split(".")
    cur = data
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value
    return data


def deep_update(base: dict, patch: dict) -> dict:
    return _deep_merge(base, patch)


def env_os() -> dict[str, str]:
    return dict(os.environ)
