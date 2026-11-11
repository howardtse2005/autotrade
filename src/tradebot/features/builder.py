"""Feature panel assembly: per-ticker indicators + cross-section + sentiment + labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradebot.core.clock import to_utc
from tradebot.core.logging import get_logger
from tradebot.features.labels import triple_barrier, uniqueness_weights
from tradebot.ingest.market.indicators import compute_features

log = get_logger("features")

META_COLS = ["ticker", "bar_ts", "close", "atr14", "y_tb", "y_ret", "y_rv", "y_entry",
             "sample_weight", "session_date"]
LABEL_COLS = ["y_tb", "y_ret", "y_rv", "y_entry"]

CROSS_SECTION_SOURCES = ["ret_4", "ret_8", "rel_volume", "atr_pct", "rsi14", "day_ret"]
REGIME_COLS = ["spy_trend", "spy_ret_1d", "rel_strength_spy", "vix_level", "vix_chg_5d"]
MICRO_COLS = ["spread_bps", "quote_age_s", "size_imbalance"]


@dataclass
class Panel:
    frame: pd.DataFrame
    feature_cols: list[str]
    schema_hash: str = ""

    def __post_init__(self) -> None:
        import hashlib

        if not self.schema_hash:
            self.schema_hash = hashlib.sha1("|".join(self.feature_cols).encode()).hexdigest()[:16]


def _sentiment_block(sent: pd.DataFrame | None, index: pd.DatetimeIndex) -> pd.DataFrame:
    if sent is None or sent.empty:
        return pd.DataFrame(index=index)
    s = sent.copy()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(index, method="ffill", tolerance=pd.Timedelta(hours=6))


def build_ticker_frame(
    ticker: str,
    bars: pd.DataFrame,
    cfg,
    daily: pd.DataFrame | None = None,
    spy: pd.DataFrame | None = None,
    vix: pd.DataFrame | None = None,
    sentiment: pd.DataFrame | None = None,
    with_labels: bool = True,
) -> pd.DataFrame:
    if bars is None or bars.empty or len(bars) < 30:
        return pd.DataFrame()
    f = compute_features(bars, cfg.data.bar_minutes, daily=daily, spy=spy, vix=vix)
    if f.empty:
        return f
    f = f.copy()
    f["close"] = bars["close"].reindex(f.index)
    f["ticker"] = ticker
    f["bar_ts"] = f.index
    f["session_date"] = pd.Series(f.index.tz_convert("America/New_York").date, index=f.index)

    sent = _sentiment_block(sentiment, f.index)
    for col in sent.columns:
        f[col] = sent[col].to_numpy()

    if with_labels:
        lab = triple_barrier(
            bars,
            f["atr14"],
            horizon_bars=cfg.ml.horizon_bars,
            tp_atr=cfg.ml.label_tp_atr,
            sl_atr=cfg.ml.label_sl_atr,
            lag_bars=cfg.data.execution_lag_bars,
        )
        for col in LABEL_COLS:
            f[col] = lab[col].to_numpy()
        f["sample_weight"] = uniqueness_weights(len(f), cfg.ml.horizon_bars, cfg.data.execution_lag_bars)
    return f


def build_panel(
    bars_by_ticker: dict[str, pd.DataFrame],
    cfg,
    daily_by_ticker: dict[str, pd.DataFrame] | None = None,
    spy: pd.DataFrame | None = None,
    vix: pd.DataFrame | None = None,
    sentiment_by_ticker: dict[str, pd.DataFrame] | None = None,
    with_labels: bool = True,
) -> Panel:
    frames = []
    daily_by_ticker = daily_by_ticker or {}
    sentiment_by_ticker = sentiment_by_ticker or {}
    for ticker, bars in bars_by_ticker.items():
        try:
            f = build_ticker_frame(
                ticker, bars, cfg,
                daily=daily_by_ticker.get(ticker),
                spy=spy, vix=vix,
                sentiment=sentiment_by_ticker.get(ticker),
                with_labels=with_labels,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("feature build failed", ticker=ticker, error=str(exc)[:200])
            continue
        if not f.empty:
            frames.append(f)
    if not frames:
        return Panel(pd.DataFrame(), [])

    panel = pd.concat(frames, axis=0, ignore_index=False)
    panel = panel.sort_index()

    # cross-sectional ranks per timestamp
    if cfg.ml.feature_groups.get("cross_section", True):
        for src in CROSS_SECTION_SOURCES:
            if src not in panel.columns:
                continue
            grp = panel.groupby(level=0)[src]
            panel[f"cs_rank_{src}"] = grp.rank(pct=True)
            panel[f"cs_demean_{src}"] = panel[src] - grp.transform("mean")

    # warmup: require a usable ATR and at least one daily context value
    panel = panel[panel["atr14"].notna()]
    panel = panel[panel["close"].notna()]
    if with_labels:
        panel = panel[panel["y_tb"].notna()]

    feature_cols = select_feature_columns(panel, cfg)
    return Panel(panel, feature_cols)


def add_cross_section(frame: pd.DataFrame, cfg, sources: list[str] | None = None) -> pd.DataFrame:
    """Cross-sectional ranks/demeans across the current universe snapshot.

    Used both by the training panel (per timestamp) and by live inference, so the
    model always sees the same definition.
    """
    if not cfg.ml.feature_groups.get("cross_section", True):
        return frame
    for src in (sources or CROSS_SECTION_SOURCES):
        if src not in frame.columns:
            continue
        col = pd.to_numeric(frame[src], errors="coerce")
        frame[f"cs_rank_{src}"] = col.rank(pct=True)
        frame[f"cs_demean_{src}"] = col - col.mean()
    return frame


def select_feature_columns(panel: pd.DataFrame, cfg) -> list[str]:
    groups = cfg.ml.feature_groups
    exclude = set(META_COLS) | {"open", "high", "low", "volume"}
    cols: list[str] = []
    for c in panel.columns:
        if c in exclude or c.startswith("y_"):
            continue
        if not pd.api.types.is_numeric_dtype(panel[c]):
            continue
        if c in REGIME_COLS and not groups.get("regime", True):
            continue
        if c in MICRO_COLS and not groups.get("microstructure", False):
            continue
        if (c.startswith("sent_") or c.startswith("news_")) and not groups.get("sentiment", True):
            continue
        cols.append(c)
    return sorted(set(cols))


def live_rows(
    bars_by_ticker: dict[str, pd.DataFrame],
    cfg,
    daily_by_ticker: dict[str, pd.DataFrame] | None = None,
    spy: pd.DataFrame | None = None,
    vix: pd.DataFrame | None = None,
    sentiment_by_ticker: dict[str, pd.DataFrame] | None = None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Latest complete bar per ticker, ready for prediction."""
    rows = []
    for ticker, bars in bars_by_ticker.items():
        if bars is None or len(bars) < 30:
            continue
        f = build_ticker_frame(
            ticker, bars, cfg,
            daily=(daily_by_ticker or {}).get(ticker),
            spy=spy, vix=vix,
            sentiment=(sentiment_by_ticker or {}).get(ticker),
            with_labels=False,
        )
        if f.empty:
            continue
        row = f.iloc[[-1]].copy()
        row["ticker"] = ticker
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    live = pd.concat(rows, axis=0)
    if feature_cols:
        for c in feature_cols:
            if c not in live.columns:
                live[c] = np.nan
        live = live[feature_cols + [c for c in live.columns if c not in feature_cols]]
    return live


def latest_bar_ts(bars_by_ticker: dict[str, pd.DataFrame]) -> pd.Timestamp | None:
    stamps = [to_utc(df.index[-1].to_pydatetime()) for df in bars_by_ticker.values() if not df.empty]
    if not stamps:
        return None
    return pd.Timestamp(max(stamps))
