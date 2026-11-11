"""Parquet-backed OHLCV store with incremental, validated appends."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from tradebot.core.clock import to_utc, now_utc
from tradebot.settings import DATA_DIR, ensure_dirs

BARS_DIR = Path(DATA_DIR) / "bars"
COLUMNS = ["open", "high", "low", "close", "volume"]


def _path(provider: str, bar_minutes: int, ticker: str, kind: str = "intraday") -> Path:
    safe = ticker.replace("/", "_").replace(".", "_")
    p = BARS_DIR / provider / f"{bar_minutes}m" / kind
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{safe}.parquet"


def to_session_close(df: pd.DataFrame) -> pd.DataFrame:
    """Re-stamp daily bars at the session close (UTC) so they are point-in-time correct.

    A daily bar is only knowable after the close, and its calendar date must be the
    ET session date (not a UTC-midnight artefact).
    """
    if df.empty:
        return df
    from tradebot.core.clock import ET

    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        dates = [ts.date() for ts in idx]
    else:
        # A UTC-midnight stamp from a date-only source means "that session date".
        dates = [
            ts.date() if (ts.hour == 0 and ts.minute == 0) else ts.tz_convert(ET).date()
            for ts in idx
        ]
    stamps = [pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=21) for d in dates]
    out.index = pd.DatetimeIndex(stamps, name="bar_ts")
    return out[~out.index.duplicated(keep="last")].sort_index()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to UTC index + canonical columns, sorted, de-duplicated."""
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("bar frame needs a DatetimeIndex")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index.name = "bar_ts"
    out.columns = [str(c).lower().replace("adj close", "adj_close") for c in out.columns]
    for c in COLUMNS:
        if c not in out.columns:
            out[c] = 0.0 if c == "volume" else float("nan")
    out = out[COLUMNS + [c for c in out.columns if c not in COLUMNS]]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["open", "high", "low", "close"])
    # OHLC sanity
    bad = (out["high"] < out["low"]) | (out["close"] <= 0)
    if bad.any():
        out = out[~bad]
    return out


def read(provider: str, bar_minutes: int, ticker: str, kind: str = "intraday") -> pd.DataFrame:
    p = _path(provider, bar_minutes, ticker, kind)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_parquet(p)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "bar_ts"
    return df


def write(provider: str, bar_minutes: int, ticker: str, df: pd.DataFrame, kind: str = "intraday") -> None:
    ensure_dirs()
    p = _path(provider, bar_minutes, ticker, kind)
    out = normalize(df)
    if out.empty:
        return
    tmp = p.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, compression="snappy")
    tmp.replace(p)


def append(provider: str, bar_minutes: int, ticker: str, df: pd.DataFrame, kind: str = "intraday") -> pd.DataFrame:
    """Merge new bars into the stored history and return the full frame."""
    new = normalize(df)
    if new.empty:
        return read(provider, bar_minutes, ticker, kind)
    old = read(provider, bar_minutes, ticker, kind)
    if old.empty:
        merged = new
    else:
        merged = normalize(pd.concat([old, new]))
    # keep 400 days max for intraday, 5y for daily
    horizon = timedelta(days=400 if kind == "intraday" else 1900)
    merged = merged[merged.index >= merged.index.max() - horizon]
    write(provider, bar_minutes, ticker, merged, kind)
    return merged


def last_ts(provider: str, bar_minutes: int, ticker: str, kind: str = "intraday") -> datetime | None:
    df = read(provider, bar_minutes, ticker, kind)
    if df.empty:
        return None
    return to_utc(df.index[-1].to_pydatetime())


def coverage(provider: str, bar_minutes: int, tickers: list[str], kind: str = "intraday") -> dict[str, int]:
    return {t: len(read(provider, bar_minutes, t, kind)) for t in tickers}


def drop_incomplete(df: pd.DataFrame, bar_minutes: int, now: datetime | None = None) -> pd.DataFrame:
    """Remove the still-forming last bar."""
    if df.empty:
        return df
    now = to_utc(now or now_utc())
    cutoff = now - timedelta(minutes=bar_minutes)
    return df[df.index <= cutoff]


def quality_report(df: pd.DataFrame, bar_minutes: int, expected_per_day: int | None = None) -> dict:
    if df.empty:
        return {"bars": 0, "ok": False, "gaps": 0, "stale_minutes": None}
    now = now_utc()
    gaps = 0
    if len(df) > 2:
        deltas = df.index.to_series().diff().dropna()
        gaps = int((deltas > pd.Timedelta(minutes=bar_minutes * 1.5)).sum())
    return {
        "bars": len(df),
        "ok": len(df) > 20,
        "gaps": gaps,
        "stale_minutes": round((now - df.index[-1].to_pydatetime()).total_seconds() / 60, 1),
        "last_bar": df.index[-1].isoformat(),
    }
