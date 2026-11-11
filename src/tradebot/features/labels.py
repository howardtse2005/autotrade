"""Triple-barrier labels with delay-aware fills (point-in-time correct)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.core.clock import ET


def _session_bounds(index: pd.DatetimeIndex) -> np.ndarray:
    """Session id per bar (so horizons are truncated at the close)."""
    dates = index.tz_convert(ET).date
    return np.array([d.toordinal() for d in dates])


def triple_barrier(
    bars: pd.DataFrame,
    atr: pd.Series,
    horizon_bars: int,
    tp_atr: float,
    sl_atr: float,
    lag_bars: int = 1,
    entry_tp_atr: float = 1.0,
    entry_sl_atr: float = 0.6,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Return label columns indexed like `bars`.

    y_tb   : +1 take-profit first, -1 stop first, 0 vertical barrier
    y_ret  : forward log return at the vertical barrier, net of costs
    y_rv   : log realised vol over the horizon
    y_entry: +1 if a "good entry" margin was achieved (SVM target)
    """
    n = len(bars)
    out = pd.DataFrame(index=bars.index, columns=["y_tb", "y_ret", "y_rv", "y_entry"], dtype=float)
    if n < horizon_bars + lag_bars + 2:
        return out

    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    lo = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    a = atr.to_numpy(float)
    sess = _session_bounds(bars.index)
    cost = cost_bps / 10_000.0

    logret = np.zeros(n)
    logret[1:] = np.log(np.maximum(c[1:], 1e-9) / np.maximum(c[:-1], 1e-9))

    y_tb = np.full(n, np.nan)
    y_ret = np.full(n, np.nan)
    y_rv = np.full(n, np.nan)
    y_entry = np.full(n, np.nan)

    for i in range(n):
        atr_i = a[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue
        entry_idx = i + 1 + lag_bars
        if entry_idx >= n:
            continue
        entry = o[entry_idx]
        if not np.isfinite(entry) or entry <= 0:
            continue
        tp = entry + tp_atr * atr_i
        sl = entry - sl_atr * atr_i
        end = min(entry_idx + horizon_bars, n - 1)
        # truncate at the session close
        same = np.nonzero(sess[entry_idx : end + 1] != sess[entry_idx])[0]
        if len(same) > 0:
            end = entry_idx + int(same[0]) - 1
        if end < entry_idx:
            continue

        label = 0
        for j in range(entry_idx, end + 1):
            hit_sl = lo[j] <= sl
            hit_tp = h[j] >= tp
            if hit_sl and hit_tp:
                label = -1  # worst case
                break
            if hit_sl:
                label = -1
                break
            if hit_tp:
                label = 1
                break

        exit_px = c[end]
        y_tb[i] = label
        y_ret[i] = float(np.log(max(exit_px, 1e-9) / entry) - cost)
        rv = logret[entry_idx : end + 1]
        y_rv[i] = float(np.log(max(np.sqrt(np.sum(rv**2)), 1e-8)))

        # good-entry margin: +1 ATR before -0.6 ATR, and MAE bounded
        e_tp = entry + entry_tp_atr * atr_i
        e_sl = entry - entry_sl_atr * atr_i
        good = 0.0
        for j in range(entry_idx, end + 1):
            if lo[j] <= e_sl:
                good = 0.0
                break
            if h[j] >= e_tp:
                good = 1.0
                break
        y_entry[i] = good

    out["y_tb"] = y_tb
    out["y_ret"] = y_ret
    out["y_rv"] = y_rv
    out["y_entry"] = y_entry
    return out


def uniqueness_weights(n: int, horizon: int, lag: int = 1) -> np.ndarray:
    """López de Prado style label-uniqueness weights (approximate, vectorised)."""
    if n <= 0:
        return np.zeros(0)
    start = np.minimum(np.arange(n) + 1 + lag, n - 1)
    end = np.minimum(start + horizon, n - 1)
    diff = np.zeros(n + 2)
    np.add.at(diff, start, 1.0)
    np.add.at(diff, end + 1, -1.0)
    concurrency = np.cumsum(diff)[:n]
    concurrency = np.maximum(concurrency, 1.0)
    w = 1.0 / concurrency
    return w / w.mean() if w.mean() > 0 else w


def realized_vol(logret: pd.Series, window: int) -> pd.Series:
    return logret.rolling(window).std()
