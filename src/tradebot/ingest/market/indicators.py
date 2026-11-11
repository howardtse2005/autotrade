"""Per-ticker technical features (TA-Lib), all strictly causal."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import talib
except Exception:  # pragma: no cover - talib is present in the env
    talib = None


def _safe(series: pd.Series | np.ndarray) -> pd.Series:
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    return s.replace([np.inf, -np.inf], np.nan)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    if talib is not None:
        return _safe(pd.Series(talib.RSI(close.to_numpy(dtype=float), timeperiod=n), index=close.index))
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return _safe(100 - 100 / (1 + rs))


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    if talib is not None:
        return _safe(pd.Series(
            talib.ATR(df["high"].to_numpy(float), df["low"].to_numpy(float),
                      df["close"].to_numpy(float), timeperiod=n),
            index=df.index,
        ))
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _safe(tr.ewm(alpha=1 / n, adjust=False).mean())


def _adx(df: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    if talib is not None:
        h, low, c = df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
        return (
            _safe(pd.Series(talib.ADX(h, low, c, timeperiod=n), index=df.index)),
            _safe(pd.Series(talib.PLUS_DI(h, low, c, timeperiod=n), index=df.index)),
            _safe(pd.Series(talib.MINUS_DI(h, low, c, timeperiod=n), index=df.index)),
        )
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _atr(df, n)
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return _safe(dx.ewm(alpha=1 / n, adjust=False).mean()), _safe(pdi), _safe(mdi)


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    if talib is not None:
        macd, sig, hist = talib.MACD(close.to_numpy(float), 12, 26, 9)
        return _safe(pd.Series(macd, index=close.index)), _safe(pd.Series(sig, index=close.index)), _safe(
            pd.Series(hist, index=close.index)
        )
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    sig = macd.ewm(span=9, adjust=False).mean()
    return _safe(macd), _safe(sig), _safe(macd - sig)


def _stoch(df: pd.DataFrame, n: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
    if talib is not None:
        k, dd = talib.STOCH(
            df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float),
            fastk_period=n, slowk_period=d, slowk_matype=0, slowd_period=d, slowd_matype=0,
        )
        return _safe(pd.Series(k, index=df.index)), _safe(pd.Series(dd, index=df.index))
    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    k = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    return _safe(k), _safe(k.rolling(d).mean())


def _willr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    if talib is not None:
        return _safe(pd.Series(
            talib.WILLR(df["high"].to_numpy(float), df["low"].to_numpy(float),
                        df["close"].to_numpy(float), timeperiod=n),
            index=df.index,
        ))
    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    return _safe(-100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan))


def _bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    upper = mid + k * sd
    lower = mid - k * sd
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    width = (upper - lower) / mid.replace(0, np.nan)
    return _safe(pct_b), _safe(width), _safe(mid)


def _obv_slope(df: pd.DataFrame, n: int = 20) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    obv = (direction * df["volume"]).cumsum()
    return _safe(obv.diff(n) / n)


def _mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    if talib is not None:
        return _safe(pd.Series(
            talib.MFI(df["high"].to_numpy(float), df["low"].to_numpy(float),
                      df["close"].to_numpy(float), df["volume"].to_numpy(float), timeperiod=n),
            index=df.index,
        ))
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos = mf.where(tp > tp.shift(), 0.0).rolling(n).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(n).sum()
    return _safe(100 - 100 / (1 + pos / neg.replace(0, np.nan)))


def compute_features(
    bars: pd.DataFrame,
    bar_minutes: int = 15,
    daily: pd.DataFrame | None = None,
    spy: pd.DataFrame | None = None,
    vix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a feature frame indexed like `bars`. Never uses future rows."""
    if bars.empty:
        return pd.DataFrame(index=bars.index)

    df = bars.copy()
    c, h, low, v = df["close"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)

    # --- momentum
    f["rsi14"] = _rsi(c, 14)
    k, d = _stoch(df)
    f["stoch_k"], f["stoch_d"] = k, d
    macd, sig, hist = _macd(c)
    f["macd_line"], f["macd_signal"], f["macd_hist"] = macd, sig, hist
    f["macd_hist_slope"] = _safe(hist.diff())
    for n in (3, 6, 12, 26):
        f[f"roc_{n}"] = _safe(c.pct_change(n))
    f["willr14"] = _willr(df, 14)

    # --- trend
    for n in (9, 21, 50):
        ema = c.ewm(span=n, adjust=False).mean()
        f[f"ema{n}_dist"] = _safe((c - ema) / ema.replace(0, np.nan))
    adx, pdi, mdi = _adx(df, 14)
    f["adx14"], f["plus_di"], f["minus_di"] = adx, pdi, mdi
    f["di_spread"] = _safe(pdi - mdi)
    up = (h.rolling(25).apply(lambda x: float(np.argmax(x)), raw=True) / 25.0)
    dn = (low.rolling(25).apply(lambda x: float(np.argmin(x)), raw=True) / 25.0)
    f["aroon_up"], f["aroon_dn"] = _safe(up), _safe(dn)

    # --- volatility
    atr = _atr(df, 14)
    f["atr14"] = atr
    f["atr_pct"] = _safe(atr / c.replace(0, np.nan))
    pct_b, width, mid = _bollinger(c, 20, 2.0)
    f["bb_pct_b"], f["bb_width"] = pct_b, width
    logret = np.log(c / c.shift())
    f["rv_8"] = _safe(logret.rolling(8).std())
    f["rv_26"] = _safe(logret.rolling(26).std())
    f["rv_1d"] = _safe(logret.rolling(max(2, int(390 / bar_minutes))).std())
    f["rv_vol_of_vol"] = _safe(f["rv_8"].rolling(20).std())
    parkinson = np.log(h / low.replace(0, np.nan)) ** 2 / (4 * np.log(2))
    f["parkinson"] = _safe(parkinson.rolling(20).mean().pow(0.5))

    # --- volume / flow
    f["volume"] = v
    f["dollar_volume"] = _safe(c * v)
    vwap = (c * v).rolling(26).sum() / v.rolling(26).sum().replace(0, np.nan)
    f["vwap_dist"] = _safe((c - vwap) / vwap.replace(0, np.nan))
    f["obv_slope"] = _obv_slope(df, 20)
    f["mfi14"] = _mfi(df, 14)
    vmean = v.rolling(26).mean()
    vstd = v.rolling(26).std()
    f["volume_z"] = _safe((v - vmean) / vstd.replace(0, np.nan))
    # relative volume vs the same time-of-day over the prior 10 sessions
    tod = df.index.hour * 60 + df.index.minute
    tmp = pd.DataFrame({"v": v.values, "tod": tod}, index=df.index)
    per_slot = tmp.groupby("tod")["v"].apply(lambda s: s.shift().rolling(10, min_periods=3).mean())
    per_slot = per_slot.reset_index(level=0, drop=True).sort_index()
    f["rel_volume"] = _safe(v / per_slot.replace(0, np.nan))
    f["log_dollar_volume"] = _safe(np.log1p(f["dollar_volume"]))

    # --- price action
    f["bar_range"] = _safe((h - low) / c.replace(0, np.nan))
    f["range_pos"] = _safe((c - low) / (h - low).replace(0, np.nan))
    f["ret_1"] = _safe(c.pct_change())
    f["ret_4"] = _safe(c.pct_change(4))
    f["ret_8"] = _safe(c.pct_change(8))
    day = df.index.tz_convert("America/New_York").date
    day_series = pd.Series(day, index=df.index)
    # Causal intraday aggregates: only bars already printed count.
    day_open = c.groupby(day_series).transform("first")
    day_high = h.groupby(day_series).cummax()
    day_low = low.groupby(day_series).cummin()
    f["day_ret"] = _safe(c / day_open.replace(0, np.nan) - 1)
    f["dist_day_high"] = _safe(c / day_high.replace(0, np.nan) - 1)
    f["dist_day_low"] = _safe(c / day_low.replace(0, np.nan) - 1)
    day_last = c.groupby(day_series).last()
    day_last.index = pd.Index(day_last.index)
    prior_close = day_series.map(day_last.shift(1))
    f["gap_pct"] = _safe(day_open / prior_close.replace(0, np.nan) - 1)
    f["intraday_range_pos"] = _safe((c - day_low) / (day_high - day_low).replace(0, np.nan))

    # --- calendar
    idx = df.index
    mins = idx.hour * 60 + idx.minute
    f["tod_sin"] = np.sin(2 * np.pi * mins / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * mins / 1440)
    f["dow"] = idx.dayofweek.astype(float)
    f["mins_since_open"] = ((mins - (9 * 60 + 30)) % 1440).astype(float)

    # --- daily context (strictly as of the *previous* completed session)
    if daily is not None and not daily.empty:
        dl = daily.copy()
        dl.index = pd.to_datetime(dl.index)
        if dl.index.tz is None:
            dl.index = dl.index.tz_localize("UTC")
        dl = dl.sort_index()
        dc = dl["close"]
        ctx = pd.DataFrame(index=dl.index)
        ctx.index = pd.DatetimeIndex([ts.date() for ts in dl.index.tz_convert("America/New_York")])
        ctx["d_trend_20"] = (dc / dc.rolling(20).mean() - 1).to_numpy()
        ctx["d_trend_50"] = (dc / dc.rolling(50).mean() - 1).to_numpy()
        ctx["d_high_52w"] = (dc / dc.rolling(252, min_periods=60).max() - 1).to_numpy()
        ctx["d_rv_20"] = np.log(dc / dc.shift()).rolling(20).std().to_numpy()
        ctx["d_ret_5"] = dc.pct_change(5).to_numpy()
        ctx = ctx[~ctx.index.duplicated(keep="last")].sort_index()
        f = _attach_daily(f, ctx, day_series)

    # --- market regime (also previous-session only)
    if spy is not None and not spy.empty:
        sp = _daily_close_by_date(spy)
        sp_trend = (sp / sp.rolling(20).mean() - 1)
        sp_ret = sp.pct_change()
        frame = pd.DataFrame({"spy_trend": sp_trend, "spy_ret_1d": sp_ret})
        f = _attach_daily(f, frame, day_series)
        f["rel_strength_spy"] = _safe(f["day_ret"] - f["spy_ret_1d"])

    if vix is not None and not vix.empty:
        vc = _daily_close_by_date(vix)
        frame = pd.DataFrame({"vix_level": vc, "vix_chg_5d": vc.pct_change(5)})
        f = _attach_daily(f, frame, day_series)

    return f.replace([np.inf, -np.inf], np.nan)


def _daily_close_by_date(df: pd.DataFrame) -> pd.Series:
    s = df["close"].copy()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    out = pd.Series(s.to_numpy(), index=pd.DatetimeIndex([ts.date() for ts in s.index]))
    return out[~out.index.duplicated(keep="last")].sort_index()


def _attach_daily(f: pd.DataFrame, ctx: pd.DataFrame, day_series: pd.Series) -> pd.DataFrame:
    """Map per-date daily values onto intraday rows using only prior sessions."""
    if ctx.empty:
        return f
    lookup = ctx.copy()
    lookup.index = pd.DatetimeIndex(pd.to_datetime(lookup.index))
    lookup = lookup[~lookup.index.duplicated(keep="last")].sort_index()
    # date D may only see the session that completed before D
    lookup.index = lookup.index + pd.Timedelta(days=1)

    keys = pd.DatetimeIndex(pd.to_datetime(day_series.values)).values.astype("datetime64[ns]")
    daily_keys = lookup.index.values.astype("datetime64[ns]")
    pos = np.searchsorted(daily_keys, keys, side="right") - 1
    values = np.full((len(keys), lookup.shape[1]), np.nan, dtype=float)
    valid = pos >= 0
    if valid.any():
        values[valid] = lookup.to_numpy(dtype=float)[pos[valid]]
    mapped = pd.DataFrame(values, index=f.index, columns=lookup.columns)
    for col in ctx.columns:
        if col not in f.columns:
            f[col] = mapped[col]
    return f


def atr_series(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    if bars.empty:
        return pd.Series(dtype=float)
    return _atr(bars, n)
