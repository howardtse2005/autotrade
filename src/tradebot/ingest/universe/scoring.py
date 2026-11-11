"""Universe scoring: z-scores of the screener's raw columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.core.types import UniverseRow


def _z(series: pd.Series, winsor: float = 3.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 3:
        return pd.Series(0.0, index=s.index)
    med = s.median()
    s = s.fillna(med)
    mu = s.mean()
    sd = s.std(ddof=0)
    if not sd or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    z = (s - mu) / sd
    return z.clip(-winsor, winsor)


def score_frame(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Add screen_score using robust z-scores of the configured factors."""
    out = df.copy()
    price = pd.to_numeric(out.get("close", 0), errors="coerce").astype(float)
    avg_vol = pd.to_numeric(out.get("average_volume_10d_calc", 0), errors="coerce").astype(float)
    dollar_volume = (price * avg_vol).replace([np.inf, -np.inf], np.nan)

    factors = {
        "rel_volume": _z(out.get("relative_volume_10d_calc", pd.Series(dtype=float))),
        "abs_change": _z(pd.to_numeric(out.get("change", 0), errors="coerce").abs()),
        "dollar_volume": _z(np.log1p(dollar_volume.clip(lower=0))),
        "atr_pct": _z(pd.to_numeric(out.get("ATR", 0), errors="coerce") / price.replace(0, np.nan)),
    }
    total_w = sum(weights.get(k, 0.0) for k in factors) or 1.0
    score = pd.Series(0.0, index=out.index)
    for k, z in factors.items():
        score += z.fillna(0.0) * (weights.get(k, 0.0) / total_w)
    out["dollar_volume"] = dollar_volume
    out["atr_pct"] = (pd.to_numeric(out.get("ATR", 0), errors="coerce") / price.replace(0, np.nan)).fillna(0)
    out["screen_score"] = score
    return out


def to_rows(df: pd.DataFrame, sector_cap_frac: float = 0.3) -> list[UniverseRow]:
    rows: list[UniverseRow] = []
    for _, r in df.iterrows():
        rows.append(
            UniverseRow(
                ticker=str(r.get("ticker", "")),
                exchange=str(r.get("exchange", "")),
                name=str(r.get("name", "") or ""),
                description=str(r.get("description", "") or ""),
                price=float(r.get("close", 0) or 0),
                volume=float(r.get("volume", 0) or 0),
                change_pct=float(r.get("change", 0) or 0),
                rel_volume=float(r.get("relative_volume_10d_calc", 0) or 0),
                market_cap=float(r.get("market_cap_basic", 0) or 0),
                avg_volume=float(r.get("average_volume_10d_calc", 0) or 0),
                dollar_volume=float(r.get("dollar_volume", 0) or 0),
                atr=float(r.get("ATR", 0) or 0),
                atr_pct=float(r.get("atr_pct", 0) or 0),
                rsi=float(r.get("RSI", 0) or 0),
                sector=str(r.get("sector", "") or "Unknown"),
                industry=str(r.get("industry", "") or ""),
                screen_score=float(r.get("screen_score", 0) or 0),
            )
        )
    return rows


def apply_sector_cap(rows: list[UniverseRow], size: int, cap_frac: float = 0.3) -> list[UniverseRow]:
    """Take the top `size` rows while keeping any one sector under cap_frac."""
    cap = max(1, int(np.floor(size * cap_frac)))
    picked: list[UniverseRow] = []
    counts: dict[str, int] = {}
    overflow: list[UniverseRow] = []
    for r in rows:
        if len(picked) >= size:
            break
        if counts.get(r.sector, 0) >= cap:
            overflow.append(r)
            continue
        picked.append(r)
        counts[r.sector] = counts.get(r.sector, 0) + 1
    if len(picked) < size:
        for r in overflow:
            if len(picked) >= size:
                break
            picked.append(r)
    for i, r in enumerate(picked, start=1):
        r.rank = i
    return picked
