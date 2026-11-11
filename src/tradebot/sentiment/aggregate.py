"""Aggregate scored news into bar-aligned sentiment features (point-in-time safe)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from tradebot.core.types import RawNewsItem
from tradebot.sentiment.finbert import Score

DEFAULT_WINDOWS = (60, 240, 1440, 4320)


def _source_weights(cfg) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, raw in (cfg.news.connectors or {}).items():
        if isinstance(raw, dict):
            out[name] = float(raw.get("weight", 0.8))
    return out


def build_points(
    items: list[RawNewsItem],
    scores: dict[str, Score],
    cfg,
) -> pd.DataFrame:
    """One row per news item with availability time, score and base weight."""
    src_w = _source_weights(cfg)
    rows = []
    for it in items:
        if not it.item_id:
            continue
        sc = scores.get(it.item_id)
        if sc is None:
            continue
        base = src_w.get(it.source_type, 0.7) * float(it.relevance or 0.5)
        base /= math.sqrt(max(1, it.cluster_size))
        rows.append(
            {
                "available_at": it.available_at,
                "published_at": it.published_at,
                "score": sc.value,
                "p_pos": sc.p_pos,
                "p_neg": sc.p_neg,
                "p_neu": sc.p_neu,
                "weight": max(base, 1e-4),
                "cluster_id": it.cluster_id or it.item_id,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["available_at", "score", "p_pos", "p_neg", "p_neu", "weight", "cluster_id"]
        )
    df = pd.DataFrame(rows).sort_values("available_at").reset_index(drop=True)
    return df


def _empty_features(index: pd.DatetimeIndex, windows=DEFAULT_WINDOWS) -> pd.DataFrame:
    cols = []
    for w in windows:
        tag = _tag(w)
        cols += [
            f"sent_mean_{tag}", f"sent_pos_{tag}", f"sent_neg_{tag}", f"sent_neu_{tag}",
            f"sent_max_neg_{tag}", f"sent_max_pos_{tag}", f"sent_std_{tag}",
            f"news_count_{tag}", f"news_unique_{tag}", f"news_neg_{tag}",
        ]
    cols += ["sent_ewm_240m", "news_count_z", "sent_delta_60_240", "mins_since_last_news",
             "has_news_24h", "news_pending_frac"]
    out = pd.DataFrame(index=index, columns=cols, dtype=float)
    out["news_count_z"] = 0.0
    out["has_news_24h"] = 0.0
    out["news_pending_frac"] = 0.0
    return out


TAGS = {60: "60m", 240: "240m", 1440: "1d", 4320: "3d"}


def _tag(minutes: int) -> str:
    if minutes in TAGS:
        return TAGS[minutes]
    if minutes % 1440 == 0:
        return f"{minutes // 1440}d"
    return f"{minutes}m"


def aggregate(
    points: pd.DataFrame,
    bar_index: pd.DatetimeIndex,
    cfg,
    pending_frac: float = 0.0,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Weighted, decayed sentiment features for every bar timestamp."""
    half_life = float(cfg.sentiment.half_life_minutes)
    feats = _empty_features(bar_index, windows)
    if points.empty or len(bar_index) == 0:
        feats["news_pending_frac"] = pending_frac
        return feats

    avail = points["available_at"].to_numpy(dtype="datetime64[ns]")
    score = points["score"].to_numpy(dtype=float)
    p_pos = points["p_pos"].to_numpy(dtype=float)
    p_neg = points["p_neg"].to_numpy(dtype=float)
    p_neu = points["p_neu"].to_numpy(dtype=float)
    weight = points["weight"].to_numpy(dtype=float)
    cluster = points["cluster_id"].to_numpy()

    bars = bar_index.values.astype("datetime64[ns]")
    hl_ns = half_life * 60 * 1e9
    ref = avail[0]

    for i, t in enumerate(bars):
        for w in windows:
            lo = np.searchsorted(avail, t - np.timedelta64(w * 60, "s"), side="right")
            hi = np.searchsorted(avail, t, side="right")
            tag = _tag(w)
            if hi <= lo:
                continue
            sl = slice(lo, hi)
            dt = (t - avail[sl]).astype("timedelta64[s]").astype(float)
            decay = np.exp(-dt / (half_life * 60.0))
            wt = weight[sl] * decay
            tot = wt.sum()
            if tot <= 0:
                continue
            feats.iat[i, feats.columns.get_loc(f"sent_mean_{tag}")] = float((wt * score[sl]).sum() / tot)
            feats.iat[i, feats.columns.get_loc(f"sent_pos_{tag}")] = float((wt * p_pos[sl]).sum() / tot)
            feats.iat[i, feats.columns.get_loc(f"sent_neg_{tag}")] = float((wt * p_neg[sl]).sum() / tot)
            feats.iat[i, feats.columns.get_loc(f"sent_neu_{tag}")] = float((wt * p_neu[sl]).sum() / tot)
            feats.iat[i, feats.columns.get_loc(f"sent_max_neg_{tag}")] = float(score[sl].min())
            feats.iat[i, feats.columns.get_loc(f"sent_max_pos_{tag}")] = float(score[sl].max())
            mu = float((wt * score[sl]).sum() / tot)
            var = float((wt * (score[sl] - mu) ** 2).sum() / tot)
            feats.iat[i, feats.columns.get_loc(f"sent_std_{tag}")] = math.sqrt(max(var, 0.0))
            feats.iat[i, feats.columns.get_loc(f"news_count_{tag}")] = float(hi - lo)
            feats.iat[i, feats.columns.get_loc(f"news_unique_{tag}")] = float(len(set(cluster[sl])))
            feats.iat[i, feats.columns.get_loc(f"news_neg_{tag}")] = float((score[sl] <= -0.6).sum())

        # full-history exponential average (no window cap)
        hi = np.searchsorted(avail, t, side="right")
        if hi > 0:
            dt = (t - avail[:hi]).astype("timedelta64[s]").astype(float)
            decay = np.exp(-dt / (half_life * 60.0))
            wt = weight[:hi] * decay
            tot = wt.sum()
            if tot > 0:
                feats.iat[i, feats.columns.get_loc("sent_ewm_240m")] = float((wt * score[:hi]).sum() / tot)
            last = avail[hi - 1]
            feats.iat[i, feats.columns.get_loc("mins_since_last_news")] = float(
                (t - last).astype("timedelta64[s]").astype(float) / 60.0
            )
            feats.iat[i, feats.columns.get_loc("has_news_24h")] = 1.0
        del ref

    feats["sent_delta_60_240"] = feats["sent_mean_60m"] - feats["sent_mean_240m"]
    counts = feats[f"news_count_{_tag(1440)}"].fillna(0.0)
    baseline = counts.rolling(20 * 26, min_periods=20).mean()
    std = counts.rolling(20 * 26, min_periods=20).std().replace(0, np.nan)
    feats["news_count_z"] = ((counts - baseline) / std).fillna(0.0).clip(-5, 5)
    feats["news_pending_frac"] = pending_frac
    return feats


def latest_features(
    items: list[RawNewsItem],
    scores: dict[str, Score],
    bar_ts: datetime,
    cfg,
    pending_frac: float = 0.0,
) -> dict[str, float]:
    """Fast path for live inference: only the most recent bar."""
    points = build_points(items, scores, cfg)
    idx = pd.DatetimeIndex([pd.Timestamp(bar_ts)])
    feats = aggregate(points, idx, cfg, pending_frac=pending_frac)
    row = feats.iloc[0].to_dict()
    return {k: (float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else np.nan)
            for k, v in row.items()}


def window_labels(windows: tuple[int, ...] = DEFAULT_WINDOWS) -> list[str]:
    out = []
    for w in windows:
        tag = _tag(w)
        out += [
            f"sent_mean_{tag}", f"sent_pos_{tag}", f"sent_neg_{tag}", f"sent_neu_{tag}",
            f"sent_max_neg_{tag}", f"sent_max_pos_{tag}", f"sent_std_{tag}",
            f"news_count_{tag}", f"news_unique_{tag}", f"news_neg_{tag}",
        ]
    return out + ["sent_ewm_240m", "news_count_z", "sent_delta_60_240",
                  "mins_since_last_news", "has_news_24h", "news_pending_frac"]
