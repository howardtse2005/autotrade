"""Feature causality, sentiment point-in-time correctness, news hygiene."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from tradebot.core.clock import now_utc
from tradebot.core.types import RawNewsItem
from tradebot.ingest.market.indicators import compute_features
from tradebot.ingest.news.dedupe import Deduplicator
from tradebot.ingest.news.relevance import build_matchers, match_item
from tradebot.core.types import UniverseRow
from tradebot.sentiment.aggregate import aggregate, build_points
from tradebot.sentiment.finbert import Score
from tradebot.settings import load_config


def _synth_bars(n: int = 600, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-09-01 13:30", periods=n, freq="15min", tz="UTC")
    ret = rng.normal(0, 0.002, n)
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.integers(50_000, 500_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol},
                        index=idx)


def test_features_are_causal():
    """Features at t computed on truncated history must equal the full-history value."""
    bars = _synth_bars()
    full = compute_features(bars, 15)
    for cut in (200, 400, len(bars)):
        trunc = compute_features(bars.iloc[:cut], 15)
        a = full.iloc[cut - 1]
        b = trunc.iloc[-1]
        diff = (a - b).abs()
        bad = diff[diff > 1e-8]
        assert bad.empty, f"look-ahead at cut={cut}: {list(bad.index)}"


def test_intraday_high_low_are_cumulative():
    bars = _synth_bars(200)
    f = compute_features(bars, 15)
    day = bars.index.tz_convert("America/New_York").date
    first_day = day[0]
    mask = day == first_day
    sub = bars[mask]
    last = f[mask].iloc[-1]
    assert last["dist_day_high"] == (sub["close"].iloc[-1] / sub["high"].max() - 1)


def _news(minutes_ago: int, score: float, item_id: str) -> tuple[RawNewsItem, Score]:
    ts = now_utc() - timedelta(minutes=minutes_ago)
    item = RawNewsItem(
        ticker="AAA", title=f"t{item_id}", snippet="", url="", source="s",
        source_type="finnhub", published_at=ts, first_seen_at=ts, relevance=0.9,
        item_id=item_id,
    )
    sc = Score(max(score, 0) + 0.05, max(-score, 0) + 0.05, 0.9)
    return item, sc


def test_sentiment_is_point_in_time():
    cfg = load_config()
    pairs = [_news(5, 0.8, "a"), _news(60, -0.7, "b"), _news(600, 0.3, "c")]
    items = [p[0] for p in pairs]
    scores = {p[0].item_id: p[1] for p in pairs}
    points = build_points(items, scores, cfg)

    early = aggregate(points, pd.DatetimeIndex([pd.Timestamp(now_utc() - timedelta(minutes=500))]),
                      cfg)
    late = aggregate(points, pd.DatetimeIndex([pd.Timestamp(now_utc())]), cfg)
    assert early["news_count_1d"].iloc[0] == 1  # only the 600-minute-old item
    assert late["news_count_1d"].iloc[0] == 3
    assert late["sent_mean_240m"].iloc[0] > 0  # the positive headline dominates


def test_sentiment_ignores_unscored_items():
    cfg = load_config()
    item, _ = _news(5, 0.8, "a")
    points = build_points([item], {}, cfg)
    assert points.empty


def test_dedupe_clusters_similar_titles():
    d = Deduplicator(threshold=90)
    now = now_utc()
    items = [
        RawNewsItem(ticker="AAA", title="Nvidia beats earnings expectations", snippet="", url="u1",
                    source="s", source_type="finnhub", published_at=now, first_seen_at=now,
                    item_id="1"),
        RawNewsItem(ticker="AAA", title="Nvidia beats earnings expectations!", snippet="", url="u2",
                    source="s", source_type="finnhub", published_at=now, first_seen_at=now,
                    item_id="2"),
        RawNewsItem(ticker="BBB", title="Completely different story about rates", snippet="",
                    url="u3", source="s", source_type="finnhub", published_at=now,
                    first_seen_at=now, item_id="3"),
    ]
    unique = d.assign(items)
    assert len(unique) == 2
    assert items[0].cluster_id == items[1].cluster_id
    assert items[0].cluster_size == 2


def test_relevance_requires_company_name_for_ambiguous_ticker():
    universe = [
        UniverseRow(ticker="A", description="Agilent Technologies Inc"),
        UniverseRow(ticker="NVDA", description="NVIDIA Corporation"),
    ]
    m = build_matchers(universe)
    # "A" alone must not match prose
    hits = dict(match_item("A new study shows rates rising", "", m))
    assert "A" not in hits
    hits2 = dict(match_item("NVIDIA raises guidance", "", m))
    assert "NVDA" in hits2


def test_cross_section_ranks_are_bounded():
    from tradebot.features.builder import add_cross_section

    cfg = load_config()
    frame = pd.DataFrame({"ret_4": [0.01, -0.02, 0.03], "rel_volume": [1.0, 2.0, 0.5]})
    out = add_cross_section(frame, cfg)
    assert out["cs_rank_ret_4"].between(0, 1).all()
    assert abs(out["cs_demean_ret_4"].sum()) < 1e-9
