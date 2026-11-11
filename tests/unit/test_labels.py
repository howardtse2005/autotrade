"""Triple-barrier labelling and leakage guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.features.labels import triple_barrier, uniqueness_weights


def _bars(prices: list[tuple[float, float, float, float]], start="2026-11-05 14:30") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(prices), freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [p[0] for p in prices],
            "high": [p[1] for p in prices],
            "low": [p[2] for p in prices],
            "close": [p[3] for p in prices],
            "volume": 1_000,
        },
        index=idx,
    )


def test_take_profit_first():
    # entry at 100, ATR 1 -> TP 101.5, SL 99.0
    bars = _bars([
        (100, 100.2, 99.8, 100),      # decision bar
        (100, 100.5, 99.9, 100.2),    # fill bar (open 100)
        (100.2, 102.0, 100.1, 101.8),  # touches TP
        (101.8, 102.0, 101.0, 101.5),
        (101.5, 101.6, 101.4, 101.5),
        (101.5, 101.6, 101.4, 101.5),
    ])
    atr = pd.Series(1.0, index=bars.index)
    out = triple_barrier(bars, atr, horizon_bars=3, tp_atr=1.5, sl_atr=1.0, lag_bars=0)
    assert out["y_tb"].iloc[0] == 1.0


def test_stop_first():
    bars = _bars([
        (100, 100.2, 99.8, 100),
        (100, 100.1, 98.0, 98.5),
        (98.5, 99.0, 97.0, 97.5),
        (97.5, 97.8, 97.0, 97.2),
        (97.2, 97.5, 97.0, 97.3),
    ])
    atr = pd.Series(1.0, index=bars.index)
    out = triple_barrier(bars, atr, horizon_bars=3, tp_atr=1.5, sl_atr=1.0, lag_bars=0)
    assert out["y_tb"].iloc[0] == -1.0


def test_both_touched_is_stop_first():
    bars = _bars([
        (100, 100.2, 99.8, 100),
        (100, 103.0, 97.0, 100.5),  # both barriers inside one bar
        (100.5, 100.6, 100.4, 100.5),
        (100.5, 100.6, 100.4, 100.5),
    ])
    atr = pd.Series(1.0, index=bars.index)
    out = triple_barrier(bars, atr, horizon_bars=2, tp_atr=1.5, sl_atr=1.0, lag_bars=0)
    assert out["y_tb"].iloc[0] == -1.0


def test_vertical_barrier_is_zero():
    bars = _bars([(100, 100.1, 99.9, 100)] * 6)
    atr = pd.Series(1.0, index=bars.index)
    out = triple_barrier(bars, atr, horizon_bars=3, tp_atr=1.5, sl_atr=1.0, lag_bars=0)
    assert out["y_tb"].iloc[0] == 0.0


def test_entry_uses_next_open_not_current_close():
    """A gap up must be filled at the next open, not at the decision close."""
    bars = _bars([
        (100, 100.1, 99.9, 100),
        (110, 110.5, 109.5, 110),   # gap up: fill here
        (110, 111.0, 109.8, 110.5),
        (110.5, 110.6, 110.4, 110.5),
        (110.5, 110.6, 110.4, 110.5),
    ])
    atr = pd.Series(1.0, index=bars.index)
    out = triple_barrier(bars, atr, horizon_bars=2, tp_atr=1.5, sl_atr=1.0, lag_bars=0,
                         cost_bps=0.0)
    # TP would be 111.5 from a 100 fill; from the real 110 fill it is 111.5 too,
    # but the return must be measured from 110.
    assert out["y_ret"].iloc[0] == pytest.approx(np.log(110.5 / 110), abs=1e-6)


def test_uniqueness_weights_are_positive_and_normalised():
    w = uniqueness_weights(200, horizon=8)
    assert len(w) == 200
    assert (w > 0).all()
    assert w.mean() == pytest.approx(1.0, abs=1e-6)


def test_labels_have_no_future_leak():
    """Labels must be NaN where the horizon runs past the end of the data."""
    bars = _bars([(100, 100.5, 99.5, 100)] * 10)
    atr = pd.Series(1.0, index=bars.index)
    out = triple_barrier(bars, atr, horizon_bars=4, tp_atr=1.5, sl_atr=1.0, lag_bars=1)
    assert out["y_tb"].iloc[-1] != out["y_tb"].iloc[-1]  # NaN
