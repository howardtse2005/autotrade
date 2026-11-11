"""Train the Tri-Lens model from cached/refreshed bars and register it.

Usage:
    python scripts/train.py [--quick] [--universe 50] [--no-fetch]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradebot.core.logging import setup_logging  # noqa: E402
from tradebot.db.session import init_db  # noqa: E402
from tradebot.features.builder import build_panel  # noqa: E402
from tradebot.ingest.market.providers import BarService  # noqa: E402
from tradebot.ingest.universe.tradingview import UniverseService  # noqa: E402
from tradebot.models.train import train_and_register  # noqa: E402
from tradebot.settings import load_config  # noqa: E402


async def main(args: argparse.Namespace) -> int:
    setup_logging()
    await init_db()
    cfg = load_config()
    if args.universe:
        cfg.universe.size = args.universe

    us = UniverseService(cfg)
    rows = await us.get(force=True)
    tickers = [r.ticker for r in rows]
    print(f"universe: {len(tickers)} tickers from {us.last_source}")

    bars_service = BarService(cfg)
    if not args.no_fetch:
        await bars_service.get_bars(tickers, force=True)
        await bars_service.get_bars(
            tickers + cfg.universe.benchmarks + ["^VIX"],
            bar_minutes=1440, kind="daily", days=cfg.data.daily_lookback_days, force=True,
        )

    bars = {t: bars_service.cached_bars(t) for t in tickers}
    daily = {t: bars_service.cached_bars(t, 1440, "daily") for t in tickers}
    spy = bars_service.cached_bars(cfg.universe.benchmarks[0], 1440, "daily")
    vix = bars_service.cached_bars("^VIX", 1440, "daily")

    t0 = time.time()
    panel = build_panel(bars, cfg, daily_by_ticker=daily, spy=spy, vix=vix)
    print(f"panel: {panel.frame.shape[0]} rows x {len(panel.feature_cols)} features "
          f"in {time.time() - t0:.1f}s")
    if panel.frame.empty:
        print("ERROR: no training rows")
        return 1

    rep = await train_and_register(panel, cfg)
    if rep.status != "ok":
        print(f"ERROR: {rep.message}")
        return 1
    print(f"\nmodel {rep.model_id}: rows={rep.rows} features={rep.features} "
          f"time={rep.duration_s}s stack_average={rep.use_average}")
    for k, v in rep.metrics.items():
        if isinstance(v, float):
            print(f"  {k:22} {v:.4f}")
    print(f"  folds                  {len(rep.folds)}")
    print("\ntop features:")
    for name, gain in list(rep.feature_importance.items())[:10]:
        print(f"  {name:24} {gain:.4f}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--universe", type=int, default=0)
    ap.add_argument("--no-fetch", action="store_true")
    raise SystemExit(asyncio.run(main(ap.parse_args())))
