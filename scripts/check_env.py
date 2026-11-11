"""Inventory .env keys, map them to connectors, and probe each data source.

Usage:  python scripts/check_env.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402

from tradebot.settings import get_env, load_config  # noqa: E402

OK = "PASS"
NO = "FAIL"


def mask(value: str) -> str:
    if not value:
        return "(empty)"
    return f"{value[:4]}…{value[-3:]} ({len(value)} chars)"


async def probe_finnhub(key: str) -> tuple[bool, str]:
    if not key:
        return False, "no key"
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get("https://finnhub.io/api/v1/quote",
                            params={"symbol": "AAPL", "token": key})
            if r.status_code == 200 and r.json().get("c"):
                return True, f"price {r.json()['c']}"
            return False, f"HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}"


async def probe_marketaux(key: str) -> tuple[bool, str]:
    if not key:
        return False, "no key"
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get("https://api.marketaux.com/v1/news/all",
                            params={"symbols": "AAPL", "limit": 1, "api_token": key})
            if r.status_code == 200:
                return True, f"{len(r.json().get('data', []))} items"
            return False, f"HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}"


async def probe_rss() -> tuple[bool, str]:
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
            r = await c.get("https://news.google.com/rss/search",
                            params={"q": "AAPL stock when:1d", "hl": "en-US", "gl": "US",
                                    "ceid": "US:en"})
            return (r.status_code == 200), f"HTTP {r.status_code}, {len(r.text)} bytes"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}"


async def main() -> int:
    env = get_env()
    cfg = load_config()
    print("=" * 72)
    print("Autotrade environment check")
    print("=" * 72)

    print("\n[IBKR]")
    print(f"  host/port      {env.IB_HOST}:{env.IB_PORT}  clientId {env.IB_CLIENT_ID}")
    print(f"  account        {env.IB_ACCOUNT or '(auto)'}")
    print(f"  mode           {env.TRADING_MODE}  (live unlocked: {bool(env.ALLOW_LIVE)})")
    print(f"  market data    type {env.IB_MARKET_DATA_TYPE} (3 = delayed)")

    print("\n[News keys]")
    checks = []
    fh = await probe_finnhub(env.FINNHUB_API_KEY)
    checks.append(("finnhub", fh))
    print(f"  FINNHUB_API_KEY   {mask(env.FINNHUB_API_KEY):32} {OK if fh[0] else NO}  {fh[1]}")
    mx = await probe_marketaux(env.MARKETAUX_API_KEY)
    checks.append(("marketaux", mx))
    print(f"  MARKETAUX_API_KEY {mask(env.MARKETAUX_API_KEY):32} {OK if mx[0] else NO}  {mx[1]}")
    rss = await probe_rss()
    checks.append(("google_news_rss", rss))
    print(f"  Google News RSS   {'(no key needed)':32} {OK if rss[0] else NO}  {rss[1]}")
    sec = bool(env.SEC_USER_AGENT)
    print(f"  SEC_USER_AGENT    {mask(env.SEC_USER_AGENT):32} {OK if sec else NO}")

    print("\n[Enabled connectors]")
    for name, raw in cfg.news.connectors.items():
        enabled = raw.get("enabled") if isinstance(raw, dict) else raw
        state = enabled
        if enabled == "auto":
            state = "auto→on" if (name == "finnhub" and env.FINNHUB_API_KEY) or (
                name == "marketaux" and env.MARKETAUX_API_KEY) else "auto→off"
        print(f"  {name:20} {state}")

    print("\n[Paths]")
    print(f"  config   {cfg.hash()}")
    print(f"  bars     {cfg.data.bar_minutes}m · providers {cfg.data.bar_provider_order}")
    print(f"  universe N={cfg.universe.size} · interval {cfg.pipeline.interval_minutes}m")

    bad = [n for n, (ok, _) in checks if not ok]
    print("\n" + "-" * 72)
    print(f"{len(checks) - len(bad)}/{len(checks)} sources reachable" +
          (f" · failing: {', '.join(bad)}" if bad else ""))
    return 0 if len(bad) < len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
