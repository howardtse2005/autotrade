"""Phase-0 IB Gateway smoke test.

Probes, in order:
  1. connect (managedAccounts, server time drift)
  2. account summary tags (base + USD)
  3. delayed market data (type 3) quotes for AAPL/MSFT
  4. historical bars (15 mins) -> decides bar provider
  5. market depth entitlement probe
  6. contract details (minTick / primaryExchange) incl. a share class
  7. PnL subscription
  8. optional order round-trip (1 share BUY LMT far below market, modify, cancel)

Usage:
    python scripts/smoke_ib.py            # no orders
    python scripts/smoke_ib.py --orders   # also run the order round-trip
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from ib_async import IB, LimitOrder, Stock  # noqa: E402

HOST = os.getenv("IB_HOST", "127.0.0.1")
PORT = int(os.getenv("IB_PORT", "4002"))
CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "17"))

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


async def main(do_orders: bool) -> int:
    ib = IB()
    errors: list[tuple[int, str]] = []
    ib.errorEvent += lambda reqId, code, msg, contract: errors.append((code, msg))

    t0 = time.time()
    try:
        await ib.connectAsync(HOST, PORT, clientId=CLIENT_ID, timeout=20)
    except Exception as exc:  # noqa: BLE001
        record("connect", False, f"{type(exc).__name__}: {exc}")
        return 1
    record("connect", ib.isConnected(), f"{HOST}:{PORT} in {time.time()-t0:.1f}s")

    accounts = ib.managedAccounts()
    record("managedAccounts", bool(accounts), f"{accounts}")

    server_time = await ib.reqCurrentTimeAsync()
    drift = abs((server_time.timestamp() - time.time())) if server_time else float("nan")
    record("server time", finite(drift) and drift < 120, f"drift {drift:.1f}s")

    # ---------------------------------------------------------------- account
    summary = await ib.accountSummaryAsync()
    tags = {(r.tag, r.currency): r.value for r in summary}
    base = next((r.currency for r in summary if r.tag == "NetLiquidation"), "?")
    wanted = ["NetLiquidation", "TotalCashValue", "AvailableFunds", "BuyingPower",
              "GrossPositionValue", "ExcessLiquidity", "InitMarginReq", "MaintMarginReq"]
    found = {t: tags.get((t, "USD")) or tags.get((t, base)) for t in wanted}
    record("accountSummary", any(v not in (None, "") for v in found.values()),
           f"base={base} " + " ".join(f"{k}={v}" for k, v in found.items() if v))

    # ------------------------------------------------------- delayed quotes
    ib.reqMarketDataType(3)
    contracts = await ib.qualifyContractsAsync(
        Stock("AAPL", "SMART", "USD"), Stock("MSFT", "SMART", "USD")
    )
    record("qualify AAPL/MSFT", len(contracts) == 2,
           f"{[(c.symbol, c.conId, c.primaryExchange) for c in contracts]}")

    tickers = await ib.reqTickersAsync(*contracts)
    await asyncio.sleep(3)
    for t in tickers:
        g = lambda name: getattr(t, name, None)  # noqa: E731
        bid = g("bid") if finite(g("bid")) and g("bid") > 0 else g("delayedBid")
        ask = g("ask") if finite(g("ask")) and g("ask") > 0 else g("delayedAsk")
        last = g("last") if finite(g("last")) and g("last") > 0 else g("delayedLast")
        record(f"quote {t.contract.symbol}", finite(last) or (finite(bid) and finite(ask)),
               f"bid={bid} ask={ask} last={last} delayedBid={g('delayedBid')} "
               f"delayedLast={g('delayedLast')} mdt={g('marketDataType')}")

    # ------------------------------------------------------ historical bars
    bar_ok = False
    try:
        bars = await ib.reqHistoricalDataAsync(
            contracts[0], endDateTime="", durationStr="5 D", barSizeSetting="15 mins",
            whatToShow="TRADES", useRTH=True, formatDate=2, timeout=60,
        )
        bar_ok = len(bars) > 0
        record("historical 15m bars", bar_ok,
               f"{len(bars)} bars; last={bars[-1].date if bars else None} close={bars[-1].close if bars else None}")
    except Exception as exc:  # noqa: BLE001
        record("historical 15m bars", False, f"{type(exc).__name__}: {exc}")

    # --------------------------------------------------------------- depth
    depth_ok = False
    try:
        t = ib.reqMktDepth(contracts[0], numRows=5)
        await asyncio.sleep(4)
        dom = t.domBids + t.domAsks
        depth_ok = len(dom) > 0
        ib.cancelMktDepth(contracts[0])
        record("market depth", depth_ok, f"{len(dom)} rows")
    except Exception as exc:  # noqa: BLE001
        record("market depth", False, f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------- contract details
    details = await ib.reqContractDetailsAsync(contracts[0])
    if details:
        cd = details[0]
        record("contract details", True,
               f"minTick={cd.minTick} primary={cd.contract.primaryExchange} "
               f"tz={cd.timeZoneId} tradingHours_len={len(cd.tradingHours or '')}")
    else:
        record("contract details", False, "none")

    # -------------------------------------------------------------- PnL
    pnl_ok = False
    if accounts:
        pnl = ib.reqPnL(accounts[0])
        await asyncio.sleep(3)
        pnl_ok = pnl is not None
        record("reqPnL", pnl_ok, f"daily={getattr(pnl, 'dailyPnL', None)} "
                                 f"unreal={getattr(pnl, 'unrealizedPnL', None)}")
        ib.cancelPnL(accounts[0])

    # ------------------------------------------------------- order round trip
    if do_orders:
        try:
            contract = contracts[1]  # MSFT
            t = await ib.reqTickersAsync(contract)
            last = t[0].last or t[0].delayedLast or t[0].close or 400.0
            px = round(max(last * 0.5, 1.0), 2)
            order = LimitOrder("BUY", 1, px, tif="DAY", orderRef="smoke:probe")
            trade = ib.placeOrder(contract, order)
            await asyncio.sleep(3)
            status1 = trade.orderStatus.status
            order.lmtPrice = round(px * 0.98, 2)
            ib.placeOrder(contract, order)
            await asyncio.sleep(2)
            ib.cancelOrder(order)
            await asyncio.sleep(2)
            record("order round-trip", status1 not in ("Cancelled", "ApiCancelled"),
                   f"placed status={status1} -> modified -> cancelled status={trade.orderStatus.status} "
                   f"(read-only API must be OFF)")
        except Exception as exc:  # noqa: BLE001
            record("order round-trip", False, f"{type(exc).__name__}: {exc}")

    ib.disconnect()

    print("\n--- IB messages (filtered) ---")
    for code, msg in errors[:40]:
        print(f"  {code}: {msg[:160]}")

    ok = sum(1 for _, o, _ in results if o)
    print(f"\n{ok}/{len(results)} checks passed")
    return 0 if ok == len(results) else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", action="store_true", help="run the order round-trip")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args.orders)))
