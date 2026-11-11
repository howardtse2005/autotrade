"""Reconcile DB positions with IBKR (IB is the source of truth)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from tradebot.core.clock import now_utc
from tradebot.core.logging import get_logger
from tradebot.core.symbols import canonical, from_ib_symbol
from tradebot.db.models import PositionRow, Trade
from tradebot.db.session import session_scope

log = get_logger("reconcile")


async def reconcile(ib_client, run_id: int | None = None) -> dict:
    """Sync `positions` with IBKR. Returns a summary of the differences found."""
    if not ib_client.connected:
        return {"ok": False, "reason": "not connected"}

    ib_positions = {
        from_ib_symbol(p.contract.symbol): (float(p.position), float(p.avgCost))
        for p in ib_client.positions()
    }
    open_orders = await ib_client.open_orders()
    protective: dict[str, dict] = {}
    for trade in open_orders:
        ref = trade.order.orderRef or ""
        ticker = from_ib_symbol(trade.contract.symbol)
        if ref.endswith(":sl"):
            protective.setdefault(ticker, {})["sl_id"] = trade.order.orderId
        elif ref.endswith(":tp"):
            protective.setdefault(ticker, {})["tp_id"] = trade.order.orderId

    # orders that the DB still thinks are working but IB no longer has
    from tradebot.db.models import OrderRow

    live_ids = {t.order.orderId for t in open_orders}
    stale = 0
    async with session_scope() as s:
        rows = (await s.execute(
            select(OrderRow).where(OrderRow.status.in_(
                ["PreSubmitted", "Submitted", "PendingSubmit", "PendingCancel"]))
        )).scalars().all()
        for r in rows:
            if r.ib_order_id not in live_ids:
                r.status = "Inactive"
                stale += 1

    created = updated = closed = 0
    async with session_scope() as s:
        rows = (
            await s.execute(select(PositionRow).where(PositionRow.status == "open"))
        ).scalars().all()
        db_by_ticker = {canonical(r.ticker): r for r in rows}

        for ticker, (qty, avg_cost) in ib_positions.items():
            if abs(qty) < 1e-9:
                continue
            row = db_by_ticker.get(ticker)
            if row is None:
                s.add(
                    PositionRow(
                        ticker=ticker, qty=qty, avg_cost=avg_cost, entry_ts=now_utc(),
                        status="open", external=True, sector="",
                        entry_snapshot={"source": "ib_reconcile"},
                    )
                )
                created += 1
            else:
                if abs(row.qty - qty) > 1e-9 or abs(row.avg_cost - avg_cost) > 1e-6:
                    row.qty = qty
                    row.avg_cost = avg_cost
                    updated += 1
                prot = protective.get(ticker, {})
                if prot.get("sl_id") and row.ib_stop_id != prot["sl_id"]:
                    row.ib_stop_id = prot["sl_id"]
                if prot.get("tp_id") and row.ib_tp_id != prot["tp_id"]:
                    row.ib_tp_id = prot["tp_id"]

        for ticker, row in db_by_ticker.items():
            if ticker in ib_positions:
                continue
            # position gone at IB: close it in the DB
            exit_price = row.avg_cost
            pnl = 0.0
            if row.qty and row.avg_cost:
                pnl = 0.0
            row.status = "closed"
            row.closed_at = now_utc()
            s.add(
                Trade(
                    position_id=row.id, ticker=row.ticker, qty=row.qty,
                    entry_ts=row.entry_ts, entry_price=row.avg_cost,
                    exit_ts=now_utc(), exit_price=exit_price, pnl=pnl,
                    exit_reason="reconcile", hold_bars=row.bars_held,
                    entry_snapshot=row.entry_snapshot,
                )
            )
            closed += 1

    summary = {"ok": True, "created": created, "updated": updated, "closed": closed,
               "stale_orders": stale,
               "ib_positions": len(ib_positions), "protective": protective}
    if created or closed or updated:
        log.info("reconciled", **{k: v for k, v in summary.items() if k != "protective"})
    return summary


async def positions_needing_protection(ib_client) -> list[dict]:
    """Open long positions without a live stop or target."""
    if not ib_client.connected:
        return []
    open_orders = await ib_client.open_orders()
    have_sl, have_tp = set(), set()
    for trade in open_orders:
        ref = trade.order.orderRef or ""
        ticker = from_ib_symbol(trade.contract.symbol)
        status = trade.orderStatus.status
        if status in ("Filled", "Cancelled", "ApiCancelled", "Inactive"):
            continue
        if ref.endswith(":sl"):
            have_sl.add(ticker)
        elif ref.endswith(":tp"):
            have_tp.add(ticker)

    out = []
    for p in ib_client.positions():
        ticker = from_ib_symbol(p.contract.symbol)
        if p.position <= 0:
            continue
        missing = []
        if ticker not in have_sl:
            missing.append("sl")
        if ticker not in have_tp:
            missing.append("tp")
        if missing:
            out.append({"ticker": ticker, "qty": float(p.position),
                        "avg_cost": float(p.avgCost), "missing": missing})
    return out


def entry_dt(value) -> datetime | None:
    return value
